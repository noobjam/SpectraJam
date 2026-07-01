# SpectraJam

SpectraJam is a family of regional Earth-observation encoders forged from
TESSERA's global pixel embeddings. The initial family members are:

- `spectrajam-rw-student` and `spectrajam-rw-lora`
- `spectrajam-il-student` and `spectrajam-il-lora`
- `spectrajam-rwil-student` and `spectrajam-rwil-lora`, the shared-country controls

The name is literal: Sentinel-2 spectra, Sentinel-1 radar, and irregular time
are "jammed" into one embedding space. It is playful enough to name a model
family while still describing the fusion idea. This repository builds and
tests those encoders; it is not a downstream crop classifier.

Embeddings are not restricted to calendar years. Annual files remain efficient
storage shards, while the model contract accepts arbitrary half-open windows,
rolling 7/14-day views, and cumulative coverage prefixes. See
[Arbitrary-window embeddings](docs/WINDOWS.md) for the causal training and
incremental-inference contract.

## Scientific position

Regional adaptation is a hypothesis, not a foregone conclusion. The TESSERA
paper reports negligible improvement from regional retraining and no benefit
from downstream encoder fine-tuning in its Austrian crop ablation. Every run is
therefore compared with one immutable frozen-TESSERA baseline.

The two adaptation tracks are deliberately different:

- Teacher-student produces a standalone compact 128-dimensional encoder. It
  combines teacher alignment with relational geometry and regional two-view
  self-supervision; teacher matching alone would only compress TESSERA.
- LoRA keeps TESSERA as the runtime and learns a small country adapter. It uses
  Q/V-aware parametrization for PyTorch's packed `in_proj_weight`, optionally
  adds FFN adapters, and is zero-output at initialization.

See [Design](docs/DESIGN.md) for the experimental matrix and [Research](docs/RESEARCH.md)
for the evidence behind the defaults.

## Non-negotiable compatibility gate

The existing sibling TesseraCrop implementation is not used as a model source.
Its band order, normalization, and topology do not match the current official
checkpoint contract. SpectraJam instead pins upstream TESSERA commit
`d06ee44a053246db3e73f104403f6eaf642e1abf` and enforces:

- S2 order: `B04,B02,B03,B08,B8A,B05,B06,B07,B11,B12`
- S1 order: `VV,VH`
- source-specific checkpoint and normalization pairing
- exact checkpoint keys and tensor shapes; no permissive loading
- an official-output parity fixture before training

## Current milestone

This first build contains the experiment contracts, deterministic sampling,
spatial/temporal split matrix, STAC work-tile discovery, durable retry/resume
ledger, checkpoint-faithful v1.1 graph, compact student, packed-Q/V LoRA,
arbitrary-window training, a mask-aware TESSERA runtime, and an exact
cache/recompute incremental builder. Candidate-frame raster construction,
upstream-compatible COG materialization, a durable embedding cache, and the
distributed runner are the next milestone. Capturing the official parity
fixture is currently a manual prerequisite. The repository does not pretend
those unfinished pieces are production-ready.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[data,train,dev]"
pytest
```

Validate an experiment contract:

```bash
spectrajam validate-config --config configs/pilot.yaml
```

That command validates the template structure and rejects provider/checkpoint
mixing. After replacing boundary/checkpoint paths and hashes, use the operational
gate, which reads and hashes the actual files:

```bash
spectrajam validate-config --config configs/pilot.yaml --operational
```

Sampling is always operational: it refuses placeholder boundary files or wrong
checksums and checks every candidate against the pinned country geometry. The
pilot begins from a candidate CSV whose required columns are
`candidate_id,country,longitude,latitude,spatial_block,stratum`. Candidate IDs
must be stable, spatial blocks must be assigned before sampling, and `stratum`
is the pinned `ecoregion × WorldCover` key.

```bash
spectrajam sample \
  --config configs/pilot.yaml \
  --candidates data/candidates.csv \
  --output data/manifests/pilot.csv
```

The output contains inclusion probability plus the complete spatial-split ×
year-split matrix. This makes
four evaluations possible without redefining the data: ordinary train,
spatial-only holdout, temporal-only holdout, and combined spatial-temporal
holdout.

Initialize the acquisition ledger:

```bash
spectrajam ledger-init \
  --config configs/pilot.yaml \
  --manifest data/manifests/pilot.csv \
  --database data/state/acquisition.sqlite

spectrajam ledger-status --database data/state/acquisition.sqlite
spectrajam ledger-assert-complete --database data/state/acquisition.sqlite
```

Ledger initialization revalidates both countries, every configured year per
candidate, stable IDs, and the expected pilot point count before binding the
database to manifest and config hashes.

Training must not start unless `ledger-assert-complete` passes. A point is never
silently replaced after a failed download. The trainer constructors also require
both a completed ledger and a parity receipt unless a test explicitly opts into
the unsafe unverified mode.

Window training additionally requires absolute observation-day tensors and
coverage bounds. Its frozen teacher/reference must encode the same selected
window; passing a precomputed annual target for a 7-day crop window is rejected
by the trainer contract.

After manually recording a fixture with the pinned official code and exact MPC
preprocessing, verify model parity with:

```bash
spectrajam verify-upstream-parity \
  --config configs/pilot.yaml \
  --fixture tests/fixtures/tessera_v11_mpc_parity.npz \
  --receipt artifacts/parity/tessera_v11_mpc.json
```

## Dataset tiers

| Tier | Spatial anchors | Years | Point-years | Purpose |
|---|---:|---|---:|---|
| Smoke | 128/country | 2023–2025 | 768 | Contract and code checks |
| Pilot | 25,000/country | 2019, 2021, 2023–2025 | 250,000 | End-to-end proof and early curves |
| Preferred full | complete 200 m lattice, about 0.55–0.65M/country | 2019–2025 | about 7.7–9.1M | Closest regional analogue to TESSERA |

The full count is finalized only after applying the pinned land boundary and
mask. Learning curves at 5k, 25k, 75k, 150k, and full anchors per country decide
whether the full lattice earns its cost.
