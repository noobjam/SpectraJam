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

Embeddings are not restricted to calendar years. Annual Parquet partitions are
the planned storage shards, while the model contract accepts arbitrary
half-open windows, rolling 7/14-day views, and cumulative coverage prefixes. See
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

This build contains the experiment contracts, deterministic sampling,
spatial/temporal split matrix, content-addressed STAC discovery, durable
retry/resume ledger, checkpoint-faithful v1.1 graph, compact student,
packed-Q/V LoRA, arbitrary-window training, a mask-aware TESSERA runtime, and
an exact cache/recompute incremental builder. It also implements the primitives
for a checksum-pinned checkpoint fetcher, exact MPC raw-value transforms,
immutable Parquet/Zstd point shards, and a synthetic execution smoke for both
adaptation tracks. The real-data frame path now pins, retries, and verifies the
World Bank v2 ADM0/NDLSA release, RESOLVE Ecoregions 2017, the ESA WorldCover
2021 grid, and only the five country-covering WorldCover COGs. It streams a
fixed-origin 200 m lattice in each pinned country UTM CRS into the sampler
contract and writes a full provenance receipt. The smoke-only sparse Sentinel
COG-to-point materializer is now implemented with catalog preflight, touched
block reads, fresh signing/retry, immutable publication, and explicit ledger
outcomes. Its first real VM run, the official parity fixture, batched
pilot/full shards, a durable embedding cache, and the distributed training
runner remain the next milestones.

For the exact verified VM state, remaining gaps, and continuation order, see
[Session handoff](docs/HANDOFF.md).

The independent frozen-model baseline in
[`plain_tessera_incremental`](plain_tessera_incremental/README.md) is deliberately
outside the regional LoRA/distillation track. It rasterizes labelled WKT fields
to 10 m pixels, acquires MPC S1/S2 observations through STAC, and emits four
cumulative plain-TESSERA embedding prefixes for the requested Harvard dataset.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate

# Ignore corporate/global pip configuration for this shell and use public PyPI.
export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL

python -m pip install --upgrade pip
python -m pip install -e ".[data,train,dev]"
pytest
```

If public PyPI still reports a proxy error after these overrides, the VM itself
does not currently have direct PyPI egress; changing pip configuration alone
cannot bypass that network policy.

Fetch the pinned 220 MiB TESSERA v1.1 MPC encoder. The command streams into a
resumable `.part`, verifies its byte count and SHA-256, fsyncs it, and only then
atomically installs it. A valid existing file is reused; an invalid destination
is never silently replaced.

```bash
spectrajam fetch-checkpoint --config configs/smoke.yaml
spectrajam validate-config --config configs/smoke.yaml --require-checkpoint
```

Then execute one optimizer update for each model track on deterministic
normalized synthetic windows: 7 days for Teacher-Student and 14 days for LoRA.
This proves checkpoint loading, forward/backward, masking, and adapter wiring;
it is explicitly not a model-quality experiment.

```bash
spectrajam model-smoke --config configs/smoke.yaml --device cuda:0
```

Validate an experiment contract:

```bash
spectrajam validate-config --config configs/pilot.yaml
```

That command validates the template structure and rejects provider/checkpoint
mixing. Fetch the pinned frame inputs before using the operational gate. The
download is resumable, fail-closed on byte count and SHA-256, and stores about
506 MB of source data rather than a global land-cover raster:

```bash
spectrajam fetch-frame-sources --source-root data/frame
spectrajam validate-config --config configs/pilot.yaml --operational
```

Build the deterministic Rwanda+Israel candidate frame. This reads local COGs
asset-major in bounded chunks, excludes WorldCover nodata and permanent water,
and attaches the numeric RESOLVE `ECO_ID × WorldCover` primary stratum:

```bash
spectrajam candidate-frame \
  --config configs/smoke.yaml \
  --source-root data/frame \
  --output data/candidates.csv \
  --receipt data/candidates.receipt.json
```

The command also verifies the current World Bank ADM0 package itself. That
package excludes NDLSA by construction; exact `ISO_A3` selection additionally
keeps the separate West Bank and Gaza territory outside the Israel frame.
Sampling remains operational: it refuses missing or wrong-checksum boundaries
and checks every candidate against its exact country feature. Its required columns are
`candidate_id,country,longitude,latitude,spatial_block,stratum`. Candidate IDs
must be stable, spatial blocks must be assigned before sampling, and `stratum`
is the pinned `ecoregion × WorldCover` key.

```bash
spectrajam sample \
  --config configs/pilot.yaml \
  --candidates data/candidates.csv \
  --candidate-receipt data/candidates.receipt.json \
  --output data/manifests/pilot.csv
```

The command verifies the candidate byte count and SHA-256 against its frame
receipt, then writes a sampling receipt beside the manifest. The output contains
inclusion probability plus the complete spatial-split × year-split matrix. This makes
four evaluations possible without redefining the data: ordinary train,
spatial-only holdout, temporal-only holdout, and combined spatial-temporal
holdout.

Discover every STAC page once per spatial work block and year. Completed query
snapshots are immutable and safely reused after interruption; repeated raw STAC
items are stored once by content hash rather than copied into every query.
Library consumers use `read_catalog_snapshot` to validate those query/item
hashes and receive item references with timezone-aware acquisition datetimes.

```bash
spectrajam catalog-discover \
  --config configs/smoke.yaml \
  --manifest data/manifests/smoke.csv \
  --output data/catalog
```

The implemented observation-format contract stores only ragged point histories:
S2 as exact `uint16[10]` plus SCL, and S1 as exact scaled-dB `int16[2]` plus
orbit. Every row binds the source-item document and catalog-query hashes. The
smoke materializer reads touched COG blocks asset-major across a country/year,
signs each canonical MPC href again on every retry, publishes one immutable
verified shard per point-year-modality, and commits its outcome to SQLite only
after publication. It never persists imagery cubes.

Initialize the acquisition ledger:

```bash
spectrajam ledger-init \
  --config configs/smoke.yaml \
  --manifest data/manifests/smoke.csv \
  --sampling-receipt data/manifests/smoke.csv.receipt.json \
  --database data/state/smoke-acquisition.sqlite

spectrajam ledger-status --database data/state/smoke-acquisition.sqlite
spectrajam ledger-assert-complete --database data/state/smoke-acquisition.sqlite
```

Ledger initialization revalidates both countries, every configured year per
candidate, stable IDs, and the expected smoke point count before binding the
database to sampling-receipt, manifest, and config hashes.

Materialize the real smoke point histories as a detached, resumable job. The
command first validates every catalog/item checksum and output publication,
then immutably binds both that catalog inventory and the exact
materializer/preprocessing/point-store/runtime contract to the existing ledger
before it claims a task. It also signs and opens representative S1/S2 COGs, so
missing MPC data-plane access cannot poison ledger tasks. `RUNNING` is replaced
by the numeric process exit code.

```bash
mkdir -p runs/logs
printf '%s\n' RUNNING > runs/smoke-materialize.exit

nohup bash -c '
  rc=0
  .venv/bin/spectrajam materialize \
    --config configs/smoke.yaml \
    --manifest data/manifests/smoke.csv \
    --sampling-receipt data/manifests/smoke.csv.receipt.json \
    --catalog-root data/catalog \
    --database data/state/smoke-acquisition.sqlite \
    --output data/pointstore/smoke \
  || rc=$?
  printf "%s\n" "$rc" > runs/smoke-materialize.exit
  exit "$rc"
' > runs/logs/smoke-materialize.log 2>&1 < /dev/null &
```

Use `tail -f runs/logs/smoke-materialize.log` for progress and
`spectrajam ledger-status --database data/state/smoke-acquisition.sqlite` for
durable counts. The current per-point publication strategy is deliberately
gated to `stage: smoke`; pilot/full acquisition first needs batched Parquet
shards to avoid millions of tiny files.

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
| Preferred full | 593,292 RWA + 506,429 ISR reference anchors | 2019–2025 | 7,698,047 | Closest regional analogue to TESSERA |

These full counts come from the pinned 2026-07-02 reference frame and must be
reproduced from its receipt on the VM. Learning curves at 5k, 25k, 75k, 150k,
and full anchors per country decide whether the full lattice earns its cost.
