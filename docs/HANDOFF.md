# SpectraJam session handoff

Last updated: 2026-07-01  
Repository: <https://github.com/noobjam/SpectraJam>  
Branch: `main`  
Implementation baseline before this handoff: `36c30ef`

## Current outcome

SpectraJam is a clean regional-model repository for Rwanda (`RWA`) and Israel
(`ISR`). It contains both requested adaptation tracks:

- a standalone compact Teacher-Student encoder;
- LoRA adapters over the pinned TESSERA v1.1 encoder;
- Rwanda-only, Israel-only, and shared-country experiment contracts.

The model contract supports arbitrary half-open time windows rather than only
calendar years. Seven- and fourteen-day windows are explicit training and
evaluation anchors. Calendar years are intended to remain storage partitions,
not the semantic definition of an embedding.

No regional model has been scientifically trained yet. The real checkpoint and
CUDA execution paths have been proven, but their inputs were deterministic
synthetic observations. The next milestone is the real sparse-data path.

## Verified VM state

The following was verified on the training VM on 2026-07-01:

- 8 × NVIDIA H100 80 GB HBM3;
- all-to-all `NV18` GPU connectivity in one NUMA domain;
- public-PyPI installation works after overriding the corporate pip index;
- the full suite passes: `112 passed`;
- the official encoder was downloaded, hashed, strictly loaded, and executed
  on `cuda:0`;
- both adaptation tracks completed a real forward/backward/optimizer update.

The filesystem observation at setup time was 21 TB total, 98% used, with about
536 GB free. Treat that value as volatile. It is sufficient for sparse point
histories, but not for persistent country-scale raster cubes.

### Verified checkpoint

```text
path:       checkpoints/tessera_v1_1_mpc_encoder.pt
bytes:      230891229
sha256:     5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3
HF revision: e037fc62cd196f9e05dde4c4104e1383541b41c5
upstream:   d06ee44a053246db3e73f104403f6eaf642e1abf
parameters: 57708738
output dim: 128
```

The SHA-256 is an independently observed release hash; upstream does not
publish a checksum. SpectraJam refuses to accept a replacement with another
digest.

### Verified CUDA model smoke

```text
Teacher-Student:
  window:                    7 days
  trainable parameters:      4,185,988
  updated parameter tensors: 80

LoRA:
  window:                    14 days
  trainable parameters:      303,616
  installed targets:         16
  updated parameter tensors: 52
```

This smoke deliberately bypasses acquisition and upstream-parity gates because
it uses synthetic normalized tensors. It proves execution and wiring, not
embedding quality.

## Implemented components

### Scientific and experiment contracts

- Frozen TESSERA v1.1 MPC baseline is mandatory.
- Checkpoint-bound S2 order:
  `B04,B02,B03,B08,B8A,B05,B06,B07,B11,B12`.
- S1 order: `VV,VH`.
- MPC/AWS checkpoints, normalization, and STAC profiles cannot be mixed.
- Spatial and temporal splits are separate and enforced.
- The frozen teacher must encode the same selected window as the student.
- Annual teacher targets are rejected for short-window training.
- Training constructors require completed acquisition and parity receipts,
  except for explicitly named synthetic tests.

### Window and model code

- arbitrary, rolling, and cumulative-prefix window sampling from 7–365 days;
- cross-year absolute-day handling without day-of-year leakage;
- compact regional Student with S1/S2 branches and window context;
- mask-aware wrapper around the exact TESSERA graph;
- packed-Q/V LoRA plus optional FFN/reducer targets;
- LoRA parameters inherit the base device and dtype, including CUDA;
- Teacher-Student and LoRA objectives and trainer steps;
- exact cache/recompute incremental embedding semantics.

### Checkpoint and compatibility code

- resumable streaming checkpoint download;
- byte-count and SHA-256 verification before atomic install;
- idempotent reuse of a valid checkpoint;
- strict checkpoint-key and tensor-shape loading;
- safe `torch.load(..., weights_only=True)`;
- upstream parity command and receipt contract.

The parity machinery exists, but the canonical real-data parity fixture has not
yet been captured.

### Sampling, discovery, and durability primitives

- deterministic stratified sampling from a prepared candidate CSV;
- stable sample IDs, inclusion probabilities, and split expansion;
- SQLite task ledger with leases, retries, checksums, and completion gates;
- paginated MPC STAC discovery with retry/backoff and no tile-level cloud cut;
- immutable query identity bound to bbox, count, provider, collection, bands,
  and query parameters;
- content-addressed raw STAC items and checksum receipts;
- recovery of an interrupted query/receipt publication by safe re-query;
- rejection of S1 items without ascending/descending orbit metadata;
- canonical immutable Parquet/Zstd point-shard schemas;
- S2 `uint16[10]`, SCL, validity, source-item/query provenance;
- S1 scaled-dB `int16[2]`, orbit, validity, source-item/query provenance;
- duplicate observation rejection and optional read-time SHA verification.

### Exact MPC value transforms

- S2 processing-baseline harmonization after 2022-01-25;
- SCL compatibility validity classes;
- S1 amplitude-to-shifted/scaled-dB conversion matching upstream truncation.

The S1 transform intentionally follows NumPy float32 behavior. Exact integer
boundaries can differ by one unit across NumPy/libm builds because upstream
truncates rather than rounds; tests use stable non-boundary values.

## Committed experiment sizes

| Tier | Anchors | Years | Point-years | Role |
|---|---:|---|---:|---|
| Smoke | 128/country | 2023–2025 | 768 | First real-data proof |
| Pilot | 25,000/country | 2019, 2021, 2023–2025 | 250,000 | Learning curves |
| Preferred full | estimated 0.55–0.65M/country | 2019–2025 | estimated 7.7–9.1M | Full regional experiment |

The full count is only an estimate until the boundary and land masks are
applied. Do not promote the pilot to full automatically. Learning curves should
decide whether the full run earns its cost.

Primary stratification is RESOLVE ecoregion × ESA WorldCover 2021. Elevation,
climate, and valid-observation quantiles are balancing variables. The current
sampler consumes these labels; it does not yet construct them.

## What is not implemented

These gaps are intentional and must not be described as production-ready:

1. Pinned downloads for the official boundary, WorldCover, ecoregion,
   elevation, and climate inputs.
2. The 200 m candidate-lattice and strata-construction command.
3. A real `data/candidates.csv` or smoke manifest.
4. Sparse COG-to-point raster reading and just-in-time MPC URL signing.
5. STAC → transform → Parquet → ledger transaction integration.
6. Explicit scientific terminal outcomes for no/insufficient observations.
7. A Parquet-to-`ObservationBatch` loader and shard-aware batch sampler.
8. The canonical official CPU/FP32 upstream parity fixture.
9. A resumable epoch/checkpoint training runner with AMP and DDP.
10. Real frozen-baseline, Student, or LoRA training/evaluation artifacts.
11. A durable incremental embedding cache.

`spectrajam validate-config --operational` currently fails by design because
the boundary files and their checksums are placeholders.

## Next milestone: real smoke-data vertical slice

Implement this sequence; do not restart with GPU validation.

### 1. Build the candidate frame

- Pin exact download URLs, versions, licenses, and SHA-256 values for World Bank
  Official Boundaries v2 and the strata inputs.
- Keep the declared Israel NDLSA exclusion policy. Any alternate operational
  extent must be a separately named experiment.
- Generate a deterministic 200 m land lattice inside each pinned boundary.
- Attach RESOLVE ecoregion and WorldCover 2021 class.
- Attach or stage elevation/climate/observation-count balancing variables.
- Write a provenance receipt and `data/candidates.csv`.

Do not silently substitute Natural Earth boundaries or an unversioned web
download.

### 2. Create the smoke universe

```bash
spectrajam sample \
  --config configs/smoke.yaml \
  --candidates data/candidates.csv \
  --output data/manifests/smoke.csv

spectrajam ledger-init \
  --config configs/smoke.yaml \
  --manifest data/manifests/smoke.csv \
  --database data/state/smoke-acquisition.sqlite

spectrajam catalog-discover \
  --config configs/smoke.yaml \
  --manifest data/manifests/smoke.csv \
  --output data/catalog
```

The first command is blocked until step 1 exists. The latter commands are
implemented but have not yet been run against a real manifest.

### 3. Implement sparse asset-major materialization

The materializer must:

- process STAC item/date assets across all points in a work unit, never query
  or open COGs once per point;
- sign MPC asset URLs again on every retry;
- use SCL first to choose the deterministic first valid overlapping S2 item;
- treat SCL classes `{0,1,2,3,8,9}` as invalid in the compatibility profile;
- use nearest-neighbor resampling for SCL and bilinear resampling for spectral
  bands on the upstream-compatible 10 m grid;
- subtract the MPC S2 +1000 offset for dates after 2022-01-25 where values are
  at least 1000;
- convert S1 amplitude using
  `(20 * log10(amplitude) + 50) * 200`, clip to `[0,32767]`, then truncate;
- retain ascending and descending orbit identity;
- stream touched COG blocks and never persist full 20 km imagery cubes;
- write same-filesystem `.part` fragments, fsync, validate, checksum, and
  publish immutably;
- use bounded retries for 408/425/429/5xx, timeouts, resets, and truncated
  reads while honoring `Retry-After`;
- commit artifact URI, SHA, observation count, and provenance to SQLite only
  after immutable publication succeeds;
- make reruns reuse valid completed work without replacing points.

The canonical durable units stay integer. Normalize to FP32 in the loader and
use BF16 autocast only inside training.

### 4. Close the parity gate

- Clone `ucam-eo/tessera` at the pinned commit.
- Build a tiny real MPC tile through the pinned official preprocessing path.
- Run official CPU/FP32 inference.
- Assert no missing or unexpected checkpoint keys.
- Compare normalized inputs exactly and raw first-128 outputs initially at
  `rtol=1e-5, atol=1e-5`.
- Retain official int8/scales as the end-to-end fixture.
- Write the parity receipt consumed by the trainers.

### 5. Train the real smoke experiment

- Load the 768 point-years into arbitrary-window batches.
- Run the frozen TESSERA baseline first.
- Run one Rwanda, one Israel, and one shared Student smoke.
- Run the corresponding LoRA smokes.
- Start on one H100 with BF16 autocast.
- Confirm losses, gradients, save/resume, data provenance, and evaluation at
  7/14/30/60/90/180/365 days.
- Add DDP only after this single-GPU real-data path is reproducible.

## Definition of done for the next milestone

The real-data milestone is complete only when:

- both countries contribute exactly 128 smoke anchors;
- all 768 point-years have explicit terminal S1 and S2 outcomes;
- the ledger reports zero unresolved tasks;
- every committed shard passes schema, provenance, count, and SHA checks;
- a killed and restarted acquisition reuses completed work;
- project storage stays below the declared budget and no raster cubes remain;
- the official parity receipt passes;
- both model tracks complete a real-data optimizer step on `cuda:0`;
- tests and an opt-in tiny live-STAC integration test pass.

## VM resume commands

From the repository root:

```bash
source .venv/bin/activate

export PIP_CONFIG_FILE=/dev/null
export PIP_INDEX_URL=https://pypi.org/simple
unset PIP_EXTRA_INDEX_URL

git pull --ff-only
python -m pip install -e ".[data,train,dev]"
pytest -q

# These are already verified; rerunning is optional and idempotent.
spectrajam fetch-checkpoint --config configs/smoke.yaml
spectrajam model-smoke --config configs/smoke.yaml --device cuda:0
```

Expected checkpoint output has `reused: true`. Do not repeat topology or generic
GPU diagnostics unless CUDA behavior actually regresses.

## Useful code entry points

- `configs/smoke.yaml`: first real-data contract.
- `src/spectrajam/cli.py`: existing operational commands.
- `src/spectrajam/sampling.py`: candidate selection and manifest expansion.
- `src/spectrajam/stac.py`: query planning and immutable discovery snapshots.
- `src/spectrajam/preprocessing.py`: parity-critical raw transforms.
- `src/spectrajam/pointstore.py`: canonical durable observation schema.
- `src/spectrajam/ledger.py`: task leases, retries, artifacts, and gates.
- `src/spectrajam/retry.py`: durable task retry loop.
- `src/spectrajam/training.py`: Student/LoRA trainer steps and gates.
- `src/spectrajam/model_smoke.py`: verified synthetic execution smoke.
- `src/spectrajam/parity.py`: official-output compatibility gate.
- `docs/DESIGN.md`: scientific and architectural decisions.
- `docs/WINDOWS.md`: arbitrary-window and incremental semantics.
- `docs/RESEARCH.md`: evidence and source record.

## Known residual risks

- Two concurrent `fetch-checkpoint` processes share one `.part`; invoke that
  command once at a time until a file lock is added.
- Checkpoint and boundary paths are currently interpreted relative to the
  repository working directory.
- The successful VM Torch/CUDA package version is not yet pinned in a lockfile.
- Current catalog discovery is work-block/year scoped; monthly atomic
  materialization units still need to be introduced.
- Political-boundary provenance requires particular care for Israel; preserve
  the documented policy and receipts.
- The 50 GiB project cap and free-space watermark are design decisions, not yet
  enforced in code.

## Copy/paste prompt for the next Codex session

```text
Continue SpectraJam from docs/HANDOFF.md on public main. Do not redo hardware,
checkpoint, or synthetic model-smoke validation unless a regression appears.
Implement the next real-data milestone in order: pinned boundary/strata inputs
and deterministic candidate frame, then the asset-major MPC COG-to-point
materializer with immutable Parquet publication and ledger commits. Preserve
the exact TESSERA v1.1 MPC band order/transforms and arbitrary-window contract.
Run tests, review integrity/crash-resume behavior, update the handoff, and push
working commits to noobjam/SpectraJam.
```
