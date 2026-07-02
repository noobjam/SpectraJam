# SpectraJam session handoff

Last updated: 2026-07-02
Repository: <https://github.com/noobjam/SpectraJam>
Branch: `main`
Baseline at the start of the frame milestone: `b19f4b6`

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
synthetic observations. The VM has now reproduced the authoritative frame,
selected the 128/country smoke universe, initialized all 1,536 acquisition
tasks, and persisted the complete STAC catalog. The sparse materializer is
implemented and tested; the next action is its first real MPC smoke run.

## Verified VM state

The following was verified on the training VM on 2026-07-01 through 2026-07-02:

- 8 × NVIDIA H100 80 GB HBM3;
- all-to-all `NV18` GPU connectivity in one NUMA domain;
- public-PyPI installation works after overriding the corporate pip index;
- the full suite passed before the frame milestone: `112 passed`;
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

- ten pinned candidate-frame inputs with exact URLs, byte counts, SHA-256,
  version, license, and producer metadata;
- resumable verified download of World Bank catalog-v2 ADM0/NDLSA and data
  dictionary, RESOLVE Ecoregions 2017, the WorldCover grid, and five local
  WorldCover 2021 v200 COGs (506,016,614 bytes total);
- deterministic per-country UTM lattice (`EPSG:32735` RWA, `EPSG:32636` ISR)
  with globally anchored 200 m centers and aligned 20 km blocks;
- exact World Bank ISO row selection with standalone ADM0/NDLSA policy checks;
- explicit Latin-1 RESOLVE decoding, component-hash validation, geometry
  repair accounting, and numeric `ECO_ID` labeling;
- nearest-pixel WorldCover labeling, nodata/permanent-water exclusion, explicit
  missing-ecoregion exclusions, and deterministic tie resolution;
- atomic candidate CSV publication plus a source/runtime/count/SHA receipt;
- candidate-receipt verification before selection and a sampling receipt that
  binds candidate, config, and manifest SHA-256 values;
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
- typed validation of persisted catalog queries and raw item documents;
- smoke-only country/year asset-major materialization with touched-block reads;
- a common country-UTM 10 m compatibility grid, nearest SCL/S1 and bilinear S2;
- fresh just-in-time MPC signing on every bounded asset retry;
- deterministic same-day S2 selection fixed by the first valid SCL item;
- independent S1 VV/VH mosaics with content-addressed composite provenance;
- per-point immutable publication, read-back validation, and ledger commit;
- explicit complete, insufficient-valid, no-source, and terminal-error counts;
- preflight binding of the exact catalog inventory before any task is claimed.
- resume binding of materializer, preprocessing, point-store, Rasterio/GDAL,
  NumPy, PyArrow, and MPC SDK versions so partial runs cannot mix semantics.

### Exact MPC value transforms

- S2 processing-baseline harmonization after 2022-01-25;
- SCL compatibility validity classes;
- S1 amplitude-to-shifted/scaled-dB conversion matching upstream truncation.

The S1 compatibility transform promotes direct Rasterio samples to float64
before the logarithm, matching stackstac's pinned default dtype. Exact
integer-boundary equivalence still belongs to the official parity fixture
because upstream truncates rather than rounds.

## Committed experiment sizes

| Tier | Anchors | Years | Point-years | Role |
|---|---:|---|---:|---|
| Smoke | 128/country | 2023–2025 | 768 | First real-data proof |
| Pilot | 25,000/country | 2019, 2021, 2023–2025 | 250,000 | Learning curves |
| Preferred full | 593,292 RWA + 506,429 ISR reference | 2019–2025 | 7,698,047 | Full regional experiment |

The table's full-count range was the pre-frame planning estimate. The reference
count below now replaces it for this exact source/runtime contract. Do not
promote the pilot to full automatically; learning curves decide whether the
full run earns its cost.

### Reference frame reproduction

The 2026-07-02 local reference run produced 1,099,721 land anchors:

| Country | Candidates | Permanent-water exclusions | Missing-ecoregion exclusions |
|---|---:|---:|---:|
| Rwanda | 593,292 | 38,336 | 1,534 (0.258%) |
| Israel | 506,429 | 12,716 | 1,440 (0.284%) |

Israel touched three NDLSA boundaries with zero positive-area overlap; Rwanda
touched none. The missing-ecoregion rates are source-edge gaps and remain below
the fail-closed 1% ceiling. Generated data is ignored by Git, so the VM must
reproduce these counts and record its own output/receipt hashes before sampling.

Reference artifact identities:

```text
source bundle:          506016614 bytes
candidate CSV:          142975195 bytes
candidate CSV SHA-256:  2127dd4ea912acf834ecbe67f496f349bfe475ff210d73b771bfb579395db4f0
candidate receipt SHA:  4ccd2c934a4ff2f06b904a7f8a99c1ca7ff93efeb79c53edb7451db676b999ad
source receipt SHA:     5b792894cac1ea16d43a60f41231fb8cac5de6dc30ab93aa95e3e3dd0ede038b
```

The receipt's import-time source hashes matched the checked-out implementation,
and a pre-review/reference rerun was byte-identical. The receipt records
GeoPandas, Pyogrio/GDAL, Rasterio/GDAL, Shapely/GEOS, PyProj/PROJ, and NumPy
versions. Frame-critical geospatial package versions are exact pins.

The real smoke selection then passed its own receipt gate:

- 128 anchors per country, 256 total;
- 768 point-years;
- manifest SHA-256
  `9a739447daa17a0663c33822f9d687ac8ce3cedb8d12c3deec8fd9baa6bf13f9`;
- sampling-receipt SHA-256
  `c6cf37d42ac364cb8fe4570eebd9027ef3685d12f526a5df40ba6d0e27f7a7c8`;
- 1,536 pending ledger tasks (768 point-years × S1/S2), zero failures, with
  the ledger bound to that sampling receipt plus the manifest and config;
- 333 country/block/year STAC work tiles;
- 666 immutable S1/S2 query snapshots, 78,287 item references, and 7,893
  content-addressed raw STAC item documents on the VM;
- current local test result: `163 passed, 1 CUDA-only skipped`;
- `validate-config --operational` passes with the pinned frame and checkpoint.

Primary stratification is RESOLVE ecoregion × ESA WorldCover 2021. Elevation,
climate, and valid-observation quantiles are balancing variables. The current
sampler consumes these labels; it does not yet construct them.

## What is not implemented

These gaps are intentional and must not be described as production-ready:

1. Pinned elevation and climate inputs and their balancing quantiles. The
   implemented frame currently persists only the declared primary stratum.
2. A committed smoke manifest; generated frame/source artifacts remain ignored
   data products and must be reproduced on the VM.
3. A completed real MPC materialization run; all 1,536 VM tasks are still
   pending at this handoff.
4. Batched Parquet publication for pilot/full scale. The current per-point
   writer is intentionally smoke-only to avoid millions of tiny files.
5. A Parquet-to-`ObservationBatch` loader and shard-aware batch sampler.
6. The canonical official CPU/FP32 upstream parity fixture.
7. A resumable epoch/checkpoint training runner with AMP and DDP.
8. Real frozen-baseline, Student, or LoRA training/evaluation artifacts.
9. A durable incremental embedding cache.

`spectrajam validate-config --operational` now succeeds after
`fetch-frame-sources` has installed the pinned ADM0 file (and after the pinned
checkpoint exists when the command also requests it).

## Next milestone: real smoke-data vertical slice

Frame construction, smoke selection, ledger initialization, and catalog
discovery are complete on the VM. Do not restart them or redo GPU validation.

### 1. Reproduce the candidate frame on the VM

```bash
spectrajam fetch-frame-sources --source-root data/frame

spectrajam candidate-frame \
  --config configs/smoke.yaml \
  --source-root data/frame \
  --output data/candidates.csv \
  --receipt data/candidates.receipt.json

spectrajam validate-config --config configs/smoke.yaml --operational
```

The source fetch is about 506 MB and is resumable. Valid completed files are
hashed and reused. An existing wrong artifact or a different receipt is never
silently replaced. Preserve the declared Israel policy; do not substitute
Natural Earth, a live ArcGIS response, or a different operational extent.

### 2. Create the smoke universe

```bash
spectrajam sample \
  --config configs/smoke.yaml \
  --candidates data/candidates.csv \
  --candidate-receipt data/candidates.receipt.json \
  --output data/manifests/smoke.csv

spectrajam ledger-init \
  --config configs/smoke.yaml \
  --manifest data/manifests/smoke.csv \
  --sampling-receipt data/manifests/smoke.csv.receipt.json \
  --database data/state/smoke-acquisition.sqlite

spectrajam catalog-discover \
  --config configs/smoke.yaml \
  --manifest data/manifests/smoke.csv \
  --output data/catalog
```

These commands completed on the VM: exactly 128 anchors per country, 768
point-years, 1,536 pending tasks, and 666 query snapshots were recorded.

### 3. Run sparse asset-major materialization

The implemented materializer:

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

It validates all catalogs and the output path before claiming work, binds the
catalog-inventory and implementation/runtime SHAs to the existing ledger, keeps
signed SAS URLs out of artifacts/errors, renews leases during block reads and
publication, and scopes a failed asset to only its applicable points. The
preflight also signs and opens representative S1/S2 assets before any claim.
This implementation is gated to the smoke config; batched pilot/full Parquet
publication is still required.

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

Record the printed PID. Monitor with:

```bash
tail -f runs/logs/smoke-materialize.log
cat runs/smoke-materialize.exit
spectrajam ledger-status --database data/state/smoke-acquisition.sqlite
```

Do not repeat topology, checkpoint, model-smoke, frame, sampling, or catalog
commands unless their recorded artifacts actually fail validation.

## Useful code entry points

- `configs/smoke.yaml`: first real-data contract.
- `src/spectrajam/cli.py`: existing operational commands.
- `src/spectrajam/frame_sources.py`: pinned source registry and verified fetch.
- `src/spectrajam/candidate_frame.py`: lattice, land mask, strata, and receipt.
- `src/spectrajam/sampling.py`: candidate selection and manifest expansion.
- `src/spectrajam/stac.py`: query planning and immutable discovery snapshots.
- `src/spectrajam/materialize.py`: sparse COG reads, selection, retry, and commit.
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

- Frame-source fetch and candidate construction use non-blocking POSIX advisory
  locks. Keep the source root on the VM's local filesystem; lock behavior on a
  differently configured network filesystem has not been validated. The older
  checkpoint fetcher still lacks its own process lock.
- Checkpoint and boundary paths are currently interpreted relative to the
  repository working directory.
- A non-default `--source-root` also requires matching `extent_policy` paths in
  the config; the committed contracts intentionally point to `data/frame`.
- The successful VM Torch/CUDA package version is not yet pinned in a lockfile.
- The smoke writer publishes one small Parquet per point-year-modality. Pilot
  and full scale need deterministic shared shards plus row indexes before use.
- The common 10 m grid and raw transforms are pinned from upstream behavior,
  but exact end-to-end equivalence remains blocked on the official real-data
  parity fixture.
- Political-boundary provenance requires particular care for Israel; preserve
  the documented policy and receipts.
- The 50 GiB project cap and free-space watermark are design decisions, not yet
  enforced in code.

## Copy/paste prompt for the next Codex session

```text
Continue SpectraJam from docs/HANDOFF.md on public main. Do not redo hardware,
checkpoint, frame, sampling, catalog, or synthetic model-smoke work unless a
recorded artifact fails validation. Pull the implemented smoke materializer,
run it with the documented nohup command, monitor its durable ledger, diagnose
any scoped failures without replacing completed points, and close all 1,536
terminal outcomes. Then capture the official real-data parity fixture and build
the Parquet-to-ObservationBatch loader for the first real Student/LoRA step.
```
