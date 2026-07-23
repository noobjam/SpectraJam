# Plain TESSERA incremental pipeline — handoff

Last updated: 2026-07-07

This is the operational checkpoint for continuing development locally and
running the real Harvard job on the VM.

## 1. Current state — read this first

- Local branch: `main`
- The standalone plain-TESSERA implementation and this handoff are delivered
  together on `main` under `plain_tessera_incremental/`.
- If this file is visible after pulling `origin/main` on the VM, the pipeline
  code and operational instructions are both present.
- A legacy v1 Harvard job was running from commit `a30db0e` and writing to
  `harvard_tessera_incremental`. Preserve that directory as an audit artifact.
- v2 writes to the separate `/mnt/noobjam/harvard_tessera_incremental_v2`
  directory. Stop the legacy process only after pulling and preflighting the
  delivered v2 code.

## 2. What is implemented

The implementation is isolated under `plain_tessera_incremental/`; it does not
use the LoRA or distillation runtimes.

- Reads
  `/mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet`.
- Validates the required columns, labels, coordinate ranges, WKT geometries, and
  UTM grid compatibility. WKT is authoritative; auxiliary coordinate/WKT-bound
  mismatches are audited rather than rejected.
- Rasterizes every valid field to a globally snapped 10 m grid and assigns its
  label only to cells whose center is inside or on the WKT boundary. Cell
  footprints may extend outside the WKT.
- Records projected WKT area/dimensions and both center-selected and
  positive-area-overlap cell counts so sparse small fields are auditable; the
  positive-area count is diagnostic and does not create label memberships.
- Computes each physical pixel once while preserving all field-to-pixel label
  memberships, including overlaps and label conflicts.
- Queries Microsoft Planetary Computer STAC for:
  - `sentinel-2-l2a`
  - `sentinel-1-rtc`
- Applies the pinned TESSERA v1.1 MPC preprocessing contract for S2, SCL, and
  ascending/descending S1.
- Adds a local 500 m projected catalog-discovery halo without dilating WKT
  labels; this is a local robustness measure, not part of the pinned upstream
  preprocessing contract. S2 dates plus S1 date/orbit groups are materialized
  with eight bounded workers.
- Loads the frozen plain encoder from
  `checkpoints/tessera_v1_1_mpc_encoder.pt`.
- Produces four cumulative, half-open prefixes:

| Window | Interval | Days |
|---|---|---:|
| `w1` | `[2024-09-01, 2025-01-01)` | 122 |
| `w2` | `[2024-09-01, 2025-05-01)` | 242 |
| `w3` | `[2024-09-01, 2025-09-01)` | 365 |
| `w4` | `[2024-09-01, 2026-01-01)` | 487 |

- Writes atomic, resumable Parquet shards and cached preprocessed timelines.
- Fingerprints the input parquet, checkpoint, code, runtime versions, resolved
  device, STAC snapshot, preprocessing policy, and task inputs so incompatible
  partial runs cannot be mixed.
- Retains both-modalities-empty prefixes with a null embedding and
  `outcome=empty_window`.

Scientific caveat: `w4` is deliberately the requested 487-day prefix, but it is
outside TESSERA's annual training contract. The plain model uses day-of-year and
has no year token, so day-of-year values repeat in this window. This is recorded
in `run.json` and must remain visible in later analysis.

## 3. Verification already completed locally

- Full repository suite: 196 tests passed; 1 CUDA-only test skipped locally.
- `git diff --check`, notebook JSON/code parsing, and Python compilation passed.
- Ruff was not installed in the final local verification environment.
- The published 230 MB MPC encoder loaded successfully with SHA-256:

  ```text
  5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3
  ```

- Real checkpoint smoke inference returned a finite `float32` tensor of shape
  `(1, 128)`.
- Live MPC discovery returned both S2 and S1 items with the required assets.
- A live one-pixel end-to-end smoke passed:

  ```text
  MPC STAC -> signed COG reads -> S2/S1 preprocessing -> plain TESSERA
  outcome=['complete'], embedding shape=(1, 128), finite=True
  ```

- A mocked full orchestrator run produced the expected cardinality:
  `field-pixel memberships × 4 windows`.

The actual `/mnt/foundry-az` parquet and VM checkpoint are not mounted in the
local development environment, so the complete Harvard job can only run on the
VM.

## 4. Next step A — verify delivery before using the VM

From the repository root:

```bash
git fetch origin main
git log -1 --oneline origin/main
git ls-tree -d --name-only origin/main plain_tessera_incremental
git status --short
```

Expected result:

- The remote tree contains `plain_tessera_incremental`.
- The development worktree is clean.
- The delivery commit SHA reported in the follow-up conversation is reachable
  from `origin/main`.

## 5. Next step B — pull and prepare the VM

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam
git switch main
git pull --ff-only origin main
git log -1 --oneline
```

Activate the VM environment and verify its existing PyTorch/CUDA build before
installing anything. The running v1 job proves the current environment already
has the runtime dependencies, and v2 adds no new package dependency.

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam

test -d .venv || python3 -m venv .venv
source .venv/bin/activate
python -c 'import torch; print("torch:", torch.__version__); print("cuda available:", torch.cuda.is_available()); print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

Only when rebuilding the environment, install the non-PyTorch dependencies from
public PyPI. Omitting the `train` extra prevents pip from replacing the VM's
known-good CUDA build; provision PyTorch separately from the matching CUDA index
if the check above fails.

```bash
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --index-url https://pypi.org/simple -e ".[data,dev]"
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --index-url https://pypi.org/simple \
  -r plain_tessera_incremental/requirements.txt
```

## 6. Next step C — verify input and checkpoint on the VM

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam
source .venv/bin/activate

test -r /mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet
ls -lh /mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet

test -r checkpoints/tessera_v1_1_mpc_encoder.pt
ls -lh checkpoints/tessera_v1_1_mpc_encoder.pt

echo "5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3  checkpoints/tessera_v1_1_mpc_encoder.pt" \
  | sha256sum -c -
```

If the checkpoint is missing, download the official encoder first:

```bash
mkdir -p checkpoints
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --index-url https://pypi.org/simple gdown
python -m gdown 1t-gfTxi3Hg_uJXpJ9etROCRgKt2myfJ2 \
  -O checkpoints/tessera_v1_1_mpc_encoder.pt

echo "5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3  checkpoints/tessera_v1_1_mpc_encoder.pt" \
  | sha256sum -c -
```

If MPC requires an account subscription, expose the key in the shell without
printing or committing it:

```bash
export PC_SDK_SUBSCRIPTION_KEY="..."
```

## 7. Next step D — checked cutover to resumable v2

The checked cutover script runs v2 preflight first. If preflight fails, it does
not touch the existing job. After a successful preflight it scans `/proc`,
requires exact Python argv, repository cwd, executable type, and PID start-time
identity, stops at most one matching process, waits for it to exit, and launches
v2. The stale legacy PID file is intentionally ignored. The old output directory
is not deleted or reused.

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam
bash plain_tessera_incremental/cutover_v2.sh && \
  tail -f logs/plain_tessera_incremental_v2.log
```

The configured output root is:

```text
/mnt/noobjam/harvard_tessera_incremental_v2
```

The exact same cutover command is also the resume command. Existing compatible
STAC snapshots, timeline caches, and validated embedding shards are reused.

## 8. Monitoring commands

```bash
OUTPUT=/mnt/noobjam/harvard_tessera_incremental_v2

tail -n 200 logs/plain_tessera_incremental_v2.log
du -sh "$OUTPUT" 2>/dev/null || true
find "$OUTPUT/embeddings" -name '*.parquet' 2>/dev/null | wc -l

test -f "$OUTPUT/COMPLETED.json" && cat "$OUTPUT/COMPLETED.json"
```

If a PID file exists:

```bash
ps -fp "$(cat logs/plain_tessera_incremental_v2.pid)"
```

## 9. Completion gate before the next scientific step

The embedding stage is complete only when:

1. `COMPLETED.json` exists.
2. The log has no unresolved traceback.
3. `embedding_rows` equals
   `4 × field_pixel_membership_count` in `COMPLETED.json`.
4. Geometry and coordinate status counts are reviewed.
5. Empty-window and per-window S1/S2 observation counts are reviewed before
   downstream modelling.

Quick cardinality check:

```bash
python -c 'import json, pathlib; root = pathlib.Path("/mnt/noobjam/harvard_tessera_incremental_v2"); result = json.loads((root / "COMPLETED.json").read_text()); expected = 4 * result["field_pixel_membership_count"]; assert result["embedding_rows"] == expected, (result["embedding_rows"], expected); print("complete:", result)'
```

After this gate passes, the non-crop sampling and pixel-level MLP workflow is
available in [`PIXEL_MLP.md`](PIXEL_MLP.md). It runs from committed scripts on
the VM and writes to `/mnt/noobjam/rwanda_worldcover_mlp`, separate from the
Harvard embedding audit directory.

## 10. Scalable restart for larger polygon sets

Use the scalable config for a clean Harvard run after any fingerprinted pipeline
change. It writes to `/mnt/noobjam/harvard_tessera_incremental_v3`, leaving v2
untouched. Its 256-pixel spatial batches reduce task fan-out, while atomic
per-date/per-orbit caches retain successful remote reads across 429/5xx failures.

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam
source .venv/bin/activate
mkdir -p logs

python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_harvard_scalable.yaml \
  --preflight-only

nohup python -u -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_harvard_scalable.yaml \
  > logs/harvard_tessera_scalable_v3.log 2>&1 < /dev/null &
echo $! > logs/harvard_tessera_scalable_v3.pid
```

The exact `nohup` command is the resume command. Validated full task timelines
are skipped. Within an interrupted task, only missing date/orbit groups are
downloaded again.

## 11. Large-field w2 classifier pilot

The v3 static `fields.parquet` is valid even if remote materialization was
stopped. Select the largest audited polygons, then run only `w2`:

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam
source .venv/bin/activate

python -m plain_tessera_incremental.tools.prepare_harvard_large_field_input \
  | tee logs/harvard_large_fields_w2_prepare.json

python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_harvard_large_fields_w2.yaml \
  --preflight-only \
  | tee logs/harvard_large_fields_w2_preflight.json

nohup python -u -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_harvard_large_fields_w2.yaml \
  > logs/harvard_large_fields_w2.log 2>&1 < /dev/null &
echo $! > logs/harvard_large_fields_w2.pid
```

Defaults retain at most 25 fields per crop class, require at least 256 pixels
per field, and preserve every 10 m pixel inside each selected polygon. Inspect
the preparation class counts and preflight `pixels_per_task` before launch.

## 12. Common failure handling

- **Checkpoint SHA mismatch:** stop and provision the exact MPC encoder; do not
  bypass the checksum.
- **`output directory belongs to a different run`:** preserve the old output and
  choose a new `output_dir` in a copied config. Do not mix runs.
- **Code was updated after a partial run:** code hashes are part of the run
  fingerprint. Archive/rename the partial output directory and start the updated
  code with a fresh output directory; do not edit `run.json` to bypass the gate.
- **Invalid ground-truth field error:** fix the listed unrecoverable WKT or UTM
  boundary records; invalid labelled fields are not silently dropped.
- **MPC authorization error:** set `PC_SDK_SUBSCRIPTION_KEY` and rerun.
- **Transient raster/STAC error:** rerun the exact command; reads are retried and
  completed compatible shards resume.
- **CUDA out-of-memory before any output:** lower `runtime.batch_size` in a copied
  config and use a new output directory, because runtime/config identity is part
  of the run fingerprint.
