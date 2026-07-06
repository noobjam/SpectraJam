# Plain TESSERA incremental pipeline — handoff

Last updated: 2026-07-06

This is the operational checkpoint for continuing development locally and
running the real Harvard job on the VM.

## 1. Current state — read this first

- Local branch: `main`
- The standalone plain-TESSERA implementation and this handoff are delivered
  together on `main` under `plain_tessera_incremental/`.
- If this file is visible after pulling `origin/main` on the VM, the pipeline
  code and operational instructions are both present.
- The full Harvard embedding job has not been started. No Harvard output is
  expected yet.

## 2. What is implemented

The implementation is isolated under `plain_tessera_incremental/`; it does not
use the LoRA or distillation runtimes.

- Reads
  `/mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet`.
- Validates the required columns, labels, coordinate ranges, WKT geometries, and
  UTM grid compatibility. WKT is authoritative; auxiliary coordinate/WKT-bound
  mismatches are audited rather than rejected.
- Rasterizes every valid field to globally snapped 10 m pixel centers.
- Computes each physical pixel once while preserving all field-to-pixel label
  memberships, including overlaps and label conflicts.
- Queries Microsoft Planetary Computer STAC for:
  - `sentinel-2-l2a`
  - `sentinel-1-rtc`
- Applies the pinned TESSERA v1.1 MPC preprocessing contract for S2, SCL, and
  ascending/descending S1.
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

- Full repository suite: 183 tests passed; 1 CUDA-only test skipped locally.
- Ruff lint and format checks passed.
- Python compilation passed.
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

Replace `/path/to/SpectraJam` with the actual clone location:

```bash
cd /path/to/SpectraJam
git switch main
git pull --ff-only origin main
git log -1 --oneline
```

Activate the VM environment. Create it only if it does not already exist:

```bash
cd /path/to/SpectraJam

test -d .venv || python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[data,train,dev]"
python -m pip install -r plain_tessera_incremental/requirements.txt
```

If the VM uses a specially installed CUDA build of PyTorch, confirm it before
launching and do not replace it with a CPU-only build:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
```

## 6. Next step C — verify input and checkpoint on the VM

```bash
cd /path/to/SpectraJam
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
python -m pip install gdown
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

## 7. Next step D — preflight before spending compute

```bash
cd /path/to/SpectraJam
source .venv/bin/activate
mkdir -p logs

python -u -m plain_tessera_incremental \
  --config plain_tessera_incremental/config.yaml \
  --preflight-only \
  2>&1 | tee logs/plain_tessera_preflight.log
```

Do not launch the full job unless preflight reports the expected parquet,
checkpoint SHA-256, four windows, and the intended resolved device.

## 8. Next step E — launch the resumable VM job

```bash
cd /path/to/SpectraJam
source .venv/bin/activate
mkdir -p logs

nohup env PYTHONUNBUFFERED=1 python -u -m plain_tessera_incremental \
  --config plain_tessera_incremental/config.yaml \
  > logs/plain_tessera_incremental.log 2>&1 &

echo $! | tee logs/plain_tessera_incremental.pid
tail -f logs/plain_tessera_incremental.log
```

The configured output root is:

```text
/mnt/foundry-az/playground/data/ground_truth/harvard_tessera_incremental
```

The exact same launch command is the resume command. Existing compatible STAC
snapshots, timeline caches, and validated embedding shards are reused.

## 9. Monitoring commands

```bash
OUTPUT=/mnt/foundry-az/playground/data/ground_truth/harvard_tessera_incremental

tail -n 200 logs/plain_tessera_incremental.log
du -sh "$OUTPUT" 2>/dev/null || true
find "$OUTPUT/embeddings" -name '*.parquet' 2>/dev/null | wc -l

test -f "$OUTPUT/COMPLETED.json" && cat "$OUTPUT/COMPLETED.json"
```

If a PID file exists:

```bash
ps -fp "$(cat logs/plain_tessera_incremental.pid)"
```

## 10. Completion gate before the next scientific step

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
python - <<'PY'
import json
from pathlib import Path

root = Path("/mnt/foundry-az/playground/data/ground_truth/harvard_tessera_incremental")
result = json.loads((root / "COMPLETED.json").read_text())
expected = 4 * result["field_pixel_membership_count"]
assert result["embedding_rows"] == expected, (result["embedding_rows"], expected)
print("complete:", result)
PY
```

After this gate passes, the next work item is to summarize coverage/outcomes by
window and land-cover label, then agree on the downstream field-level modelling
step. That next scientific step is intentionally not implemented yet.

## 11. Common failure handling

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
