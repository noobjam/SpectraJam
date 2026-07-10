# Rwanda pixel-level land-cover MLP

This runbook is for the repository workflow used by this project: develop and
test locally, push the code, pull it on the GPU VM, and run all data acquisition
and training there.

The current Harvard embedding output contains only pixels selected by Harvard
field WKTs. It cannot supply non-crop examples by itself. This workflow first
samples non-crop locations from ESA WorldCover 2021, embeds those locations with
the same frozen TESSERA checkpoint and temporal windows, and only then merges
them with unambiguous pure-crop pixels.

## Label contract

- Crop labels default to `Bean`, `Irish Potato`, `Maize`, and `Rice` from the
  Harvard field data. Intercropping labels are excluded.
- Non-crop labels come from ESA WorldCover 2021 v200 at 10 m.
- Candidate non-crop locations are placed on a deterministic 200 m lattice.
- A non-crop label is retained only when the WorldCover class agrees at the
  center and all eight positions at a 20 m radius.
- Known Harvard field geometries can be buffered by 30 m and excluded from
  non-crop sampling.
- The generated 2 m WKT around each selected center causes the existing
  pipeline to embed exactly one globally snapped 10 m TESSERA pixel.
- WorldCover is weak supervision, not field-survey truth. Its 2021 date also
  differs from the 2024–2025 imagery used by the default `w2` embedding.
- Because WorldCover itself was produced from Sentinel-1/2, held-out metrics on
  these labels measure agreement with WorldCover, not independent real-world
  land-cover accuracy. A separate reference dataset is required for that claim.

## 1. Pull and activate the VM environment

```bash
cd /mnt/KSA-Oasis/El-Mohammed/SpectraJam
git switch main
git pull --ff-only origin main
source .venv/bin/activate

python -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")'
```

Install the geospatial packages without reinstalling the VM's working PyTorch
build:

```bash
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --index-url https://pypi.org/simple -e ".[data]"
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= \
  python -m pip install --index-url https://pypi.org/simple \
  -r plain_tessera_incremental/requirements.txt
```

## 2. Download the pinned Rwanda map inputs

This downloads and verifies the two Rwanda-covering WorldCover tiles and the
World Bank Rwanda boundary. The registry pins URLs, byte counts, SHA-256 hashes,
versions, and licenses. Rerunning the command reuses verified files.

```bash
python -m plain_tessera_incremental.tools.download_rwanda_worldcover
```

The default source directory is:

```text
/mnt/noobjam/rwanda_worldcover_mlp/sources
```

## 3. Build deterministic non-crop WKT samples

### Fast pilot: 100 locations per class and w2 only

Use this first. It selects at most 700 non-crop locations (100 for each of the
seven WorldCover classes) and embeds only the prefixes through `w2`, the window
used by the dataset builder. It is sufficient to check the end-to-end spatial
classifier workflow, but it is deliberately too small for final performance claims.

```bash
python -m plain_tessera_incremental.tools.prepare_worldcover_noncrop_input \
  --class-codes 10 20 30 50 60 80 90 \
  --samples-per-class 100 \
  --exclude-wkt-parquet /mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet \
  --output /mnt/noobjam/rwanda_worldcover_mlp/worldcover_noncrop_wkt_pilot_100.parquet

python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_worldcover_noncrop_pilot_w2.yaml \
  --preflight-only

mkdir -p logs
nohup python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_worldcover_noncrop_pilot_w2.yaml \
  > logs/rwanda_worldcover_noncrop_pilot_w2.log 2>&1 &
echo $! > logs/rwanda_worldcover_noncrop_pilot_w2.pid
```

After completion, point the builder at this pilot output:

```bash
python -m plain_tessera_incremental.tools.build_pixel_classification_dataset \
  --noncrop-root /mnt/noobjam/rwanda_worldcover_mlp/tessera_embeddings_pilot_w2 \
  --output /mnt/noobjam/rwanda_worldcover_mlp/pixel_classification_pilot_w2.parquet
```

### Full run: 2,000 locations per class

```bash
python -m plain_tessera_incremental.tools.prepare_worldcover_noncrop_input \
  --class-codes 10 20 30 50 60 80 90 \
  --samples-per-class 2000 \
  --exclude-wkt-parquet /mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet
```

The selected WorldCover classes are tree cover (10), shrubland (20), grassland
(30), built-up (50), bare/sparse vegetation (60), permanent water (80), and
herbaceous wetland (90). At most 2,000 pure locations are selected per class,
for a maximum of 14,000 non-crop pixels. Review the printed
`pure_candidate_counts` and `selected_class_counts` before inference; classes
with insufficient pure coverage contain fewer samples.

The generated input and provenance manifest are:

```text
/mnt/noobjam/rwanda_worldcover_mlp/worldcover_noncrop_wkt.parquet
/mnt/noobjam/rwanda_worldcover_mlp/worldcover_noncrop_wkt.parquet.manifest.json
```

## 4. Generate non-crop TESSERA embeddings

The committed config deliberately matches the Harvard v2 checkpoint,
preprocessing, grid, and four temporal windows while writing to a separate
output root.

```bash
python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_worldcover_noncrop.yaml \
  --preflight-only

mkdir -p logs
nohup python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config_worldcover_noncrop.yaml \
  > logs/rwanda_worldcover_noncrop_embeddings.log 2>&1 &
echo $! > logs/rwanda_worldcover_noncrop_embeddings.pid
```

Monitor and verify completion:

```bash
tail -f logs/rwanda_worldcover_noncrop_embeddings.log
cat /mnt/noobjam/rwanda_worldcover_mlp/tessera_embeddings/COMPLETED.json
```

To inspect live progress and embedding health without waiting for completion,
open the companion notebook from the repository root:

```bash
python -m jupyter lab plain_tessera_incremental/notebooks/inspect_worldcover_lulc_embeddings.ipynb
```

It verifies the published shard metadata/schema, reports per-class and per-window
completion plus S1/S2 observation coverage, and displays a sampled PCA/map
diagnostic. Set `TESSERA_OUTPUT_DIR` only if the embedding output is elsewhere.

## 5. Build the classification dataset

The builder uses `w2` by default, excludes ambiguous Harvard pixels, removes
duplicate field memberships, removes any crop/non-crop pixel overlap, caps each
class at 10,000 pixels, and assigns complete 10 km spatial blocks to train,
validation, or test. A Harvard field touching more than one block is removed so
one field cannot cross splits. The builder refuses a split unless every class
occurs in all three spatial partitions.

```bash
python -m plain_tessera_incremental.tools.build_pixel_classification_dataset
```

Review the class-by-split counts in:

```text
/mnt/noobjam/rwanda_worldcover_mlp/pixel_classification_w2.parquet.manifest.json
```

## 6. Train and evaluate the MLP on the GPU

```bash
python -m plain_tessera_incremental.tools.train_pixel_mlp --device cuda
```

The trainer standardizes embeddings using training pixels only, applies
inverse-square-root class weights, selects the checkpoint by validation macro
F1, and evaluates the selected checkpoint once on the spatially held-out test
blocks.

Outputs are written to:

```text
/mnt/noobjam/rwanda_worldcover_mlp/mlp_w2/
```

Key files are `metrics.json`, `best_model.pt`, `training_history.csv`,
`confusion_matrix.csv`, and `test_predictions.parquet`.

Do not interpret this test set as an independent estimate after repeatedly
tuning against it. Once the workflow stabilizes, freeze a new geographic region
or independent labeled dataset for final evaluation.
