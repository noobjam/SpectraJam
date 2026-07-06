# Plain TESSERA incremental pixel embeddings

This folder is an independent frozen-TESSERA v1.1 baseline. It does not load,
train, or import the LoRA and distillation runtimes.

For the exact current git/VM handoff state and copy-paste continuation commands,
see [`HANDOFF.md`](HANDOFF.md).

It reads the Harvard field labels from:

```text
/mnt/foundry-az/playground/data/ground_truth/harvard_wkt.parquet
```

Every WKT is rasterized onto a globally snapped 10 m UTM grid using the
pixel-center rule. TESSERA runs once per physical pixel. A pixel that belongs to
multiple fields is embedded once and then linked back to each field's `id` and
`landcover`; embeddings are never spatially averaged before or after the model.
WKT is authoritative for rasterization and UTM selection. `LONGITUDE`/`LATITUDE`
values are range-validated but may be auxiliary reference points; whether they
fall inside the WKT bounds is recorded as `coordinate_status` in
`fields.parquet` and summarized in the run manifests. Missing labels,
unrecoverable WKT, and fields crossing a UTM-zone boundary still fail the run
instead of silently dropping labelled fields.

## Fixed temporal contract

All intervals are half-open cumulative prefixes:

| Window | Input interval |
|---|---|
| `w1` | `[2024-09-01, 2025-01-01)` |
| `w2` | `[2024-09-01, 2025-05-01)` |
| `w3` | `[2024-09-01, 2025-09-01)` |
| `w4` | `[2024-09-01, 2026-01-01)` |

The final cutoff includes all of 2025-12-31. Each prefix is recomputed exactly;
plain TESSERA is bidirectional, so an earlier embedding cannot be updated by
adding a vector delta.

`w4` is intentionally a requested 487-day experiment. TESSERA v1.1 was trained
with annual sequences and encodes raw day-of-year without a year token, so this
window repeats day-of-year values and is marked as outside the annual contract in
`run.json`.

## Provider and preprocessing

The implementation uses Microsoft Planetary Computer STAC:

- Sentinel-2 L2A: `sentinel-2-l2a`
- Sentinel-1 RTC: `sentinel-1-rtc`

It mirrors the pinned upstream v1.1 MPC preprocessing contract: canonical S2
band order, bilinear spectral resampling, nearest-neighbor SCL, the post-2022
`-1000` BOA correction, compatibility SCL mask `{0,1,2,3,8,9}`, S1 scaled-dB
conversion, orbit-specific S1 normalization, and deterministic 8…256 observation
buckets. STAC item JSON is saved unsigned and assets are signed immediately
before raster reads.

## Checkpoint

The repository convention is used directly:

```text
checkpoints/tessera_v1_1_mpc_encoder.pt
```

If that gitignored file has not already been provisioned, download the official
[`tessera_v1_1_mpc_encoder.pt`](https://drive.google.com/file/d/1t-gfTxi3Hg_uJXpJ9etROCRgKt2myfJ2/view)
and place it at the path above before preflight.

The default config verifies the published encoder SHA-256:

```text
5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3
```

## Run

From the repository root:

```bash
python -m pip install -e ".[data,train,dev]"
python -m pip install -r plain_tessera_incremental/requirements.txt

python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config.yaml \
  --preflight-only

python -m plain_tessera_incremental \
  --config plain_tessera_incremental/config.yaml
```

If the MPC account requires it, expose `PC_SDK_SUBSCRIPTION_KEY` in the runtime
environment. The pipeline never logs or persists that key or signed SAS URLs.

## Output

The default destination is:

```text
/mnt/foundry-az/playground/data/ground_truth/harvard_tessera_incremental
```

Artifacts are:

- `run.json`: immutable input, checkpoint, preprocessing, and window identity.
- `fields.parquet`: original rows/WKT plus geometry/coordinate audits and pixel
  count.
- `pixels.parquet`: unique 10 m physical pixels and their coordinates.
- `field_pixels.parquet`: field-to-pixel memberships, overlap counts, and label
  conflicts.
- `stac/*.json`: immutable unsigned catalog snapshots per 20 km work tile.
- `cache/*.npz`: preprocessed S1/S2 timelines reused by all four prefixes.
- `embeddings/window_id=w*/part.parquet`: long-format labelled pixel embeddings.
- `COMPLETED.json`: final counts after every atomic shard is present.

Each embedding row contains the field and pixel IDs, field label, pixel center,
window bounds, source/valid/model-input counts, outcome, and a nullable
`list<float32>` embedding whose complete rows are enforced to length 128.
Both-modalities-empty prefixes are retained with
`outcome=empty_window` and a null embedding.

Rerunning the same command resumes from validated Parquet shards and cached
timelines. A changed parquet, checkpoint, configuration, or preprocessing
contract is rejected rather than mixed into an existing output directory.
