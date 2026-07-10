from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from spectrajam.contracts import sha256_file


DEFAULT_CROP_LABELS = ("Bean", "Irish Potato", "Maize", "Rice")
REQUIRED_COLUMNS = (
    "run_fingerprint",
    "field_uid",
    "pixel_id",
    "utm_epsg",
    "pixel_x_index",
    "pixel_y_index",
    "pixel_longitude",
    "pixel_latitude",
    "window_id",
    "outcome",
    "landcover",
    "embedding",
)
SPLIT_NAMES = ("train", "validation", "test")


def _require_dependencies() -> dict[str, Any]:
    try:
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised on the GPU VM
        raise RuntimeError(
            "numpy, pandas, and pyarrow are required to build the MLP dataset"
        ) from error
    return {"np": np, "pd": pd, "pa": pa, "pq": pq}


def _window_contract(run: dict[str, Any], window_id: str) -> dict[str, Any]:
    config = run.get("config")
    if not isinstance(config, dict):
        raise RuntimeError("run.json is missing its config contract")
    windows = {
        str(item.get("window_id")): item
        for item in config.get("windows", [])
        if isinstance(item, dict)
    }
    if window_id not in windows:
        raise RuntimeError(f"run.json does not contain window {window_id!r}")
    grid = {
        "pixel_size_m": config.get("pixel_size_m"),
        "work_tile_m": config.get("work_tile_m"),
    }
    acquisition = {
        "stac_endpoint": config.get("stac_endpoint"),
        "s2_collection": config.get("s2_collection"),
        "s1_collection": config.get("s1_collection"),
    }
    return {
        "schema_version": run.get("schema_version"),
        "preprocessing_version": run.get("preprocessing_version"),
        "checkpoint_sha256": run.get("checkpoint_sha256"),
        "window": windows[window_id],
        "grid": grid,
        "acquisition": acquisition,
    }


def _read_embedding_root(
    root: Path,
    window_id: str,
    dependencies: dict[str, Any],
) -> tuple[dict[str, Any], Any]:
    pd = dependencies["pd"]
    pq = dependencies["pq"]
    run_path = root / "run.json"
    if not run_path.is_file():
        raise FileNotFoundError(f"run.json not found: {run_path}")
    run = json.loads(run_path.read_text())
    fingerprint = str(run.get("run_fingerprint", ""))
    if not fingerprint:
        raise RuntimeError(f"run fingerprint is missing from {run_path}")
    completion_path = root / "COMPLETED.json"
    if not completion_path.is_file():
        raise FileNotFoundError(f"embedding run is incomplete: {completion_path} is missing")
    completion = json.loads(completion_path.read_text())
    if not completion.get("completed") or completion.get("run_fingerprint") != fingerprint:
        raise RuntimeError(f"invalid completion marker: {completion_path}")
    shard_dir = root / "embeddings" / f"window_id={window_id}"
    files = sorted(shard_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {window_id} embedding shards found in {shard_dir}")
    parts = []
    for path in files:
        parquet = pq.ParquetFile(path)
        metadata = {
            key.decode(): value.decode()
            for key, value in (parquet.metadata.metadata or {}).items()
        }
        expected = {
            "run_fingerprint": fingerprint,
            "artifact": "pixel_embeddings",
            "window_id": window_id,
        }
        mismatched = {
            key: (metadata.get(key), value)
            for key, value in expected.items()
            if metadata.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"embedding metadata mismatch in {path}: {mismatched}")
        parts.append(parquet.read(columns=list(REQUIRED_COLUMNS)).to_pandas())
    return run, pd.concat(parts, ignore_index=True)


def _valid_vector(value: object, np: Any) -> bool:
    if value is None:
        return False
    vector = np.asarray(value, dtype=np.float32)
    return vector.shape == (128,) and bool(np.isfinite(vector).all())


def collapse_unambiguous_pixels(
    rows: Any,
    allowed_labels: set[str] | None,
    source: str,
    np: Any,
) -> tuple[Any, dict[str, int]]:
    rows = rows.copy()
    rows["landcover"] = rows["landcover"].astype(str).str.strip()
    label_counts = rows.groupby("pixel_id")["landcover"].nunique()
    ambiguous = set(label_counts[label_counts.gt(1)].index.astype(str))
    rows = rows[rows["outcome"].eq("complete")].copy()
    valid_vector = rows["embedding"].map(lambda value: _valid_vector(value, np))
    invalid_embedding_rows = int((~valid_vector).sum())
    rows = rows[valid_vector].copy()
    rows = rows[~rows["pixel_id"].astype(str).isin(ambiguous)]
    if allowed_labels is not None:
        rows = rows[rows["landcover"].isin(allowed_labels)]
    duplicate_rows = int(rows.duplicated("pixel_id", keep="first").sum())
    rows = rows.sort_values(["pixel_id", "landcover"], kind="stable")
    rows = rows.drop_duplicates("pixel_id", keep="first").reset_index(drop=True)
    rows["source"] = source
    rows = rows.rename(columns={"landcover": "label"})
    return rows, {
        "invalid_embedding_rows": invalid_embedding_rows,
        "ambiguous_pixels": len(ambiguous),
        "duplicate_membership_rows": duplicate_rows,
        "retained_pixels": len(rows),
    }


def _stable_rank(seed: int, value: str) -> int:
    digest = hashlib.sha256(f"{seed}:{value}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _cap_per_class(frame: Any, limit: int | None, seed: int) -> Any:
    if limit is None:
        return frame.reset_index(drop=True)
    if limit < 1:
        raise ValueError("max-pixels-per-class must be positive")
    selected = []
    for _, group in frame.groupby("label", sort=True):
        group = group.copy()
        group["_rank"] = [
            _stable_rank(seed, str(pixel_id)) for pixel_id in group["pixel_id"]
        ]
        selected.append(group.nsmallest(limit, "_rank").drop(columns="_rank"))
    if not selected:
        return frame.iloc[0:0].copy()
    pd = __import__("pandas")
    return pd.concat(selected, ignore_index=True)


def _hash_fraction(seed: int, block: str) -> float:
    digest = hashlib.sha256(f"{seed}:{block}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def choose_spatial_split(
    frame: Any,
    ratios: tuple[float, float, float],
    seed: int,
    attempts: int,
) -> tuple[Any, int]:
    pd = __import__("pandas")
    if len(ratios) != 3 or any(value <= 0 for value in ratios):
        raise ValueError("split ratios must contain three positive values")
    total_ratio = sum(ratios)
    ratios = tuple(value / total_ratio for value in ratios)
    blocks_per_class = frame.groupby("label")["spatial_block"].nunique()
    impossible = blocks_per_class[blocks_per_class.lt(3)]
    if len(impossible):
        raise RuntimeError(
            "spatial splitting requires at least three blocks per class: "
            + json.dumps({str(key): int(value) for key, value in impossible.items()})
        )
    block_label = pd.crosstab(frame["spatial_block"], frame["label"])
    block_sizes = block_label.sum(axis=1)
    class_totals = block_label.sum(axis=0)
    threshold_1 = ratios[0]
    threshold_2 = ratios[0] + ratios[1]
    best: tuple[float, int, dict[str, str]] | None = None
    for candidate_seed in range(seed, seed + attempts):
        assignments = {}
        for block in block_label.index.astype(str):
            value = _hash_fraction(candidate_seed, block)
            assignments[block] = (
                "train"
                if value < threshold_1
                else "validation"
                if value < threshold_2
                else "test"
            )
        split_counts = {name: block_label.iloc[0:0].sum(axis=0) for name in SPLIT_NAMES}
        split_sizes = {name: 0 for name in SPLIT_NAMES}
        for block, split in assignments.items():
            split_counts[split] = split_counts[split] + block_label.loc[block]
            split_sizes[split] += int(block_sizes.loc[block])
        if any(split_sizes[name] == 0 for name in SPLIT_NAMES):
            continue
        if any((split_counts[name] == 0).any() for name in SPLIT_NAMES):
            continue
        total_rows = len(frame)
        score = sum(
            abs(split_sizes[name] / total_rows - ratios[index])
            for index, name in enumerate(SPLIT_NAMES)
        )
        for index, name in enumerate(SPLIT_NAMES):
            score += float(
                ((split_counts[name] / class_totals) - ratios[index]).abs().mean()
            )
        candidate = (score, candidate_seed, assignments)
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError(
            f"no class-complete spatial split found in {attempts} seeds; "
            "increase spatial coverage or reduce the class set"
        )
    output = frame.copy()
    output["split"] = output["spatial_block"].map(best[2])
    return output, best[1]


def _concat(frames: list[Any], pd: Any) -> Any:
    return pd.concat(frames, ignore_index=True)


def _write_dataset(frame: Any, path: Path, dependencies: dict[str, Any]) -> None:
    np = dependencies["np"]
    pa = dependencies["pa"]
    pq = dependencies["pq"]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    vector_column = pa.array(
        [np.asarray(value, dtype=np.float32).tolist() for value in frame["embedding"]],
        type=pa.list_(pa.float32()),
    )
    embedding_index = table.schema.get_field_index("embedding")
    table = table.set_column(embedding_index, "embedding", vector_column)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"artifact": b"pixel_landcover_classification_dataset",
            b"schema_version": b"1",
        }
    )
    part = path.with_suffix(path.suffix + ".part")
    pq.write_table(table, part, compression="zstd", use_dictionary=True)
    if pq.read_metadata(part).num_rows != len(frame):
        raise RuntimeError(f"Parquet row-count validation failed: {part}")
    part.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def build(args: argparse.Namespace) -> dict[str, Any]:
    dependencies = _require_dependencies()
    np = dependencies["np"]
    pd = dependencies["pd"]
    crop_root = Path(args.crop_root).expanduser()
    noncrop_root = Path(args.noncrop_root).expanduser()
    crop_run, crop_rows = _read_embedding_root(
        crop_root, args.window_id, dependencies
    )
    noncrop_run, noncrop_rows = _read_embedding_root(
        noncrop_root, args.window_id, dependencies
    )
    crop_contract = _window_contract(crop_run, args.window_id)
    noncrop_contract = _window_contract(noncrop_run, args.window_id)
    if crop_contract != noncrop_contract:
        raise RuntimeError(
            "crop and non-crop embeddings are not comparable:\n"
            + json.dumps(
                {"crop": crop_contract, "noncrop": noncrop_contract},
                indent=2,
                sort_keys=True,
            )
        )
    crop, crop_audit = collapse_unambiguous_pixels(
        crop_rows,
        set(args.crop_labels),
        "harvard",
        np,
    )
    noncrop, noncrop_audit = collapse_unambiguous_pixels(
        noncrop_rows,
        None,
        "worldcover_2021",
        np,
    )
    noncrop = noncrop[noncrop["label"].str.startswith("Non-crop: ")].copy()
    overlap = set(crop["pixel_id"].astype(str)) & set(noncrop["pixel_id"].astype(str))
    if overlap:
        crop = crop[~crop["pixel_id"].astype(str).isin(overlap)]
        noncrop = noncrop[~noncrop["pixel_id"].astype(str).isin(overlap)]
    crop["source_run_fingerprint"] = str(crop_run["run_fingerprint"])
    noncrop["source_run_fingerprint"] = str(noncrop_run["run_fingerprint"])
    frame = _concat([crop, noncrop], pd)
    if frame.empty:
        raise RuntimeError("no pixel embeddings remain after label filtering")
    block_metres = int(round(args.spatial_block_km * 1_000))
    if block_metres < 1:
        raise ValueError("spatial-block-km must be positive")
    x_metres = frame["pixel_x_index"].astype(np.int64) * PIXEL_SIZE_M
    y_metres = frame["pixel_y_index"].astype(np.int64) * PIXEL_SIZE_M
    frame["spatial_block"] = (
        "utm-"
        + frame["utm_epsg"].astype(np.int64).astype(str)
        + "-bx"
        + np.floor_divide(x_metres, block_metres).astype(str)
        + "-by"
        + np.floor_divide(y_metres, block_metres).astype(str)
    )
    crop_rows = frame["source"].eq("harvard")
    crop_block_counts = frame[crop_rows].groupby("field_uid")["spatial_block"].nunique()
    cross_block_fields = set(crop_block_counts[crop_block_counts.gt(1)].index.astype(str))
    cross_block_mask = crop_rows & frame["field_uid"].astype(str).isin(cross_block_fields)
    cross_block_pixels = int(cross_block_mask.sum())
    frame = frame[~cross_block_mask].copy()
    frame = _cap_per_class(frame, args.max_pixels_per_class, args.seed)
    ratios = (args.train_ratio, args.validation_ratio, args.test_ratio)
    frame, split_seed = choose_spatial_split(
        frame,
        ratios,
        args.seed,
        args.split_attempts,
    )
    labels = sorted(frame["label"].unique())
    class_ids = {label: index for index, label in enumerate(labels)}
    frame["class_id"] = frame["label"].map(class_ids).astype(np.int64)
    selected_columns = [
        "field_uid",
        "pixel_id",
        "utm_epsg",
        "pixel_x_index",
        "pixel_y_index",
        "pixel_longitude",
        "pixel_latitude",
        "window_id",
        "source",
        "source_run_fingerprint",
        "label",
        "class_id",
        "spatial_block",
        "split",
        "embedding",
    ]
    frame = frame[selected_columns].sort_values(
        ["split", "class_id", "pixel_id"], kind="stable"
    )
    output = Path(args.output).expanduser()
    _write_dataset(frame, output, dependencies)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else output.with_suffix(
        output.suffix + ".manifest.json"
    )
    class_split_counts = (
        frame.groupby(["label", "split"]).size().unstack(fill_value=0).to_dict(orient="index")
    )
    payload = {
        "schema": "pixel-landcover-classification-dataset-v1",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(frame),
        "window_id": args.window_id,
        "embedding_contract": crop_contract,
        "class_ids": class_ids,
        "class_split_counts": class_split_counts,
        "source_audit": {
            "crop": crop_audit,
            "noncrop": noncrop_audit,
            "cross_source_overlap_removed": len(overlap),
            "cross_block_crop_fields_removed": len(cross_block_fields),
            "cross_block_crop_pixels_removed": cross_block_pixels,
        },
        "spatial_split": {
            "block_km": args.spatial_block_km,
            "requested_seed": args.seed,
            "selected_seed": split_seed,
            "ratios": dict(zip(SPLIT_NAMES, ratios, strict=True)),
            "blocks": int(frame["spatial_block"].nunique()),
        },
    }
    _write_json_atomic(manifest_path, payload)
    return {**payload, "manifest": str(manifest_path)}


PIXEL_SIZE_M = 10


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge pure-crop and WorldCover non-crop pixel embeddings"
    )
    parser.add_argument(
        "--crop-root",
        default="/mnt/noobjam/harvard_tessera_incremental_v2",
    )
    parser.add_argument(
        "--noncrop-root",
        default="/mnt/noobjam/rwanda_worldcover_mlp/tessera_embeddings",
    )
    parser.add_argument("--window-id", default="w2")
    parser.add_argument("--crop-labels", nargs="+", default=list(DEFAULT_CROP_LABELS))
    parser.add_argument("--max-pixels-per-class", type=int, default=10_000)
    parser.add_argument("--spatial-block-km", type=float, default=10.0)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--split-attempts", type=int, default=4_096)
    parser.add_argument("--seed", type=int, default=24_051_995)
    parser.add_argument(
        "--output",
        default="/mnt/noobjam/rwanda_worldcover_mlp/pixel_classification_w2.parquet",
    )
    parser.add_argument("--manifest")
    args = parser.parse_args()
    print(json.dumps(build(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
