from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from spectrajam.contracts import sha256_file


DEFAULT_CROP_LABELS = ("Bean", "Irish Potato", "Maize", "Rice")
SOURCE_COLUMNS = ("LONGITUDE", "LATITUDE", "QUADKEY", "landcover", "wkt", "id")
AUDIT_COLUMNS = (
    "field_uid",
    "geometry_sha256",
    "geometry_status",
    "center_pixel_count",
)


def _require_dependencies() -> dict[str, Any]:
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:  # pragma: no cover - exercised on the GPU VM
        raise RuntimeError("pandas and pyarrow are required to select large fields") from error
    return {"pd": pd, "pa": pa, "pq": pq}


def _write_parquet_atomic(
    frame: Any,
    path: Path,
    dependencies: dict[str, Any],
) -> None:
    pa = dependencies["pa"]
    pq = dependencies["pq"]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"artifact": b"harvard_large_field_input",
            b"schema_version": b"1",
        }
    )
    temporary = path.with_suffix(path.suffix + ".part")
    pq.write_table(table, temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(frame):
        raise RuntimeError(f"Parquet row-count validation failed: {temporary}")
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.min_pixels < 1 or args.max_fields_per_class < 1:
        raise ValueError("min-pixels and max-fields-per-class must be positive")
    labels = tuple(str(value).strip() for value in args.crop_labels)
    if not labels or any(not value for value in labels) or len(labels) != len(set(labels)):
        raise ValueError("crop-labels must contain unique non-empty values")

    dependencies = _require_dependencies()
    pd = dependencies["pd"]
    fields_path = Path(args.fields).expanduser()
    if not fields_path.is_file():
        raise FileNotFoundError(f"audited fields parquet not found: {fields_path}")
    frame = pd.read_parquet(fields_path)
    required = set(SOURCE_COLUMNS) | set(AUDIT_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"audited fields parquet is missing columns: {sorted(missing)}")

    frame = frame.copy()
    frame["landcover"] = frame["landcover"].astype(str).str.strip()
    frame["center_pixel_count"] = pd.to_numeric(
        frame["center_pixel_count"], errors="coerce"
    )
    qualified = frame[
        frame["geometry_status"].isin({"valid", "repaired"})
        & frame["landcover"].isin(labels)
        & frame["center_pixel_count"].ge(args.min_pixels)
    ].copy()
    geometry_label_counts = qualified.groupby("geometry_sha256")[
        "landcover"
    ].nunique()
    conflicting_geometries = set(
        geometry_label_counts[geometry_label_counts.gt(1)].index
    )
    conflicting_rows = qualified["geometry_sha256"].isin(conflicting_geometries)
    candidates = qualified[~conflicting_rows].sort_values(
        ["landcover", "center_pixel_count", "field_uid"],
        ascending=[True, False, True],
        kind="stable",
    )
    candidate_count_before_deduplication = len(candidates)
    candidates = candidates.drop_duplicates(
        ["landcover", "geometry_sha256"],
        keep="first",
    )
    excluded_candidate_rows = {
        "cross_label_geometry_conflict": int(conflicting_rows.sum()),
        "duplicate_geometry": candidate_count_before_deduplication - len(candidates),
    }

    selected_parts = []
    available_field_counts: dict[str, int] = {}
    available_pixel_estimates: dict[str, int] = {}
    selected_field_counts: dict[str, int] = {}
    selected_pixel_estimates: dict[str, int] = {}
    for label in labels:
        available = candidates[candidates["landcover"].eq(label)].sort_values(
            ["center_pixel_count", "field_uid"],
            ascending=[False, True],
            kind="stable",
        )
        if available.empty:
            raise RuntimeError(
                f"no valid {label!r} field contains at least {args.min_pixels} pixels"
            )
        selected = available.head(args.max_fields_per_class)
        selected_parts.append(selected)
        available_field_counts[label] = len(available)
        available_pixel_estimates[label] = int(available["center_pixel_count"].sum())
        selected_field_counts[label] = len(selected)
        selected_pixel_estimates[label] = int(selected["center_pixel_count"].sum())

    output_frame = pd.concat(selected_parts, ignore_index=True)[list(SOURCE_COLUMNS)]
    output = Path(args.output).expanduser()
    _write_parquet_atomic(output_frame, output, dependencies)
    manifest_path = (
        Path(args.manifest).expanduser()
        if args.manifest
        else output.with_suffix(output.suffix + ".manifest.json")
    )
    payload = {
        "schema": "harvard-large-field-input-v1",
        "source_fields": str(fields_path),
        "source_fields_sha256": sha256_file(fields_path),
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(output_frame),
        "parameters": {
            "min_pixels": args.min_pixels,
            "max_fields_per_class": args.max_fields_per_class,
            "crop_labels": list(labels),
        },
        "excluded_candidate_rows": excluded_candidate_rows,
        "available_field_counts": available_field_counts,
        "available_pixel_count_estimates": available_pixel_estimates,
        "selected_field_counts": selected_field_counts,
        "selected_pixel_count_estimates": selected_pixel_estimates,
    }
    _write_json_atomic(manifest_path, payload)
    return {**payload, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the largest audited Harvard polygons for a fast w2 AOI run"
    )
    parser.add_argument(
        "--fields",
        default="/mnt/noobjam/harvard_tessera_incremental_v3/fields.parquet",
    )
    parser.add_argument("--min-pixels", type=int, default=32)
    parser.add_argument("--max-fields-per-class", type=int, default=25)
    parser.add_argument("--crop-labels", nargs="+", default=list(DEFAULT_CROP_LABELS))
    parser.add_argument(
        "--output",
        default="/mnt/noobjam/harvard_large_fields_w2/harvard_large_fields.parquet",
    )
    parser.add_argument("--manifest")
    args = parser.parse_args()
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
