from __future__ import annotations

import json
import logging
from importlib import metadata
import platform
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

from . import PREPROCESSING_VERSION, SCHEMA_VERSION
from .catalog import MPCCatalog, snapshot_sha256
from .config import PipelineConfig
from .geometry import (
    PixelCell,
    RasterWindow,
    WorkTileKey,
    canonical_geometry_sha256,
    expand_projected_bounds,
    parse_field_geometry,
    pixel_cells_for_geometry,
    pixel_centers_wgs84,
    positive_area_pixel_count,
    project_geometry,
    projected_bounds_to_wgs84,
    raster_chunk_for_pixel,
    utm_epsg,
    work_tile_bounds,
    work_tile_for_pixel,
)
from .inference import PlainTesseraRunner
from .materialize import MPCMaterializer, PixelTimelines
from .storage import (
    EMBEDDING_COLUMNS,
    canonical_sha256,
    parquet_matches,
    sha256_file,
    write_dataframe_atomic,
    write_embedding_shard,
    write_json_atomic,
)


LOGGER = logging.getLogger(__name__)
MODULE_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = MODULE_ROOT.parent


def _text(value: object) -> str:
    return "" if pd.isna(value) else str(value)


def _required_columns(config: PipelineConfig) -> set[str]:
    return {
        config.longitude_column,
        config.latitude_column,
        config.quadkey_column,
        config.label_column,
        config.wkt_column,
        config.id_column,
    }


def validate_parquet_schema(config: PipelineConfig) -> tuple[str, ...]:
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required to inspect the ground-truth parquet") from error
    columns = tuple(pq.ParquetFile(config.input_parquet).schema_arrow.names)
    missing = _required_columns(config) - set(columns)
    if missing:
        raise ValueError(f"ground-truth parquet is missing columns: {sorted(missing)}")
    return columns


def validate_ground_truth_values(source: pd.DataFrame, config: PipelineConfig) -> None:
    for column in (
        config.id_column,
        config.label_column,
        config.quadkey_column,
        config.wkt_column,
    ):
        missing = source[column].isna() | source[column].astype(str).str.strip().eq("")
        if missing.any():
            rows = source.index[missing].tolist()[:10]
            raise ValueError(f"ground-truth column {column!r} is empty at rows {rows}")
    longitude = pd.to_numeric(source[config.longitude_column], errors="coerce").to_numpy()
    latitude = pd.to_numeric(source[config.latitude_column], errors="coerce").to_numpy()
    invalid = (
        ~np.isfinite(longitude)
        | ~np.isfinite(latitude)
        | (longitude < -180)
        | (longitude > 180)
        | (latitude < -90)
        | (latitude > 90)
    )
    if invalid.any():
        rows = source.index[invalid].tolist()[:10]
        raise ValueError(f"ground-truth coordinates are invalid at rows {rows}")


def _field_base_hash(
    source_id: str,
    label: str,
    quadkey: str,
    geometry_hash: str,
) -> str:
    return canonical_sha256(
        {
            "source_id": source_id,
            "landcover": label,
            "quadkey": quadkey,
            "geometry_sha256": geometry_hash,
        }
    )


def prepare_field_pixels(
    source: pd.DataFrame, config: PipelineConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    fields = source.copy().reset_index(drop=True)
    duplicate_counts: dict[str, int] = {}
    pixel_map: dict[str, PixelCell] = {}
    membership_records: list[dict[str, object]] = []
    audits: list[dict[str, object]] = []

    for row_number, row in fields.iterrows():
        source_id = _text(row[config.id_column])
        label = _text(row[config.label_column])
        quadkey = _text(row[config.quadkey_column])
        raw_wkt = row[config.wkt_column]
        longitude = float(row[config.longitude_column])
        latitude = float(row[config.latitude_column])
        status = "valid"
        reason = ""
        coordinate_status = "not_checked"
        geometry_hash = canonical_sha256({"unparsed_wkt": _text(raw_wkt)})
        epsg: int | None = None
        cells: tuple[PixelCell, ...] = ()
        area_m2: float | None = None
        bbox_width_m: float | None = None
        bbox_height_m: float | None = None
        positive_area_count: int | None = None
        try:
            geometry, geometry_status = parse_field_geometry(raw_wkt)
            status = geometry_status
            geometry_hash = canonical_geometry_sha256(geometry)
            min_lon, min_lat, max_lon, max_lat = geometry.bounds
            tolerance = 1e-9
            within_wkt_bounds = (
                min_lon - tolerance <= longitude <= max_lon + tolerance
                and min_lat - tolerance <= latitude <= max_lat + tolerance
            )
            coordinate_status = (
                "within_wkt_bounds" if within_wkt_bounds else "outside_wkt_bounds"
            )
            point = geometry.representative_point()
            epsg = utm_epsg(float(point.x), float(point.y))
            corner_epsgs = {
                utm_epsg(lon, lat)
                for lon in (min_lon, max_lon)
                for lat in (min_lat, max_lat)
            }
            if corner_epsgs != {epsg}:
                raise ValueError("field crosses a UTM zone or hemisphere boundary")
            projected = project_geometry(geometry, epsg)
            projected_min_x, projected_min_y, projected_max_x, projected_max_y = (
                projected.bounds
            )
            area_m2 = float(projected.area)
            bbox_width_m = float(projected_max_x - projected_min_x)
            bbox_height_m = float(projected_max_y - projected_min_y)
            cells = pixel_cells_for_geometry(projected, epsg, config.pixel_size_m)
            positive_area_count = positive_area_pixel_count(
                projected, config.pixel_size_m
            )
            if not cells:
                status = "zero_pixel_centers"
                reason = "no snapped 10 m pixel center is covered by the field"
        except RuntimeError:
            raise
        except Exception as error:
            status = "invalid_field"
            reason = str(error)

        base_hash = _field_base_hash(source_id, label, quadkey, geometry_hash)
        duplicate_ordinal = duplicate_counts.get(base_hash, 0)
        duplicate_counts[base_hash] = duplicate_ordinal + 1
        field_uid = f"{base_hash[:24]}-{duplicate_ordinal:04d}"
        for cell in cells:
            pixel_map.setdefault(cell.pixel_id, cell)
            membership_records.append(
                {
                    "field_uid": field_uid,
                    "source_id": source_id,
                    "landcover": label,
                    "quadkey": quadkey,
                    "pixel_id": cell.pixel_id,
                }
            )
        audits.append(
            {
                "field_uid": field_uid,
                "field_base_sha256": base_hash,
                "geometry_sha256": geometry_hash,
                "geometry_status": status,
                "geometry_reason": reason,
                "coordinate_status": coordinate_status,
                "utm_epsg": epsg,
                "area_m2": area_m2,
                "bbox_width_m": bbox_width_m,
                "bbox_height_m": bbox_height_m,
                "center_pixel_count": len(cells),
                "positive_area_pixel_count": positive_area_count,
                "pixel_count": len(cells),
                "duplicate_ordinal": duplicate_ordinal,
                "source_row_number": row_number,
            }
        )

    audit_frame = pd.DataFrame(audits)
    fields = pd.concat([fields, audit_frame.drop(columns=["source_row_number"])], axis=1)
    fields["duplicate_count"] = fields.groupby("field_base_sha256")["field_uid"].transform(
        "size"
    )
    if not pixel_map:
        pixels = pd.DataFrame(
            columns=[
                "pixel_id",
                "utm_epsg",
                "pixel_x_index",
                "pixel_y_index",
                "pixel_longitude",
                "pixel_latitude",
                "work_x_index",
                "work_y_index",
                "chunk_x_index",
                "chunk_y_index",
            ]
        )
        memberships = pd.DataFrame(
            columns=[
                "field_uid",
                "source_id",
                "landcover",
                "quadkey",
                "pixel_id",
                "utm_epsg",
                "pixel_x_index",
                "pixel_y_index",
                "pixel_longitude",
                "pixel_latitude",
                "work_x_index",
                "work_y_index",
                "chunk_x_index",
                "chunk_y_index",
                "overlap_field_count",
                "label_conflict",
            ]
        )
        return fields, pixels, memberships

    ordered_cells = sorted(pixel_map.values(), key=lambda cell: cell.pixel_id)
    lonlat = pixel_centers_wgs84(ordered_cells)
    pixel_records: list[dict[str, object]] = []
    for cell in ordered_cells:
        work = work_tile_for_pixel(cell, config.work_tile_m)
        chunk = raster_chunk_for_pixel(cell, config.raster_chunk_pixels)
        longitude, latitude = lonlat[cell.pixel_id]
        pixel_records.append(
            {
                "pixel_id": cell.pixel_id,
                "utm_epsg": cell.epsg,
                "pixel_x_index": cell.x_index,
                "pixel_y_index": cell.y_index,
                "pixel_longitude": longitude,
                "pixel_latitude": latitude,
                "work_x_index": work.x_index,
                "work_y_index": work.y_index,
                "chunk_x_index": chunk.x_index,
                "chunk_y_index": chunk.y_index,
            }
        )
    pixels = pd.DataFrame(pixel_records)
    memberships = pd.DataFrame(membership_records).merge(
        pixels, on="pixel_id", how="left", validate="many_to_one"
    )
    overlap = memberships.groupby("pixel_id")["field_uid"].transform("nunique")
    conflicts = memberships.groupby("pixel_id")["landcover"].transform("nunique") > 1
    memberships["overlap_field_count"] = overlap.astype(np.int32)
    memberships["label_conflict"] = conflicts.astype(bool)
    return fields, pixels, memberships


def _config_fingerprint_payload(config: PipelineConfig) -> dict[str, Any]:
    payload = asdict(config)
    payload["input_parquet"] = str(config.input_parquet)
    payload["output_dir"] = str(config.output_dir)
    payload["checkpoint_path"] = str(config.checkpoint_path)
    payload["windows"] = [
        {
            "window_id": window.window_id,
            "ordinal": window.ordinal,
            "start": window.start.isoformat(),
            "end_exclusive": window.end_exclusive.isoformat(),
        }
        for window in config.windows
    ]
    return payload


def runtime_identity(config: PipelineConfig) -> dict[str, Any]:
    try:
        import pyproj
        import rasterio
        import torch
    except ImportError as error:
        raise RuntimeError(
            "runtime dependencies are missing; follow plain_tessera_incremental/README.md"
        ) from error
    resolved_device = config.device
    if resolved_device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    if resolved_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    distributions = (
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
        "rasterio",
        "pyproj",
        "shapely",
        "stackstac",
        "dask",
        "xarray",
        "pystac",
        "pystac-client",
        "planetary-computer",
    )
    versions: dict[str, str] = {}
    for name in distributions:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "missing"
    if any(value == "missing" for value in versions.values()):
        missing = sorted(name for name, value in versions.items() if value == "missing")
        raise RuntimeError(f"runtime dependencies are missing: {missing}")
    result: dict[str, Any] = {
        "resolved_device": resolved_device,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": versions,
        "gdal": rasterio.__gdal_version__,
        "proj": pyproj.proj_version_str,
        "torch_cuda": torch.version.cuda,
    }
    if resolved_device == "cuda":
        result["cuda_device_name"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return result


def code_identity() -> dict[str, str]:
    paths = sorted(MODULE_ROOT.glob("*.py")) + [
        REPOSITORY_ROOT / "src" / "spectrajam" / "models" / "tessera_v11.py",
        REPOSITORY_ROOT / "src" / "spectrajam" / "normalization.py",
    ]
    return {
        str(path.relative_to(REPOSITORY_ROOT)): sha256_file(path)
        for path in paths
    }


def preflight(config: PipelineConfig) -> dict[str, Any]:
    config.validate(require_files=True)
    columns = validate_parquet_schema(config)
    validation_frame = pd.read_parquet(
        config.input_parquet,
        columns=sorted(_required_columns(config)),
    )
    validate_ground_truth_values(validation_frame, config)
    _, preflight_pixels, preflight_memberships = prepare_field_pixels(
        validation_frame, config
    )
    task_columns = [
        "utm_epsg",
        "work_x_index",
        "work_y_index",
        "chunk_x_index",
        "chunk_y_index",
    ]
    task_count = int(preflight_memberships.groupby(task_columns).ngroups)
    runtime = runtime_identity(config)
    checkpoint_sha256 = sha256_file(config.checkpoint_path)
    if config.checkpoint_sha256 and checkpoint_sha256 != config.checkpoint_sha256:
        raise ValueError(
            "TESSERA checkpoint SHA-256 mismatch: "
            f"expected {config.checkpoint_sha256}, got {checkpoint_sha256}"
        )
    return {
        "input_parquet": str(config.input_parquet),
        "input_columns": list(columns),
        "input_rows": len(validation_frame),
        "unique_pixel_count": len(preflight_pixels),
        "field_pixel_membership_count": len(preflight_memberships),
        "estimated_task_count": task_count,
        "expected_embedding_rows": len(preflight_memberships) * len(config.windows),
        "checkpoint_path": str(config.checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "output_dir": str(config.output_dir),
        "runtime": runtime,
        "acquisition": {
            "stac_query_halo_m": config.stac_query_halo_m,
            "materialize_workers": config.materialize_workers,
        },
        "windows": [
            {
                "id": window.window_id,
                "start": window.start.isoformat(),
                "end_exclusive": window.end_exclusive.isoformat(),
                "duration_days": window.duration_days,
                "outside_annual_contract": window.duration_days > 366,
            }
            for window in config.windows
        ],
    }


def _load_or_materialize(
    path: Path,
    expected_pixel_ids: tuple[str, ...],
    materializer: MPCMaterializer,
    snapshot: dict[str, Any],
    grid: RasterWindow,
    rows: np.ndarray,
    columns: np.ndarray,
) -> PixelTimelines:
    if path.is_file():
        cached = PixelTimelines.load(path)
        if cached.pixel_ids != expected_pixel_ids:
            raise RuntimeError(f"timeline cache pixel order mismatch: {path}")
        return cached
    timelines = materializer.materialize(
        snapshot["s2_items"],
        snapshot["s1_items"],
        grid,
        expected_pixel_ids,
        rows,
        columns,
    )
    timelines.save(path)
    return timelines


def _task_shards(config: PipelineConfig, task_key: str) -> dict[str, Path]:
    return {
        window.window_id: config.output_dir
        / "embeddings"
        / f"window_id={window.window_id}"
        / f"{task_key}.parquet"
        for window in config.windows
    }


def run_pipeline(config: PipelineConfig) -> dict[str, Any]:
    checks = preflight(config)
    input_sha256 = sha256_file(config.input_parquet)
    checkpoint_sha256 = str(checks["checkpoint_sha256"])
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "preprocessing_version": PREPROCESSING_VERSION,
        "input_sha256": input_sha256,
        "checkpoint_sha256": checkpoint_sha256,
        "runtime": checks["runtime"],
        "code_sha256": code_identity(),
        "config": _config_fingerprint_payload(config),
    }
    run_fingerprint = canonical_sha256(fingerprint_payload)
    run_path = config.output_dir / "run.json"
    if run_path.is_file():
        existing = json.loads(run_path.read_text())
        if existing.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"output directory belongs to a different run: {config.output_dir}"
            )

    LOGGER.info("reading ground truth: %s", config.input_parquet)
    source = pd.read_parquet(config.input_parquet)
    validate_ground_truth_values(source, config)
    fields, pixels, memberships = prepare_field_pixels(source, config)
    status_counts = {
        str(key): int(value)
        for key, value in fields["geometry_status"].value_counts().to_dict().items()
    }
    coordinate_status_counts = {
        str(key): int(value)
        for key, value in fields["coordinate_status"].value_counts().to_dict().items()
    }
    outside_coordinate_count = coordinate_status_counts.get("outside_wkt_bounds", 0)
    if outside_coordinate_count:
        LOGGER.warning(
            "%d fields have LONGITUDE/LATITUDE outside their WKT bounds; "
            "WKT remains authoritative and the mismatch is recorded in fields.parquet",
            outside_coordinate_count,
        )
    invalid_fields = fields[fields["geometry_status"] == "invalid_field"]
    if not invalid_fields.empty:
        examples = invalid_fields[
            ["field_uid", config.id_column, "geometry_reason"]
        ].head(10)
        raise ValueError(
            "invalid ground-truth fields must be corrected before inference: "
            f"{examples.to_dict(orient='records')}"
        )
    expected_rows = len(memberships) * len(config.windows)
    manifest = {
        **fingerprint_payload,
        "run_fingerprint": run_fingerprint,
        "field_count": len(fields),
        "unique_pixel_count": len(pixels),
        "field_pixel_membership_count": len(memberships),
        "expected_embedding_rows": expected_rows,
        "geometry_status_counts": status_counts,
        "coordinate_status_counts": coordinate_status_counts,
        "outside_annual_contract_windows": [
            window.window_id for window in config.windows if window.duration_days > 366
        ],
    }
    if not run_path.is_file():
        write_json_atomic(run_path, manifest)
    write_dataframe_atomic(fields, config.output_dir / "fields.parquet", run_fingerprint, "fields")
    write_dataframe_atomic(pixels, config.output_dir / "pixels.parquet", run_fingerprint, "pixels")
    write_dataframe_atomic(
        memberships,
        config.output_dir / "field_pixels.parquet",
        run_fingerprint,
        "field_pixels",
    )
    if memberships.empty:
        completion = {
            "run_fingerprint": run_fingerprint,
            "completed": True,
            "embedding_rows": 0,
            "geometry_status_counts": status_counts,
            "coordinate_status_counts": coordinate_status_counts,
            "message": "no field contained a snapped 10 m pixel center",
        }
        write_json_atomic(config.output_dir / "COMPLETED.json", completion)
        return completion

    runner = PlainTesseraRunner(
        str(config.checkpoint_path),
        checkpoint_sha256,
        device=str(checks["runtime"]["resolved_device"]),
        batch_size=config.batch_size,
    )
    catalog = MPCCatalog(
        config.stac_endpoint,
        config.s2_collection,
        config.s1_collection,
        config.stac_request_retries,
    )
    materializer = MPCMaterializer(
        config.stack_chunksize,
        config.stac_request_retries,
        config.materialize_workers,
    )
    group_columns = [
        "utm_epsg",
        "work_x_index",
        "work_y_index",
        "chunk_x_index",
        "chunk_y_index",
    ]
    task_count = 0
    verified_embedding_rows = 0
    for group_key, task_memberships in memberships.groupby(group_columns, sort=True):
        epsg, work_x, work_y, chunk_x, chunk_y = (int(value) for value in group_key)
        task_identity = {
            "epsg": epsg,
            "work_x": work_x,
            "work_y": work_y,
            "chunk_x": chunk_x,
            "chunk_y": chunk_y,
        }
        task_key = canonical_sha256(task_identity)[:24]
        shards = _task_shards(config, task_key)
        unique_pixels = (
            task_memberships[
                ["pixel_id", "pixel_x_index", "pixel_y_index", "utm_epsg"]
            ]
            .drop_duplicates("pixel_id")
            .sort_values("pixel_id")
            .reset_index(drop=True)
        )
        pixel_ids = tuple(unique_pixels["pixel_id"].astype(str))
        position = {pixel_id: index for index, pixel_id in enumerate(pixel_ids)}
        task_memberships = task_memberships.copy()
        task_memberships["pixel_position"] = task_memberships["pixel_id"].map(position)
        task_pixels = tuple(
            PixelCell(epsg, int(row.pixel_x_index), int(row.pixel_y_index))
            for row in unique_pixels.itertuples(index=False)
        )
        grid = RasterWindow.from_pixels(task_pixels)
        local = [
            grid.local_indices(pixel)
            for pixel in task_pixels
        ]
        rows = np.asarray([value[0] for value in local], dtype=np.int64)
        columns = np.asarray([value[1] for value in local], dtype=np.int64)
        work = WorkTileKey(epsg, work_x, work_y)
        query_bounds = expand_projected_bounds(
            work_tile_bounds(work, config.work_tile_m), config.stac_query_halo_m
        )
        bbox = projected_bounds_to_wgs84(query_bounds, epsg)
        snapshot_path = config.output_dir / "stac" / f"{work.key}.json"
        snapshot = catalog.load_or_create_snapshot(
            snapshot_path,
            work.key,
            bbox,
            config.windows[0].start,
            config.windows[-1].end_exclusive,
        )
        cache_key = canonical_sha256(
            {
                "run_fingerprint": run_fingerprint,
                "task": task_identity,
                "pixel_ids": pixel_ids,
                "stac_snapshot_sha256": snapshot_sha256(snapshot),
                "preprocessing_version": PREPROCESSING_VERSION,
            }
        )
        if all(
            parquet_matches(
                path,
                {
                    "run_fingerprint": run_fingerprint,
                    "artifact": "pixel_embeddings",
                    "window_id": window_id,
                    "task_key": task_key,
                    "task_fingerprint": cache_key,
                },
                expected_rows=len(task_memberships),
                required_columns=EMBEDDING_COLUMNS,
            )
            for window_id, path in shards.items()
        ):
            LOGGER.info("resume: skipping complete task %s", task_key)
            task_count += 1
            verified_embedding_rows += len(task_memberships) * len(config.windows)
            continue
        cache_path = config.output_dir / "cache" / f"{cache_key}.npz"
        timelines = _load_or_materialize(
            cache_path,
            pixel_ids,
            materializer,
            snapshot,
            grid,
            rows,
            columns,
        )
        for window in config.windows:
            results = runner.embed_window(timelines, window)
            write_embedding_shard(
                task_memberships,
                results,
                window,
                shards[window.window_id],
                run_fingerprint,
                task_key,
                cache_key,
            )
        task_count += 1
        verified_embedding_rows += len(task_memberships) * len(config.windows)
        LOGGER.info("completed task %s (%d unique pixels)", task_key, len(pixel_ids))

    if verified_embedding_rows != expected_rows:
        raise RuntimeError(
            "final embedding row count does not match field-pixel × window cardinality: "
            f"{verified_embedding_rows} != {expected_rows}"
        )
    completion = {
        "run_fingerprint": run_fingerprint,
        "completed": True,
        "field_count": len(fields),
        "unique_pixel_count": len(pixels),
        "field_pixel_membership_count": len(memberships),
        "embedding_rows": verified_embedding_rows,
        "geometry_status_counts": status_counts,
        "coordinate_status_counts": coordinate_status_counts,
        "task_count": task_count,
    }
    write_json_atomic(config.output_dir / "COMPLETED.json", completion)
    return completion
