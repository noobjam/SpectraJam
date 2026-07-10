from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import math
import os
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from spectrajam.candidate_frame import lattice_center, lattice_index_bounds, worldcover_tile_id
from spectrajam.contracts import ContractError, sha256_file
from spectrajam.frame_sources import FRAME_SOURCES


TARGET_EPSG = 32735
PIXEL_SIZE_M = 10
WORLD_COVER_LABELS = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare / sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss and lichen",
}
DEFAULT_NONCROP_CODES = (10, 20, 30, 50, 60, 80, 90)
TILE_KEYS = {
    "S03E027": "worldcover_2021_s03e027",
    "S03E030": "worldcover_2021_s03e030",
}


def _require_dependencies() -> dict[str, Any]:
    try:
        import geopandas as gpd
        import numpy as np
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        import pyproj
        import rasterio
        import shapely
        from shapely.ops import transform as transform_geometry
    except ImportError as error:  # pragma: no cover - exercised on the GPU VM
        raise RuntimeError(
            'install the data dependencies with python -m pip install -e ".[data]"'
        ) from error
    return {
        "gpd": gpd,
        "np": np,
        "pd": pd,
        "pa": pa,
        "pq": pq,
        "pyproj": pyproj,
        "rasterio": rasterio,
        "shapely": shapely,
        "transform_geometry": transform_geometry,
    }


def _verified_source(source_root: Path, key: str) -> Path:
    source = FRAME_SOURCES[key]
    path = source.destination(source_root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {key}: {path}; run download_rwanda_worldcover first"
        )
    if path.stat().st_size != source.artifact.expected_bytes:
        raise ContractError(f"unexpected byte count for {path}")
    if sha256_file(path) != source.artifact.sha256:
        raise ContractError(f"SHA-256 mismatch for {path}")
    return path


def _open_tiles(stack: ExitStack, source_root: Path, rasterio: Any) -> dict[str, Any]:
    datasets: dict[str, Any] = {}
    for tile, key in TILE_KEYS.items():
        path = _verified_source(source_root, key)
        dataset = stack.enter_context(rasterio.open(path))
        if (
            dataset.crs is None
            or dataset.crs.to_epsg() != 4326
            or dataset.count != 1
            or dataset.dtypes != ("uint8",)
            or dataset.nodata != 0
            or dataset.width != 36_000
            or dataset.height != 36_000
        ):
            raise ContractError(f"unexpected WorldCover raster contract: {path}")
        datasets[tile] = dataset
    return datasets


def _sample_worldcover(
    longitudes: Any,
    latitudes: Any,
    datasets: dict[str, Any],
    np: Any,
) -> Any:
    tile_ids = [
        worldcover_tile_id(float(lon), float(lat))
        for lon, lat in zip(longitudes, latitudes, strict=True)
    ]
    unknown = sorted(set(tile_ids) - set(datasets))
    if unknown:
        raise ContractError(f"Rwanda samples require unpinned WorldCover tiles: {unknown}")
    values = np.zeros(len(tile_ids), dtype=np.uint8)
    for tile in sorted(set(tile_ids)):
        indices = np.fromiter(
            (index for index, value in enumerate(tile_ids) if value == tile),
            dtype=np.int64,
        )
        coordinates = zip(longitudes[indices], latitudes[indices], strict=True)
        sampled = datasets[tile].sample(coordinates, indexes=1, masked=False)
        values[indices] = np.fromiter(
            (int(value[0]) for value in sampled),
            dtype=np.uint8,
            count=len(indices),
        )
    invalid = sorted(set(map(int, values)) - set(WORLD_COVER_LABELS) - {0})
    if invalid:
        raise ContractError(f"WorldCover returned unknown class codes: {invalid}")
    return values


def _load_rwanda_boundary(source_root: Path, dependencies: dict[str, Any]) -> Any:
    gpd = dependencies["gpd"]
    path = _verified_source(source_root, "world_bank_admin0")
    frame = gpd.read_file(path, layer="WB_GAD_ADM0", engine="pyogrio")
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise ContractError("World Bank Admin 0 boundary must use EPSG:4326")
    selected = frame.loc[frame["ISO_A3"].eq("RWA")]
    if len(selected) != 1 or selected.iloc[0]["NAM_0"] != "Rwanda":
        raise ContractError("World Bank Admin 0 does not contain one canonical Rwanda row")
    metric = selected.to_crs(epsg=TARGET_EPSG).geometry.iloc[0]
    if metric.is_empty or not metric.is_valid:
        raise ContractError("projected Rwanda boundary is invalid")
    return metric


def _load_exclusion(
    path: Path | None,
    wkt_column: str,
    buffer_m: float,
    dependencies: dict[str, Any],
) -> Any | None:
    if path is None:
        return None
    pd = dependencies["pd"]
    gpd = dependencies["gpd"]
    shapely = dependencies["shapely"]
    if not path.is_file():
        raise FileNotFoundError(f"exclusion parquet not found: {path}")
    frame = pd.read_parquet(path, columns=[wkt_column])
    geometries = gpd.GeoSeries.from_wkt(frame[wkt_column], crs="EPSG:4326")
    geometries = geometries[geometries.notna() & ~geometries.is_empty]
    if geometries.empty:
        raise ContractError(f"no valid exclusion geometries in {path}")
    metric = geometries.to_crs(epsg=TARGET_EPSG)
    union = shapely.union_all(metric.to_numpy())
    return union.buffer(buffer_m) if buffer_m else union


def _rank(seed: int, pixel_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{pixel_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def _keep_smallest(
    heaps: dict[int, list[tuple[int, str, dict[str, Any]]]],
    class_code: int,
    record: dict[str, Any],
    limit: int,
    seed: int,
) -> None:
    pixel_id = str(record["pixel_id"])
    entry = (-_rank(seed, pixel_id), pixel_id, record)
    heap = heaps.setdefault(class_code, [])
    heapq.heappush(heap, entry)
    if len(heap) > limit:
        heapq.heappop(heap)


def _write_parquet_atomic(frame: Any, path: Path, dependencies: dict[str, Any]) -> None:
    pa = dependencies["pa"]
    pq = dependencies["pq"]
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {
            **(table.schema.metadata or {}),
            b"artifact": b"rwanda_worldcover_2021_noncrop_wkt",
            b"schema_version": b"1",
        }
    )
    part = path.with_suffix(path.suffix + ".part")
    pq.write_table(table, part, compression="zstd")
    if pq.read_metadata(part).num_rows != len(frame):
        raise RuntimeError(f"Parquet row-count validation failed: {part}")
    part.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    part.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(part, path)


def _patch_geometry(record: dict[str, Any], width_m: int, shapely: Any) -> Any:
    cells = width_m // PIXEL_SIZE_M
    left_index = int(record["pixel_x_index"]) - cells // 2
    bottom_index = int(record["pixel_y_index"]) - cells // 2
    return shapely.box(
        left_index * PIXEL_SIZE_M,
        bottom_index * PIXEL_SIZE_M,
        (left_index + cells) * PIXEL_SIZE_M,
        (bottom_index + cells) * PIXEL_SIZE_M,
    )


def _patch_is_pure(
    record: dict[str, Any],
    class_code: int,
    width_m: int,
    purity_radius_m: float,
    inverse: Any,
    datasets: dict[str, Any],
    np: Any,
) -> bool:
    """Require one WorldCover class across every patch cell and its purity halo."""
    cells = width_m // PIXEL_SIZE_M
    halo_cells = math.ceil(purity_radius_m / PIXEL_SIZE_M)
    left_index = int(record["pixel_x_index"]) - cells // 2 - halo_cells
    bottom_index = int(record["pixel_y_index"]) - cells // 2 - halo_cells
    indices = np.arange(cells + 2 * halo_cells, dtype=np.int64)
    x_indices, y_indices = np.meshgrid(left_index + indices, bottom_index + indices)
    xs = (x_indices.reshape(-1).astype(np.float64) + 0.5) * PIXEL_SIZE_M
    ys = (y_indices.reshape(-1).astype(np.float64) + 0.5) * PIXEL_SIZE_M
    longitudes, latitudes = inverse.transform(xs, ys)
    values = _sample_worldcover(
        np.asarray(longitudes, dtype=np.float64),
        np.asarray(latitudes, dtype=np.float64),
        datasets,
        np,
    )
    return bool(np.all(values == class_code))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    if args.samples_per_class < 1 or args.lattice_spacing_m < 20:
        raise ValueError("samples-per-class must be positive and lattice spacing must be >= 20 m")
    if args.chunk_size < 1:
        raise ValueError("chunk-size must be positive")
    if args.purity_radius_m < 0 or args.exclusion_buffer_m < 0:
        raise ValueError("purity and exclusion distances must be non-negative")
    if args.patch_width_m:
        if args.patch_width_m < PIXEL_SIZE_M or args.patch_width_m % PIXEL_SIZE_M:
            raise ValueError("patch-width-m must be a positive multiple of 10 m")
    if args.patch_candidate_multiplier < 1:
        raise ValueError("patch-candidate-multiplier must be positive")
    requested_codes = tuple(args.class_codes or DEFAULT_NONCROP_CODES)
    unknown = sorted(set(requested_codes) - set(DEFAULT_NONCROP_CODES))
    if unknown:
        raise ValueError(f"non-crop class codes are invalid: {unknown}")

    dependencies = _require_dependencies()
    np = dependencies["np"]
    pd = dependencies["pd"]
    pyproj = dependencies["pyproj"]
    shapely = dependencies["shapely"]
    transform_geometry = dependencies["transform_geometry"]

    source_root = Path(args.source_root).expanduser()
    output = Path(args.output).expanduser()
    boundary = _load_rwanda_boundary(source_root, dependencies)
    exclusion_path = (
        None
        if args.exclude_wkt_parquet is None
        else Path(args.exclude_wkt_parquet).expanduser()
    )
    patch_margin_m = math.sqrt(2.0) * args.patch_width_m / 2.0
    exclusion = _load_exclusion(
        exclusion_path,
        args.wkt_column,
        args.exclusion_buffer_m + patch_margin_m,
        dependencies,
    )
    sampling_boundary = boundary.buffer(
        -math.sqrt(2.0) * (args.purity_radius_m + args.patch_width_m / 2.0)
    )
    if sampling_boundary.is_empty:
        raise ContractError("purity radius removes the complete Rwanda sampling extent")
    inverse = pyproj.Transformer.from_crs(TARGET_EPSG, 4326, always_xy=True)

    min_x, min_y, max_x, max_y = boundary.bounds
    x_first, x_last = lattice_index_bounds(min_x, max_x, args.lattice_spacing_m)
    y_first, y_last = lattice_index_bounds(min_y, max_y, args.lattice_spacing_m)
    x_count = x_last - x_first + 1
    envelope_count = x_count * (y_last - y_first + 1)
    heaps: dict[int, list[tuple[int, str, dict[str, Any]]]] = {}
    candidate_limit = args.samples_per_class * (
        args.patch_candidate_multiplier if args.patch_width_m else 1
    )
    pure_counts: Counter[int] = Counter()
    inspected = 0

    with ExitStack() as stack:
        datasets = _open_tiles(stack, source_root, dependencies["rasterio"])
        for start in range(0, envelope_count, args.chunk_size):
            stop = min(start + args.chunk_size, envelope_count)
            flat = np.arange(start, stop, dtype=np.int64)
            lattice_x = np.asarray(
                [
                    lattice_center(int(value), args.lattice_spacing_m)
                    for value in x_first + flat % x_count
                ]
            )
            lattice_y = np.asarray(
                [
                    lattice_center(int(value), args.lattice_spacing_m)
                    for value in y_first + flat // x_count
                ]
            )
            pixel_x_index = np.floor(lattice_x / PIXEL_SIZE_M).astype(np.int64)
            pixel_y_index = np.floor(lattice_y / PIXEL_SIZE_M).astype(np.int64)
            xs = (pixel_x_index.astype(np.float64) + 0.5) * PIXEL_SIZE_M
            ys = (pixel_y_index.astype(np.float64) + 0.5) * PIXEL_SIZE_M
            points = shapely.points(xs, ys)
            keep = shapely.covers(sampling_boundary, points)
            if exclusion is not None:
                keep &= ~shapely.intersects(exclusion, points)
            if not keep.any():
                continue
            xs = xs[keep]
            ys = ys[keep]
            pixel_x_index = pixel_x_index[keep]
            pixel_y_index = pixel_y_index[keep]
            longitudes, latitudes = inverse.transform(xs, ys)
            longitudes = np.asarray(longitudes, dtype=np.float64)
            latitudes = np.asarray(latitudes, dtype=np.float64)
            values = _sample_worldcover(longitudes, latitudes, datasets, np)
            pure = np.isin(values, requested_codes)
            radius = float(args.purity_radius_m)
            offsets = (-radius, 0.0, radius) if radius else (0.0,)
            for dx in offsets:
                for dy in offsets:
                    if dx == 0 and dy == 0:
                        continue
                    neighbor_lon, neighbor_lat = inverse.transform(xs + dx, ys + dy)
                    neighbor = _sample_worldcover(
                        np.asarray(neighbor_lon),
                        np.asarray(neighbor_lat),
                        datasets,
                        np,
                    )
                    pure &= neighbor == values
            inspected += len(values)
            for position in np.flatnonzero(pure):
                class_code = int(values[position])
                x_index = int(pixel_x_index[position])
                y_index = int(pixel_y_index[position])
                pixel_id = f"utm-{TARGET_EPSG}-{PIXEL_SIZE_M}m-{x_index}-{y_index}"
                record = {
                    "pixel_id": pixel_id,
                    "pixel_x_index": x_index,
                    "pixel_y_index": y_index,
                    "x": float(xs[position]),
                    "y": float(ys[position]),
                    "longitude": float(longitudes[position]),
                    "latitude": float(latitudes[position]),
                    "worldcover_class": class_code,
                }
                pure_counts[class_code] += 1
                _keep_smallest(
                    heaps,
                    class_code,
                    record,
                    candidate_limit,
                    args.seed,
                )

        selected_records: list[dict[str, Any]] = []
        for class_code in sorted(heaps):
            selected_for_class = 0
            for _, _, record in sorted(heaps[class_code], key=lambda value: (-value[0], value[1])):
                if args.patch_width_m and not _patch_is_pure(
                    record,
                    class_code,
                    args.patch_width_m,
                    args.purity_radius_m,
                    inverse,
                    datasets,
                    np,
                ):
                    continue
                selected_records.append(record)
                selected_for_class += 1
                if selected_for_class == args.samples_per_class:
                    break

    if not selected_records:
        raise RuntimeError("no pure non-crop WorldCover samples were selected")
    forward_geometry = pyproj.Transformer.from_crs(TARGET_EPSG, 4326, always_xy=True)
    half_width = args.wkt_width_m / 2.0
    rows = []
    for record in selected_records:
        class_code = int(record["worldcover_class"])
        geometry_metric = (
            _patch_geometry(record, args.patch_width_m, shapely)
            if args.patch_width_m
            else shapely.box(
                record["x"] - half_width,
                record["y"] - half_width,
                record["x"] + half_width,
                record["y"] + half_width,
            )
        )
        geometry_wgs84 = transform_geometry(forward_geometry.transform, geometry_metric)
        rows.append(
            {
                "id": (
                    f"wc2021-{class_code:03d}-patch{args.patch_width_m}m-{record['pixel_id']}"
                    if args.patch_width_m
                    else f"wc2021-{class_code:03d}-{record['pixel_id']}"
                ),
                "landcover": f"Non-crop: {WORLD_COVER_LABELS[class_code]}",
                "wkt": geometry_wgs84.wkt,
                "LONGITUDE": record["longitude"],
                "LATITUDE": record["latitude"],
                "QUADKEY": "rwanda-worldcover-2021",
                "worldcover_class": class_code,
                "pixel_id": record["pixel_id"],
                "pixel_x_index": record["pixel_x_index"],
                "pixel_y_index": record["pixel_y_index"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["worldcover_class", "pixel_id"], kind="stable"
    )
    _write_parquet_atomic(frame, output, dependencies)
    manifest_path = Path(args.manifest).expanduser() if args.manifest else output.with_suffix(
        output.suffix + ".manifest.json"
    )
    class_counts = {
        str(code): int(count)
        for code, count in frame["worldcover_class"].value_counts().sort_index().items()
    }
    payload = {
        "schema": "rwanda-worldcover-noncrop-wkt-v1",
        "output": str(output),
        "output_sha256": sha256_file(output),
        "rows": len(frame),
        "parameters": {
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "class_codes": list(requested_codes),
            "lattice_spacing_m": args.lattice_spacing_m,
            "purity_radius_m": args.purity_radius_m,
            "wkt_width_m": args.wkt_width_m,
            "patch_width_m": args.patch_width_m,
            "patch_candidate_multiplier": args.patch_candidate_multiplier,
            "exclusion_buffer_m": args.exclusion_buffer_m,
            "exclude_wkt_parquet": None if exclusion_path is None else str(exclusion_path),
        },
        "inspected_inside_boundary": inspected,
        "pure_candidate_counts": {str(key): value for key, value in sorted(pure_counts.items())},
        "selected_class_counts": class_counts,
        "sources": {
            key: {
                "path": str(FRAME_SOURCES[key].destination(source_root)),
                "sha256": FRAME_SOURCES[key].artifact.sha256,
            }
            for key in ("world_bank_admin0", *TILE_KEYS.values())
        },
    }
    _write_json_atomic(manifest_path, payload)
    return {**payload, "manifest": str(manifest_path)}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create single-pixel WKT inputs for pure Rwanda WorldCover non-crop classes"
    )
    parser.add_argument(
        "--source-root",
        default="/mnt/noobjam/rwanda_worldcover_mlp/sources",
    )
    parser.add_argument(
        "--output",
        default="/mnt/noobjam/rwanda_worldcover_mlp/worldcover_noncrop_wkt.parquet",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--samples-per-class", type=int, default=2_000)
    parser.add_argument("--class-codes", type=int, nargs="+")
    parser.add_argument("--lattice-spacing-m", type=float, default=200.0)
    parser.add_argument("--purity-radius-m", type=float, default=20.0)
    parser.add_argument("--wkt-width-m", type=float, default=2.0)
    parser.add_argument(
        "--patch-width-m",
        type=int,
        default=0,
        help="emit pure square patches instead of single-pixel WKTs (multiple of 10 m)",
    )
    parser.add_argument(
        "--patch-candidate-multiplier",
        type=int,
        default=64,
        help="rank this many candidates per requested pure patch before validation",
    )
    parser.add_argument("--seed", type=int, default=24_051_995)
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--exclude-wkt-parquet")
    parser.add_argument("--wkt-column", default="wkt")
    parser.add_argument("--exclusion-buffer-m", type=float, default=30.0)
    args = parser.parse_args()
    if args.wkt_width_m <= 0 or args.wkt_width_m >= PIXEL_SIZE_M:
        parser.error("--wkt-width-m must be greater than zero and smaller than 10 m")
    print(json.dumps(prepare(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
