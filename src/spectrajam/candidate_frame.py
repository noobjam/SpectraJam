from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import zipfile
from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import frame_sources as frame_sources_module
from .config import ExperimentConfig
from .contracts import ContractError, sha256_file
from .frame_sources import (
    FRAME_SOURCES,
    FrameSource,
    frame_operation_lock,
    verify_frame_sources,
)

COUNTRY_GRID_CRS = {"RWA": "EPSG:32735", "ISR": "EPSG:32636"}
COUNTRY_BOUNDARY_IDENTITY = {
    "RWA": {"WB_STATUS": "Member State", "NAM_0": "Rwanda"},
    "ISR": {"WB_STATUS": "Member State", "NAM_0": "Israel"},
}
WORLD_COVER_CLASSES = frozenset({10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100})
WORLD_COVER_EXCLUDED = frozenset({0, 80})
MAX_MISSING_ECOREGION_RATE = 0.01
RESOLVE_MEMBERS = {
    "Ecoregions2017.dbf": "036987c540eef2dfd6e8a0678f3b85193649b3f63f545463dfcb2fed4f973ab5",
    "Ecoregions2017.prj": "db2f4b2a30c81a88981162acd3ff4629b86350834194efed1a6650446ccaecf4",
    "Ecoregions2017.shp": "d564235263d7a2f118cbb4fcb6e7593ef5f8c2f3600604d46047aa2712a83574",
    "Ecoregions2017.shx": "140342bc19d04a6f1ee11d1099a6c115f8c076f1396c6f2a5a55182471ff4c42",
}
_IMPLEMENTATION_HASHES = {
    "candidate_frame_sha256": sha256_file(Path(__file__)),
    "frame_sources_sha256": sha256_file(Path(frame_sources_module.__file__)),
}


@dataclass(frozen=True, slots=True)
class CandidateFrameResult:
    output: Path
    receipt: Path
    output_sha256: str
    candidates_by_country: dict[str, int]
    exclusions_by_country: dict[str, dict[str, int]]


def lattice_index_bounds(
    minimum: float, maximum: float, spacing_m: float
) -> tuple[int, int]:
    """Return inclusive indices for globally anchored projected cell centers."""
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ContractError("invalid projected lattice bounds")
    if not math.isfinite(spacing_m) or spacing_m <= 0:
        raise ContractError("lattice spacing must be positive")
    first = math.ceil(minimum / spacing_m - 0.5)
    last = math.floor(maximum / spacing_m - 0.5)
    if first > last:
        raise ContractError("projected extent contains no lattice centers")
    return first, last


def lattice_center(index: int, spacing_m: float) -> float:
    return (index + 0.5) * spacing_m


def worldcover_tile_id(longitude: float, latitude: float) -> str:
    """Map a WGS84 point to the half-open 3 degree WorldCover tile grid."""
    if not (-180 <= longitude < 180 and -90 < latitude < 90):
        raise ContractError(
            f"invalid WGS84 coordinate for WorldCover: ({longitude}, {latitude})"
        )
    west = math.floor(longitude / 3) * 3
    # North-up rasters include their top edge and exclude their bottom edge.
    # Thus an exact 30 N point belongs to the N27 tile, not N30.
    south = math.ceil(latitude / 3) * 3 - 3
    return (
        f"{'N' if south >= 0 else 'S'}{abs(south):02d}"
        f"{'E' if west >= 0 else 'W'}{abs(west):03d}"
    )


def candidate_id(country: str, epsg: int, x_index: int, y_index: int) -> str:
    return f"{country}-e{epsg}-x{x_index}-y{y_index}"


def spatial_block_id(
    country: str,
    epsg: int,
    x: float,
    y: float,
    block_size_m: float,
) -> str:
    if block_size_m <= 0:
        raise ContractError("spatial block size must be positive")
    return (
        f"{country}-e{epsg}-bx{math.floor(x / block_size_m)}"
        f"-by{math.floor(y / block_size_m)}"
    )


def _source_path(root: Path, key: str) -> Path:
    return FRAME_SOURCES[key].destination(root)


def frame_source_paths(root: str | Path) -> dict[str, Path]:
    source_root = Path(root)
    return {key: source.destination(source_root) for key, source in FRAME_SOURCES.items()}


def _require_optional_dependencies() -> dict[str, Any]:
    try:
        import geopandas as gpd
        import numpy as np
        import pyogrio
        import pyproj
        import rasterio
        import shapely
        from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
        from shapely.ops import transform, unary_union
        from shapely.strtree import STRtree
    except ImportError as error:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "install spectrajam[data] to construct the candidate frame"
        ) from error
    return {
        "gpd": gpd,
        "np": np,
        "pyogrio": pyogrio,
        "pyproj": pyproj,
        "rasterio": rasterio,
        "shapely": shapely,
        "GeometryCollection": GeometryCollection,
        "MultiPolygon": MultiPolygon,
        "Polygon": Polygon,
        "transform": transform,
        "unary_union": unary_union,
        "STRtree": STRtree,
    }


def _polygonal_part(geometry: Any, dependencies: Mapping[str, Any]) -> Any:
    Polygon = dependencies["Polygon"]
    MultiPolygon = dependencies["MultiPolygon"]
    GeometryCollection = dependencies["GeometryCollection"]
    unary_union = dependencies["unary_union"]
    if isinstance(geometry, (Polygon, MultiPolygon)):
        return geometry
    if isinstance(geometry, GeometryCollection):
        polygonal = [
            _polygonal_part(part, dependencies)
            for part in geometry.geoms
            if isinstance(part, (Polygon, MultiPolygon, GeometryCollection))
        ]
        polygonal = [part for part in polygonal if not part.is_empty]
        return unary_union(polygonal) if polygonal else Polygon()
    return Polygon()


def _make_polygonal_valid(geometry: Any, dependencies: Mapping[str, Any]) -> Any:
    shapely = dependencies["shapely"]
    repaired = geometry if geometry.is_valid else shapely.make_valid(geometry)
    return _polygonal_part(repaired, dependencies)


def _load_boundaries(
    config: ExperimentConfig,
    admin0_path: Path,
    ndlsa_path: Path,
    dependencies: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, int]]:
    gpd = dependencies["gpd"]
    admin0 = gpd.read_file(admin0_path, layer="WB_GAD_ADM0", engine="pyogrio")
    if admin0.crs is None or admin0.crs.to_epsg() != 4326:
        raise ContractError("World Bank ADM0 must use EPSG:4326")

    ndlsa = gpd.read_file(
        ndlsa_path, layer="WB_GAD_ADM0_NDLSA", engine="pyogrio"
    )
    if ndlsa.crs is None or ndlsa.crs.to_epsg() != 4326 or len(ndlsa) != 24:
        raise ContractError("World Bank NDLSA must contain 24 EPSG:4326 features")
    if set(ndlsa["WB_STATUS"]) != {"Non-determined legal status area"}:
        raise ContractError("World Bank NDLSA contains an unexpected status")
    ndlsa_geometries = [
        _make_polygonal_valid(value, dependencies) for value in ndlsa.geometry
    ]
    ndlsa_union = dependencies["unary_union"](ndlsa_geometries)

    boundaries: dict[str, Any] = {}
    touches: dict[str, int] = {}
    for country in config.countries:
        identity = COUNTRY_BOUNDARY_IDENTITY[country]
        selected = admin0.loc[admin0["ISO_A3"] == country]
        if len(selected) != 1:
            raise ContractError(
                f"World Bank ADM0 contains {len(selected)} rows for {country}; expected one"
            )
        row = selected.iloc[0]
        for field, expected in identity.items():
            if row[field] != expected:
                raise ContractError(
                    f"World Bank ADM0 {country} {field}={row[field]!r}, expected {expected!r}"
                )
        boundary = _make_polygonal_valid(row.geometry, dependencies)
        if boundary.is_empty:
            raise ContractError(f"World Bank ADM0 {country} geometry is empty")

        touching = sum(boundary.touches(value) for value in ndlsa_geometries)
        overlap = boundary.intersection(ndlsa_union)
        if not overlap.is_empty and overlap.area > 1e-12:
            raise ContractError(
                f"World Bank ADM0 {country} overlaps the excluded NDLSA fabric"
            )
        effective = boundary.difference(ndlsa_union)
        effective = _make_polygonal_valid(effective, dependencies)
        if effective.is_empty or not effective.is_valid:
            raise ContractError(f"effective {country} boundary is invalid")
        boundaries[country] = effective
        touches[country] = int(touching)
    return boundaries, touches


def _extract_resolve_archive(archive: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        names = set(source.namelist())
        if names != set(RESOLVE_MEMBERS):
            raise ContractError(
                f"RESOLVE archive members differ: {sorted(names)}"
            )
        for name, expected_sha256 in RESOLVE_MEMBERS.items():
            target = destination / name
            with source.open(name) as compressed, target.open("wb") as output:
                shutil.copyfileobj(compressed, output, length=1024 * 1024)
            if sha256_file(target) != expected_sha256:
                raise ContractError(f"RESOLVE member checksum mismatch: {name}")
    return destination / "Ecoregions2017.shp"


def _load_ecoregions(
    archive: Path, dependencies: Mapping[str, Any]
) -> tuple[Any, int]:
    gpd = dependencies["gpd"]
    with tempfile.TemporaryDirectory(prefix="spectrajam-resolve-") as temporary:
        shape_path = _extract_resolve_archive(archive, Path(temporary))
        frame = gpd.read_file(
            shape_path, encoding="ISO-8859-1", engine="pyogrio"
        )
    if frame.crs is None or frame.crs.to_epsg() != 4326:
        raise ContractError("RESOLVE Ecoregions 2017 must use EPSG:4326")
    if len(frame) != 847 or set(frame["ECO_ID"].astype(int)) != set(range(847)):
        raise ContractError("RESOLVE Ecoregions 2017 feature identity has changed")
    if frame["ECO_ID"].duplicated().any():
        raise ContractError("RESOLVE ECO_ID values must be unique")
    required = {"ECO_ID", "ECO_NAME", "BIOME_NUM", "BIOME_NAME", "REALM"}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(f"RESOLVE Ecoregions is missing fields: {sorted(missing)}")

    repair_count = int((~frame.geometry.is_valid).sum())
    frame = frame[[*sorted(required), "geometry"]].copy()
    frame["geometry"] = [
        _make_polygonal_valid(value, dependencies) for value in frame.geometry
    ]
    if frame.geometry.is_empty.any() or not frame.geometry.is_valid.all():
        raise ContractError("RESOLVE geometry repair did not produce valid polygons")
    frame["ECO_ID"] = frame["ECO_ID"].astype(int)
    return frame.sort_values("ECO_ID").reset_index(drop=True), repair_count


def _validate_worldcover_grid(path: Path) -> None:
    with path.open() as stream:
        grid = json.load(stream)
    features = grid.get("features", [])
    if len(features) != 2651:
        raise ContractError("ESA WorldCover grid must contain 2,651 tiles")
    tiles = {feature.get("properties", {}).get("ll_tile") for feature in features}
    required = {"S03E027", "S03E030", "N27E033", "N30E033", "N33E033"}
    if not required <= tiles:
        raise ContractError("ESA WorldCover grid is missing a country-covering tile")


def _worldcover_sources() -> dict[str, FrameSource]:
    result = {}
    for source in FRAME_SOURCES.values():
        metadata = dict(source.metadata)
        if source.role == "land-cover-map":
            result[str(metadata["tile"])] = source
    return result


def _worldcover_tile_bounds(tile: str) -> tuple[int, int, int, int]:
    south = int(tile[1:3]) * (1 if tile[0] == "N" else -1)
    west = int(tile[4:7]) * (1 if tile[3] == "E" else -1)
    return west, south, west + 3, south + 3


def _open_worldcover(
    stack: ExitStack,
    source_root: Path,
    dependencies: Mapping[str, Any],
) -> dict[str, Any]:
    rasterio = dependencies["rasterio"]
    datasets = {}
    for tile, source in _worldcover_sources().items():
        dataset = stack.enter_context(rasterio.open(source.destination(source_root)))
        expected_bounds = _worldcover_tile_bounds(tile)
        expected_step = 1 / 12_000
        if (
            dataset.crs is None
            or dataset.crs.to_epsg() != 4326
            or dataset.count != 1
            or dataset.dtypes != ("uint8",)
            or dataset.nodata != 0
            or dataset.width != 36_000
            or dataset.height != 36_000
            or any(
                not math.isclose(actual, expected, abs_tol=1e-10)
                for actual, expected in zip(
                    dataset.bounds, expected_bounds, strict=True
                )
            )
            or not math.isclose(dataset.transform.a, expected_step, abs_tol=1e-12)
            or not math.isclose(dataset.transform.b, 0, abs_tol=1e-12)
            or not math.isclose(dataset.transform.d, 0, abs_tol=1e-12)
            or not math.isclose(dataset.transform.e, -expected_step, abs_tol=1e-12)
            or not math.isclose(dataset.transform.c, expected_bounds[0], abs_tol=1e-10)
            or not math.isclose(dataset.transform.f, expected_bounds[3], abs_tol=1e-10)
        ):
            raise ContractError(f"WorldCover tile {tile} has an unexpected raster contract")
        datasets[tile] = dataset
    return datasets


def _sample_worldcover(
    longitudes: Any,
    latitudes: Any,
    datasets: Mapping[str, Any],
    dependencies: Mapping[str, Any],
) -> Any:
    np = dependencies["np"]
    tile_ids = [
        worldcover_tile_id(float(lon), float(lat))
        for lon, lat in zip(longitudes, latitudes, strict=True)
    ]
    unknown_tiles = sorted(set(tile_ids) - set(datasets))
    if unknown_tiles:
        raise ContractError(f"candidate points require unpinned WorldCover tiles: {unknown_tiles}")
    values = np.zeros(len(tile_ids), dtype=np.uint8)
    for tile in sorted(set(tile_ids)):
        indices = np.fromiter(
            (index for index, value in enumerate(tile_ids) if value == tile),
            dtype=np.int64,
        )
        coordinates = zip(longitudes[indices], latitudes[indices], strict=True)
        sampled = datasets[tile].sample(coordinates, indexes=1, masked=False)
        values[indices] = np.fromiter(
            (int(value[0]) for value in sampled), dtype=np.uint8, count=len(indices)
        )
    invalid = sorted(set(map(int, values)) - WORLD_COVER_CLASSES - {0})
    if invalid:
        raise ContractError(f"WorldCover returned unknown class codes: {invalid}")
    return values


def _assign_ecoregions(
    longitudes: Any,
    latitudes: Any,
    ecoregions: Any,
    tree: Any,
    dependencies: Mapping[str, Any],
) -> tuple[Any, int]:
    np = dependencies["np"]
    points = dependencies["shapely"].points(longitudes, latitudes)
    pairs = tree.query(points, predicate="intersects")
    assignments = np.full(len(points), -1, dtype=np.int32)
    if pairs.shape[1]:
        point_indices, geometry_indices = pairs
        eco_ids = ecoregions["ECO_ID"].to_numpy(dtype=np.int32)
        order = np.lexsort((eco_ids[geometry_indices], point_indices))
        sorted_points = point_indices[order]
        sorted_eco_ids = eco_ids[geometry_indices[order]]
        unique_points, first, counts = np.unique(
            sorted_points, return_index=True, return_counts=True
        )
        assignments[unique_points] = sorted_eco_ids[first]
        ambiguous = int((counts > 1).sum())
    else:
        ambiguous = 0
    return assignments, ambiguous


def _install_generated(part: Path, destination: Path) -> None:
    if destination.exists():
        if sha256_file(destination) != sha256_file(part):
            raise ContractError(f"refusing to replace different generated artifact: {destination}")
        part.unlink()
        return
    os.replace(part, destination)
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("w", encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())
    _install_generated(part, path)


def verify_candidate_frame_receipt(
    candidates: str | Path, receipt: str | Path
) -> dict[str, Any]:
    candidate_path = Path(candidates)
    receipt_path = Path(receipt)
    if not receipt_path.is_file():
        raise ContractError(f"candidate-frame receipt not found: {receipt_path}")
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid candidate-frame receipt: {receipt_path}") from error
    if payload.get("schema") != "spectrajam-candidate-frame-v1":
        raise ContractError("candidate-frame receipt has an unexpected schema")
    identity = payload.get("candidate_output")
    if not isinstance(identity, dict):
        raise ContractError("candidate-frame receipt is missing candidate_output")
    if not candidate_path.is_file():
        raise ContractError(f"candidate CSV not found: {candidate_path}")
    if candidate_path.stat().st_size != identity.get("bytes"):
        raise ContractError("candidate CSV byte count does not match its receipt")
    if sha256_file(candidate_path) != identity.get("sha256"):
        raise ContractError("candidate CSV SHA-256 does not match its receipt")
    return payload


def _source_receipts(source_root: Path) -> list[dict[str, object]]:
    return [
        source.provenance(source.destination(source_root))
        for source in FRAME_SOURCES.values()
    ]


def candidate_frame_contract(config: ExperimentConfig) -> dict[str, object]:
    """Return only config fields that can change candidate-frame bytes."""
    return {
        "countries": list(config.countries),
        "lattice_spacing_m": config.sampling.lattice_spacing_m,
        "spatial_block_km": config.sampling.spatial_block_km,
        "extent_policy": {
            country: {
                "source": policy.source,
                "boundary_sha256": policy.boundary_sha256.lower(),
                "ndlsa": policy.ndlsa,
            }
            for country, policy in sorted(config.extents.items())
        },
        "primary_strata": list(config.strata.primary),
        "country_grid_crs": COUNTRY_GRID_CRS,
        "worldcover_excluded_classes": sorted(WORLD_COVER_EXCLUDED),
        "maximum_missing_ecoregion_rate": MAX_MISSING_ECOREGION_RATE,
    }


def _build_candidate_frame(
    config: ExperimentConfig,
    source_root: str | Path,
    output: str | Path,
    receipt: str | Path,
    *,
    chunk_size: int = 50_000,
) -> CandidateFrameResult:
    """Build the pinned 200 m land frame and its deterministic provenance receipt."""
    if chunk_size < 1:
        raise ContractError("candidate-frame chunk size must be positive")
    if tuple(config.countries) != ("RWA", "ISR"):
        raise ContractError("candidate frame requires countries in canonical RWA, ISR order")
    block_size_m = config.sampling.spatial_block_km * 1000
    block_cells = block_size_m / config.sampling.lattice_spacing_m
    if not math.isclose(block_cells, round(block_cells), abs_tol=1e-9):
        raise ContractError("spatial block size must be an integer number of lattice cells")

    source_root = Path(source_root)
    output_path = Path(output)
    receipt_path = Path(receipt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part = output_path.with_suffix(output_path.suffix + ".part")
    verify_frame_sources(source_root)
    admin0_source = FRAME_SOURCES["world_bank_admin0"]
    for country, policy in config.extents.items():
        if (
            policy.source != "world_bank_official_boundaries_v2"
            or policy.boundary_sha256.lower() != admin0_source.artifact.sha256
            or policy.ndlsa != "exclude"
        ):
            raise ContractError(f"{country} extent policy does not match the pinned frame")
    dependencies = _require_optional_dependencies()
    np = dependencies["np"]
    pyproj = dependencies["pyproj"]
    shapely = dependencies["shapely"]
    transform_geometry = dependencies["transform"]

    paths = frame_source_paths(source_root)
    _validate_worldcover_grid(paths["worldcover_2021_grid"])
    boundaries, ndlsa_touches = _load_boundaries(
        config,
        paths["world_bank_admin0"],
        paths["world_bank_ndlsa"],
        dependencies,
    )
    ecoregions, repair_count = _load_ecoregions(
        paths["resolve_ecoregions_2017"], dependencies
    )
    tree = dependencies["STRtree"](ecoregions.geometry.to_numpy())
    eco_names = dict(
        zip(ecoregions["ECO_ID"], ecoregions["ECO_NAME"], strict=True)
    )

    fields = [
        "candidate_id",
        "country",
        "longitude",
        "latitude",
        "spatial_block",
        "stratum",
        "ecoregion_id",
        "ecoregion_name",
        "worldcover_class",
    ]
    counts: Counter[str] = Counter()
    exclusions: dict[str, Counter[str]] = {
        country: Counter() for country in config.countries
    }
    strata: dict[str, Counter[str]] = {
        country: Counter() for country in config.countries
    }
    ambiguous_by_country: Counter[str] = Counter()
    crs_receipts: dict[str, dict[str, object]] = {}

    with ExitStack() as stack, part.open("w", newline="", encoding="utf-8") as stream:
        datasets = _open_worldcover(stack, source_root, dependencies)
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for country in config.countries:
            crs_name = COUNTRY_GRID_CRS[country]
            crs = pyproj.CRS.from_user_input(crs_name)
            epsg = int(crs.to_epsg())
            forward = pyproj.Transformer.from_crs(4326, crs, always_xy=True)
            inverse = pyproj.Transformer.from_crs(crs, 4326, always_xy=True)
            boundary_metric = transform_geometry(forward.transform, boundaries[country])
            x_first, x_last = lattice_index_bounds(
                boundary_metric.bounds[0],
                boundary_metric.bounds[2],
                config.sampling.lattice_spacing_m,
            )
            y_first, y_last = lattice_index_bounds(
                boundary_metric.bounds[1],
                boundary_metric.bounds[3],
                config.sampling.lattice_spacing_m,
            )
            x_count = x_last - x_first + 1
            y_count = y_last - y_first + 1
            envelope_count = x_count * y_count
            crs_receipts[country] = {
                "epsg": epsg,
                "wkt": crs.to_wkt(),
                "lattice_origin_m": [0, 0],
                "cell_center_offset": [0.5, 0.5],
                "x_index_bounds": [x_first, x_last],
                "y_index_bounds": [y_first, y_last],
                "envelope_centers": envelope_count,
            }

            for start in range(0, envelope_count, chunk_size):
                stop = min(start + chunk_size, envelope_count)
                flat = np.arange(start, stop, dtype=np.int64)
                x_indices = x_first + flat % x_count
                y_indices = y_first + flat // x_count
                xs = (x_indices.astype(np.float64) + 0.5) * config.sampling.lattice_spacing_m
                ys = (y_indices.astype(np.float64) + 0.5) * config.sampling.lattice_spacing_m
                points_metric = shapely.points(xs, ys)
                inside = shapely.covers(boundary_metric, points_metric)
                exclusions[country]["outside_boundary"] += int((~inside).sum())
                if not inside.any():
                    continue
                xs = xs[inside]
                ys = ys[inside]
                x_indices = x_indices[inside]
                y_indices = y_indices[inside]
                longitudes, latitudes = inverse.transform(xs, ys)
                longitudes = np.asarray(longitudes, dtype=np.float64)
                latitudes = np.asarray(latitudes, dtype=np.float64)

                land_cover = _sample_worldcover(
                    longitudes, latitudes, datasets, dependencies
                )
                exclusions[country]["worldcover_nodata"] += int((land_cover == 0).sum())
                exclusions[country]["permanent_water"] += int((land_cover == 80).sum())
                keep = ~np.isin(land_cover, tuple(WORLD_COVER_EXCLUDED))
                if not keep.any():
                    continue
                xs = xs[keep]
                ys = ys[keep]
                x_indices = x_indices[keep]
                y_indices = y_indices[keep]
                longitudes = longitudes[keep]
                latitudes = latitudes[keep]
                land_cover = land_cover[keep]

                eco_ids, ambiguous = _assign_ecoregions(
                    longitudes, latitudes, ecoregions, tree, dependencies
                )
                ambiguous_by_country[country] += ambiguous
                has_ecoregion = eco_ids >= 0
                exclusions[country]["missing_ecoregion"] += int(
                    (~has_ecoregion).sum()
                )
                if not has_ecoregion.any():
                    continue
                xs = xs[has_ecoregion]
                ys = ys[has_ecoregion]
                x_indices = x_indices[has_ecoregion]
                y_indices = y_indices[has_ecoregion]
                longitudes = longitudes[has_ecoregion]
                latitudes = latitudes[has_ecoregion]
                land_cover = land_cover[has_ecoregion]
                eco_ids = eco_ids[has_ecoregion]
                for index in range(len(longitudes)):
                    eco_id = int(eco_ids[index])
                    cover = int(land_cover[index])
                    stratum = f"eco:{eco_id:04d}|wc:{cover:03d}"
                    row = {
                        "candidate_id": candidate_id(
                            country, epsg, int(x_indices[index]), int(y_indices[index])
                        ),
                        "country": country,
                        "longitude": f"{float(longitudes[index]):.10f}",
                        "latitude": f"{float(latitudes[index]):.10f}",
                        "spatial_block": spatial_block_id(
                            country,
                            epsg,
                            float(xs[index]),
                            float(ys[index]),
                            block_size_m,
                        ),
                        "stratum": stratum,
                        "ecoregion_id": eco_id,
                        "ecoregion_name": eco_names[eco_id],
                        "worldcover_class": cover,
                    }
                    writer.writerow(row)
                    counts[country] += 1
                    strata[country][stratum] += 1
        stream.flush()
        os.fsync(stream.fileno())

    missing_country = set(counts) != set(config.countries) or any(
        counts[country] == 0 for country in config.countries
    )
    if missing_country:
        raise ContractError(f"candidate frame is missing a country: {dict(counts)}")
    missing_ecoregion_rates = {}
    for country in config.countries:
        missing = exclusions[country]["missing_ecoregion"]
        eligible = counts[country] + missing
        rate = missing / eligible
        if rate > MAX_MISSING_ECOREGION_RATE:
            raise ContractError(
                f"{country} missing-ecoregion rate {rate:.3%} exceeds "
                f"{MAX_MISSING_ECOREGION_RATE:.1%}"
            )
        missing_ecoregion_rates[country] = rate
    _install_generated(part, output_path)
    output_sha256 = sha256_file(output_path)

    frame_contract = candidate_frame_contract(config)
    receipt_payload: dict[str, Any] = {
        "schema": "spectrajam-candidate-frame-v1",
        "implementation": {
            **_IMPLEMENTATION_HASHES,
            "config_contract_sha256": hashlib.sha256(
                json.dumps(
                    frame_contract,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        },
        "frame_contract": frame_contract,
        "countries": list(config.countries),
        "boundary_policy": "world-bank-v2-admin0-with-ndlsa-excluded",
        "sources": _source_receipts(source_root),
        "candidate_output": {
            "path": str(output_path),
            "bytes": output_path.stat().st_size,
            "sha256": output_sha256,
            "columns": fields,
        },
        "lattice": {
            "spacing_m": config.sampling.lattice_spacing_m,
            "spatial_block_km": config.sampling.spatial_block_km,
            "country_crs": crs_receipts,
        },
        "stratification": {
            "key": "eco:{ECO_ID:04d}|wc:{WorldCover class:03d}",
            "resolve_geometry_repairs": repair_count,
            "maximum_missing_ecoregion_rate": MAX_MISSING_ECOREGION_RATE,
            "missing_ecoregion_rates": missing_ecoregion_rates,
            "ecoregion_boundary_ties_resolved_to_lowest_id": dict(ambiguous_by_country),
            "stratum_counts": {
                country: dict(sorted(strata[country].items()))
                for country in config.countries
            },
        },
        "counts": {
            "candidates": {country: counts[country] for country in config.countries},
            "exclusions": {
                country: dict(sorted(exclusions[country].items()))
                for country in config.countries
            },
            "ndlsa_boundary_touches": ndlsa_touches,
        },
        "runtime": {
            "geopandas": dependencies["gpd"].__version__,
            "numpy": dependencies["np"].__version__,
            "pyproj": dependencies["pyproj"].__version__,
            "pyogrio": dependencies["pyogrio"].__version__,
            "pyogrio_gdal": dependencies["pyogrio"].__gdal_version__,
            "rasterio": dependencies["rasterio"].__version__,
            "rasterio_gdal": dependencies["rasterio"].__gdal_version__,
            "shapely": dependencies["shapely"].__version__,
            "geos": dependencies["shapely"].geos_version_string,
            "proj": pyproj.proj_version_str,
        },
    }
    write_json_atomic(receipt_path, receipt_payload)
    return CandidateFrameResult(
        output=output_path,
        receipt=receipt_path,
        output_sha256=output_sha256,
        candidates_by_country={country: counts[country] for country in config.countries},
        exclusions_by_country={
            country: dict(exclusions[country]) for country in config.countries
        },
    )


def build_candidate_frame(
    config: ExperimentConfig,
    source_root: str | Path,
    output: str | Path,
    receipt: str | Path,
    *,
    chunk_size: int = 50_000,
) -> CandidateFrameResult:
    """Build one candidate frame at a time under an exclusive source-root lock."""
    with frame_operation_lock(source_root, "candidate-frame"):
        return _build_candidate_frame(
            config,
            source_root,
            output,
            receipt,
            chunk_size=chunk_size,
        )
