from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .contracts import ContractError, PointYear, sha256_file, stable_sample_id


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    country: str
    longitude: float
    latitude: float
    spatial_block: str
    stratum: str


@dataclass(frozen=True, slots=True)
class SelectedPoint:
    candidate: Candidate
    split: str
    inclusion_probability: float


PROJECTED_LATTICE_DISTANCE_TOLERANCE = 0.005


class _MinimumDistanceGuard:
    def __init__(self, minimum_meters: float, relative_tolerance: float = 0.0):
        if not 0 <= relative_tolerance < 1:
            raise ContractError("minimum-distance tolerance must be in [0, 1)")
        self.minimum_meters = minimum_meters
        self.effective_minimum_meters = minimum_meters * (1 - relative_tolerance)
        self.cell_degrees = minimum_meters / 111_320.0
        self.cells: dict[tuple[str, int, int], list[tuple[float, float]]] = defaultdict(list)

    @staticmethod
    def _distance_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
        lon1, lat1 = map(math.radians, first)
        lon2, lat2 = map(math.radians, second)
        delta_lon = lon2 - lon1
        delta_lat = lat2 - lat1
        value = (
            math.sin(delta_lat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        )
        return 6_371_008.8 * 2 * math.asin(min(1.0, math.sqrt(value)))

    def add(self, candidate: Candidate) -> None:
        x = math.floor(candidate.longitude / self.cell_degrees)
        y = math.floor(candidate.latitude / self.cell_degrees)
        location = (candidate.longitude, candidate.latitude)
        for delta_x in range(-2, 3):
            for delta_y in range(-2, 3):
                for other in self.cells.get(
                    (candidate.country, x + delta_x, y + delta_y), ()
                ):
                    if (
                        self._distance_meters(location, other)
                        < self.effective_minimum_meters
                    ):
                        raise ContractError(
                            f"candidate {candidate.candidate_id} violates the "
                            f"{self.minimum_meters:g} m minimum distance"
                        )
        self.cells[(candidate.country, x, y)].append(location)


def _stable_unit_interval(seed: int, value: str) -> float:
    digest = hashlib.blake2b(f"{seed}:{value}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") / 2**64


def assign_block_split(
    block_id: str,
    ratios: Mapping[str, float],
    seed: int,
) -> str:
    """Assign an entire spatial block to one split using a stable hash."""
    if set(ratios) != {"train", "validation", "test"}:
        raise ContractError("ratios must define train, validation, and test")
    if abs(sum(ratios.values()) - 1.0) > 1e-9:
        raise ContractError("split ratios must sum to one")
    value = _stable_unit_interval(seed, f"block:{block_id}")
    cumulative = 0.0
    for name in ("train", "validation", "test"):
        cumulative += ratios[name]
        if value < cumulative:
            return name
    return "test"


def allocate_strata(
    counts: Mapping[str, int],
    total: int,
    min_per_stratum: int,
    allocation_power: float,
) -> dict[str, int]:
    """Allocate a bounded power allocation with a rare-stratum floor.

    A power of 1 is candidate/area proportional, 0 is equal allocation, and the
    recommended 0.5 produces allocation proportional to ``sqrt(N_h)``.
    """
    available = {key: int(value) for key, value in counts.items() if int(value) > 0}
    if not available:
        raise ContractError("no non-empty strata")
    if total <= 0:
        raise ContractError("sample total must be positive")
    if sum(available.values()) < total:
        raise ContractError(
            f"requested {total} samples but only {sum(available.values())} candidates exist"
        )
    if min_per_stratum < 0 or not 0 <= allocation_power <= 1:
        raise ContractError("invalid stratum allocation parameters")

    allocation = {key: min(min_per_stratum, count) for key, count in available.items()}
    reserved = sum(allocation.values())
    if reserved > total:
        raise ContractError(
            "min_per_stratum reserves more samples than the requested total; "
            "reduce the floor or merge sparse strata"
        )

    remaining = total - reserved
    while remaining:
        capacity = {key: available[key] - allocation[key] for key in available}
        open_keys = [key for key, value in capacity.items() if value > 0]
        if not open_keys:
            raise ContractError("allocation exhausted all strata before reaching the target")

        powered = {key: capacity[key] ** allocation_power for key in open_keys}
        denominator = sum(powered.values())
        weights = {key: powered[key] / denominator for key in open_keys}
        raw = {key: remaining * weights[key] for key in open_keys}
        increments = {
            key: min(capacity[key], int(math.floor(raw[key]))) for key in open_keys
        }
        assigned = sum(increments.values())

        if assigned == 0:
            order = sorted(
                open_keys,
                key=lambda key: (raw[key] - math.floor(raw[key]), weights[key], key),
                reverse=True,
            )
            for key in order[:remaining]:
                increments[key] = 1
            assigned = sum(increments.values())

        for key, increment in increments.items():
            allocation[key] += increment
        remaining -= assigned

    if sum(allocation.values()) != total:
        raise AssertionError("stratum allocation did not conserve the requested sample count")
    return allocation


def select_candidates(
    candidates: Sequence[Candidate],
    points_per_country: int | None,
    min_per_stratum: int,
    allocation_power: float,
    split_ratios: Mapping[str, float],
    seed: int,
    min_distance_m: float = 200.0,
    min_distance_relative_tolerance: float = 0.0,
) -> list[SelectedPoint]:
    """Deterministically select candidates while keeping spatial blocks intact."""
    selected: list[SelectedPoint] = []
    by_country: dict[str, list[Candidate]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_locations: set[tuple[str, float, float]] = set()
    for candidate in candidates:
        if not candidate.candidate_id or candidate.candidate_id in seen_ids:
            raise ContractError(f"duplicate or empty candidate_id: {candidate.candidate_id!r}")
        if candidate.country not in {"RWA", "ISR"}:
            raise ContractError(f"unsupported candidate country: {candidate.country}")
        if not candidate.spatial_block or not candidate.stratum:
            raise ContractError("candidate spatial_block and stratum are required")
        location = (
            candidate.country,
            round(candidate.longitude, 8),
            round(candidate.latitude, 8),
        )
        if location in seen_locations:
            raise ContractError(f"duplicate candidate location: {location}")
        seen_ids.add(candidate.candidate_id)
        seen_locations.add(location)
        by_country[candidate.country].append(candidate)

    if set(by_country) != {"RWA", "ISR"}:
        raise ContractError(
            f"candidate countries {sorted(by_country)} do not match ['ISR', 'RWA']"
        )

    for country in sorted(by_country):
        country_candidates = by_country[country]
        if points_per_country is None:
            for candidate in country_candidates:
                selected.append(
                    SelectedPoint(
                        candidate=candidate,
                        split=assign_block_split(candidate.spatial_block, split_ratios, seed),
                        inclusion_probability=1.0,
                    )
                )
            continue
        counts = Counter(candidate.stratum for candidate in country_candidates)
        allocation = allocate_strata(
            counts,
            points_per_country,
            min_per_stratum,
            allocation_power,
        )
        by_stratum: dict[str, list[Candidate]] = defaultdict(list)
        for candidate in country_candidates:
            by_stratum[candidate.stratum].append(candidate)

        for stratum, target in sorted(allocation.items()):
            ranked = sorted(
                by_stratum[stratum],
                key=lambda item: (
                    _stable_unit_interval(seed, f"candidate:{item.candidate_id}"),
                    item.candidate_id,
                ),
            )
            for candidate in ranked[:target]:
                split = assign_block_split(candidate.spatial_block, split_ratios, seed)
                selected.append(
                    SelectedPoint(
                        candidate=candidate,
                        split=split,
                        inclusion_probability=target / len(by_stratum[stratum]),
                    )
                )

    assert_no_block_leakage(selected)
    result = sorted(
        selected,
        key=lambda item: (
            item.candidate.country,
            item.split,
            item.candidate.stratum,
            item.candidate.candidate_id,
        ),
    )
    distance_guard = _MinimumDistanceGuard(
        min_distance_m, min_distance_relative_tolerance
    )
    for point in result:
        distance_guard.add(point.candidate)
    return result


def assert_no_block_leakage(points: Iterable[SelectedPoint]) -> None:
    split_by_block: dict[str, str] = {}
    for point in points:
        block = point.candidate.spatial_block
        previous = split_by_block.setdefault(block, point.split)
        if previous != point.split:
            raise ContractError(
                f"spatial block {block} occurs in both {previous} and {point.split}"
            )


def stream_full_lattice(
    candidates: Iterable[Candidate],
    split_ratios: Mapping[str, float],
    seed: int,
    min_distance_m: float = 200.0,
    min_distance_relative_tolerance: float = 0.0,
) -> Iterator[SelectedPoint]:
    """Stream a complete lattice without retaining millions of point objects."""
    seen_ids: set[str] = set()
    seen_locations: set[tuple[str, float, float]] = set()
    seen_countries: set[str] = set()
    distance_guard = _MinimumDistanceGuard(
        min_distance_m, min_distance_relative_tolerance
    )
    for candidate in candidates:
        if not candidate.candidate_id or candidate.candidate_id in seen_ids:
            raise ContractError(f"duplicate or empty candidate_id: {candidate.candidate_id!r}")
        location = (
            candidate.country,
            round(candidate.longitude, 8),
            round(candidate.latitude, 8),
        )
        if location in seen_locations:
            raise ContractError(f"duplicate candidate location: {location}")
        if candidate.country not in {"RWA", "ISR"}:
            raise ContractError(f"unsupported candidate country: {candidate.country}")
        if not candidate.spatial_block or not candidate.stratum:
            raise ContractError("candidate spatial_block and stratum are required")
        seen_ids.add(candidate.candidate_id)
        seen_locations.add(location)
        seen_countries.add(candidate.country)
        distance_guard.add(candidate)
        yield SelectedPoint(
            candidate=candidate,
            split=assign_block_split(candidate.spatial_block, split_ratios, seed),
            inclusion_probability=1.0,
        )
    if seen_countries != {"RWA", "ISR"}:
        raise ContractError(
            f"candidate countries {sorted(seen_countries)} do not match ['ISR', 'RWA']"
        )


def expand_years(
    points: Iterable[SelectedPoint],
    years_by_split: Mapping[str, Sequence[int]],
) -> Iterator[PointYear]:
    """Build the full spatial-split × year-split evaluation matrix."""
    for point in points:
        candidate = point.candidate
        for year_split in ("train", "validation", "test"):
            for year in years_by_split[year_split]:
                yield PointYear(
                    sample_id=stable_sample_id(
                        candidate.country, candidate.candidate_id, year
                    ),
                    candidate_id=candidate.candidate_id,
                    country=candidate.country,
                    longitude=candidate.longitude,
                    latitude=candidate.latitude,
                    spatial_block=candidate.spatial_block,
                    stratum=candidate.stratum,
                    inclusion_probability=point.inclusion_probability,
                    spatial_split=point.split,
                    year_split=year_split,
                    year=int(year),
                )


def read_candidates(path: str | Path) -> Iterator[Candidate]:
    required = {
        "candidate_id",
        "country",
        "longitude",
        "latitude",
        "spatial_block",
        "stratum",
    }
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ContractError(f"candidate CSV is missing columns: {sorted(missing)}")
        for row in reader:
            yield Candidate(
                candidate_id=row["candidate_id"],
                country=row["country"],
                longitude=float(row["longitude"]),
                latitude=float(row["latitude"]),
                spatial_block=row["spatial_block"],
                stratum=row["stratum"],
            )


def validate_candidate_extents(
    candidates: Iterable[Candidate], boundary_paths: Mapping[str, str]
) -> Iterator[Candidate]:
    """Reject candidate points outside the checksum-verified country extents.

    World Bank ADM0 releases contain every country in one file.  Select the
    configured ISO feature before preparing the geometry; unioning the full
    release would turn this check into a nearly global extent check.
    """
    try:
        import geopandas as gpd
        from pyproj import Transformer
        from shapely.geometry import Point
        from shapely.prepared import prep
    except ImportError as error:
        raise RuntimeError("install spectrajam[data] to validate candidate boundaries") from error

    from .candidate_frame import COUNTRY_GRID_CRS

    boundaries = {}
    loaded_frames = {}
    for country, path in boundary_paths.items():
        frame = loaded_frames.get(path)
        if frame is None:
            frame = gpd.read_file(path, engine="pyogrio")
            loaded_frames[path] = frame
        if frame.empty or frame.crs is None:
            raise ContractError(f"boundary {path} is empty or has no CRS")
        if "ISO_A3" not in frame.columns:
            raise ContractError(f"boundary {path} has no ISO_A3 field")
        selected = frame.loc[frame["ISO_A3"] == country]
        if len(selected) != 1:
            raise ContractError(
                f"boundary {path} contains {len(selected)} features for ISO_A3={country}; "
                "expected exactly one"
            )
        metric_crs = COUNTRY_GRID_CRS[country]
        geometry = selected.to_crs(metric_crs).geometry.iloc[0]
        if geometry.is_empty or not geometry.is_valid:
            raise ContractError(f"boundary {path} has invalid geometry")
        boundaries[country] = (
            prep(geometry),
            Transformer.from_crs("EPSG:4326", metric_crs, always_xy=True),
        )

    for candidate in candidates:
        boundary_contract = boundaries.get(candidate.country)
        if boundary_contract is None:
            raise ContractError(f"no verified boundary for {candidate.country}")
        boundary, transformer = boundary_contract
        x, y = transformer.transform(candidate.longitude, candidate.latitude)
        if not boundary.covers(Point(x, y)):
            raise ContractError(
                f"candidate {candidate.candidate_id} lies outside the pinned "
                f"{candidate.country} extent"
            )
        yield candidate


def write_manifest(path: str | Path, records: Iterable[PointYear]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    fields = [
        "sample_id",
        "candidate_id",
        "country",
        "longitude",
        "latitude",
        "spatial_block",
        "stratum",
        "inclusion_probability",
        "spatial_split",
        "year_split",
        "year",
    ]
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        count = 0
        for record in records:
            writer.writerow({field: getattr(record, field) for field in fields})
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    if destination.exists():
        if sha256_file(destination) != sha256_file(temporary):
            raise ContractError(f"refusing to replace different manifest: {destination}")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return count


def read_manifest(path: str | Path) -> Iterator[PointYear]:
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        for row in reader:
            record = PointYear(
                sample_id=row["sample_id"],
                candidate_id=row["candidate_id"],
                country=row["country"],
                longitude=float(row["longitude"]),
                latitude=float(row["latitude"]),
                spatial_block=row["spatial_block"],
                stratum=row["stratum"],
                inclusion_probability=float(row["inclusion_probability"]),
                spatial_split=row["spatial_split"],
                year_split=row["year_split"],
                year=int(row["year"]),
            )
            expected = stable_sample_id(record.country, record.candidate_id, record.year)
            if record.sample_id != expected:
                raise ContractError(
                    f"sample_id mismatch for {record.candidate_id}/{record.year}: "
                    f"expected {expected}, got {record.sample_id}"
                )
            yield record


def verify_sampling_receipt(
    manifest: str | Path,
    config: str | Path,
    receipt: str | Path,
) -> dict[str, object]:
    manifest_path = Path(manifest)
    receipt_path = Path(receipt)
    if not receipt_path.is_file():
        raise ContractError(f"sampling receipt not found: {receipt_path}")
    try:
        payload = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid sampling receipt: {receipt_path}") from error
    if payload.get("schema") != "spectrajam-sampling-v1":
        raise ContractError("sampling receipt has an unexpected schema")
    identity = payload.get("manifest")
    if not isinstance(identity, dict):
        raise ContractError("sampling receipt is missing manifest identity")
    if not manifest_path.is_file():
        raise ContractError(f"manifest not found: {manifest_path}")
    if manifest_path.stat().st_size != identity.get("bytes"):
        raise ContractError("manifest byte count does not match its sampling receipt")
    if sha256_file(manifest_path) != identity.get("sha256"):
        raise ContractError("manifest SHA-256 does not match its sampling receipt")
    if sha256_file(config) != payload.get("config_sha256"):
        raise ContractError("config SHA-256 does not match the sampling receipt")
    return payload


def validate_manifest_universe(
    records: Iterable[PointYear],
    countries: Sequence[str],
    years_by_split: Mapping[str, Sequence[int]],
    expected_points_per_country: int | None,
) -> Iterator[PointYear]:
    """Stream-validate that every candidate has the complete configured year matrix."""
    year_role = {
        int(year): split for split, years in years_by_split.items() for year in years
    }
    expected_years = set(year_role)
    expected_countries = set(countries)
    point_counts = Counter()
    current_key: tuple[str, str] | None = None
    current_years: set[int] = set()
    current_signature: tuple[object, ...] | None = None

    def finish_group() -> None:
        if current_key is None:
            return
        if current_years != expected_years:
            raise ContractError(
                f"candidate {current_key} has years {sorted(current_years)}, "
                f"expected {sorted(expected_years)}"
            )
        point_counts[current_key[0]] += 1

    for record in records:
        if record.country not in expected_countries:
            raise ContractError(f"manifest contains unconfigured country {record.country}")
        if year_role.get(record.year) != record.year_split:
            raise ContractError(
                f"year {record.year} is assigned to {record.year_split}, "
                f"expected {year_role.get(record.year)}"
            )
        key = (record.country, record.candidate_id)
        signature = (
            record.longitude,
            record.latitude,
            record.spatial_block,
            record.stratum,
            record.inclusion_probability,
            record.spatial_split,
        )
        if key != current_key:
            finish_group()
            current_key = key
            current_years = set()
            current_signature = signature
        elif signature != current_signature:
            raise ContractError(f"candidate metadata changes across years: {key}")
        if record.year in current_years:
            raise ContractError(f"candidate {key} repeats year {record.year}")
        current_years.add(record.year)
        yield record
    finish_group()

    if set(point_counts) != expected_countries:
        raise ContractError(
            f"manifest countries {sorted(point_counts)} do not match {sorted(expected_countries)}"
        )
    if expected_points_per_country is not None:
        wrong = {
            country: count
            for country, count in point_counts.items()
            if count != expected_points_per_country
        }
        if wrong:
            raise ContractError(
                f"manifest point counts {wrong} do not match {expected_points_per_country}"
            )
