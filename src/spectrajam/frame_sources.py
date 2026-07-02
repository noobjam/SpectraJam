from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any

from .artifacts import PinnedArtifact, fetch_verified_artifact
from .contracts import ContractError, require_sha256

CC_BY_4_0 = "CC-BY-4.0"


@dataclass(frozen=True, slots=True)
class FrameSource:
    key: str
    artifact: PinnedArtifact
    relative_destination: str
    producer: str
    product: str
    version: str
    license: str
    role: str
    metadata: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        allowed = "abcdefghijklmnopqrstuvwxyz0123456789_"
        if not self.key or any(value not in allowed for value in self.key):
            raise ValueError("frame-source key must use lowercase letters, digits, and underscores")
        relative = PurePosixPath(self.relative_destination)
        if relative.is_absolute() or not relative.parts or ".." in relative.parts:
            raise ValueError("frame-source destination must be a safe relative path")
        metadata_keys = [key for key, _ in self.metadata]
        if len(metadata_keys) != len(set(metadata_keys)):
            raise ValueError("frame-source metadata keys must be unique")

    def destination(self, root: str | Path) -> Path:
        return Path(root).joinpath(*PurePosixPath(self.relative_destination).parts)

    def provenance(self, destination: str | Path) -> dict[str, object]:
        return {
            "key": self.key,
            "producer": self.producer,
            "product": self.product,
            "version": self.version,
            "license": self.license,
            "role": self.role,
            "url": self.artifact.url,
            "revision": self.artifact.revision,
            "filename": self.artifact.filename,
            "relative_destination": self.relative_destination,
            "destination": str(Path(destination)),
            "expected_bytes": self.artifact.expected_bytes,
            "sha256": self.artifact.sha256.lower(),
            "checksum_provenance": "independently-observed-2026-07-02",
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FetchedFrameSource:
    source: FrameSource
    path: Path
    reused: bool

    def provenance(self) -> dict[str, object]:
        receipt = self.source.provenance(self.path)
        receipt["reused"] = self.reused
        receipt["verified"] = True
        return receipt


_WORLD_BANK_CATALOG = (
    "https://datacatalog.worldbank.org/infrastructure-data/search/dataset/"
    "0038272/world-bank-official-boundaries"
)
_WORLD_BANK_GPKG_BASE = (
    "https://datacatalogfiles.worldbank.org/ddh-published/0038272/2/DR0095370/"
    "World%20Bank%20Official%20Boundaries%20%28GeoPackage%29/"
)
_WORLD_BANK_TERMS = "https://www.worldbank.org/en/about/legal/terms-of-use-for-datasets"
_WORLD_BANK_ATTRIBUTION = (
    "The World Bank: World Bank Official Boundaries: "
    "World Bank Data Catalog dataset 0038272"
)
_WORLDCOVER_BASE = (
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
)
_WORLDCOVER_DOI = "https://doi.org/10.5281/zenodo.7254221"


def _worldcover_tile(
    tile: str, expected_bytes: int, sha256: str
) -> FrameSource:
    filename = f"ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
    return FrameSource(
        key=f"worldcover_2021_{tile.lower()}",
        artifact=PinnedArtifact(
            filename=filename,
            url=f"{_WORLDCOVER_BASE}{filename}",
            revision="2021-v200",
            expected_bytes=expected_bytes,
            sha256=sha256,
        ),
        relative_destination=f"worldcover/2021-v200/map/{filename}",
        producer="ESA WorldCover Consortium",
        product="ESA WorldCover 10 m 2021",
        version="v200 (2.0.0)",
        license=CC_BY_4_0,
        role="land-cover-map",
        metadata=(
            ("tile", tile),
            ("crs", "EPSG:4326"),
            ("data_type", "uint8"),
            ("nodata", 0),
            ("class_codes", (10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)),
            ("citation", _WORLDCOVER_DOI),
        ),
    )


_SOURCES = (
    FrameSource(
        key="world_bank_admin0",
        artifact=PinnedArtifact(
            filename="World Bank Official Boundaries - Admin 0.gpkg",
            url=(
                f"{_WORLD_BANK_GPKG_BASE}"
                "World%20Bank%20Official%20Boundaries%20-%20Admin%200.gpkg"
                "?versionid=2026-06-12T17%3A13%3A04.3916935Z"
            ),
            revision="v2/DR0095370/2026-06-12",
            expected_bytes=63_422_464,
            sha256="97f0c8a0fa848b9a8414dbeb2e058fa37d59b13794ec232a87da000bdf4b117e",
        ),
        relative_destination="boundaries/world-bank-v2/admin0.gpkg",
        producer="World Bank",
        product="World Bank Official Boundaries",
        version="2",
        license=CC_BY_4_0,
        role="admin-0-boundary",
        metadata=(
            ("layer", "WB_GAD_ADM0"),
            ("catalog", _WORLD_BANK_CATALOG),
            ("dataset_id", "0038272"),
            ("release_id", "DR0095370"),
            ("dataset_terms", _WORLD_BANK_TERMS),
            ("attribution", _WORLD_BANK_ATTRIBUTION),
        ),
    ),
    FrameSource(
        key="world_bank_ndlsa",
        artifact=PinnedArtifact(
            filename="World Bank Official Boundaries - NDLSA.gpkg",
            url=(
                f"{_WORLD_BANK_GPKG_BASE}"
                "World%20Bank%20Official%20Boundaries%20-%20NDLSA.gpkg"
                "?versionid=2026-06-12T17%3A13%3A17.3238298Z"
            ),
            revision="v2/DR0095370/2026-06-12",
            expected_bytes=684_032,
            sha256="159ef2d133d12491eb6ce2f0d0d1032083209b0cf7d28ddda774a503055d2fa4",
        ),
        relative_destination="boundaries/world-bank-v2/ndlsa.gpkg",
        producer="World Bank",
        product="World Bank Official Boundaries",
        version="2",
        license=CC_BY_4_0,
        role="non-determined-legal-status-areas",
        metadata=(
            ("layer", "WB_GAD_ADM0_NDLSA"),
            ("policy", "exclude"),
            ("catalog", _WORLD_BANK_CATALOG),
            ("dataset_id", "0038272"),
            ("release_id", "DR0095370"),
            ("dataset_terms", _WORLD_BANK_TERMS),
            ("attribution", _WORLD_BANK_ATTRIBUTION),
        ),
    ),
    FrameSource(
        key="world_bank_data_dictionary",
        artifact=PinnedArtifact(
            filename="DataDictionary.xlsx",
            url=(
                "https://datacatalogfiles.worldbank.org/ddh-published/0038272/"
                "DR0095372/DataDictionary.xlsx"
                "?versionid=2026-06-12T17%3A15%3A01.2525134Z"
            ),
            revision="v2/DR0095372/2026-06-12",
            expected_bytes=16_467,
            sha256="012aefc5f5b8a4b36cf8c43e18a233f8e4bb44b47f4c4f8ad43bb9472102541c",
        ),
        relative_destination="boundaries/world-bank-v2/DataDictionary.xlsx",
        producer="World Bank",
        product="World Bank Official Boundaries Data Dictionary",
        version="2",
        license=CC_BY_4_0,
        role="boundary-data-dictionary",
        metadata=(
            ("catalog", _WORLD_BANK_CATALOG),
            ("dataset_id", "0038272"),
            ("release_id", "DR0095372"),
            ("dataset_terms", _WORLD_BANK_TERMS),
            ("attribution", _WORLD_BANK_ATTRIBUTION),
        ),
    ),
    FrameSource(
        key="resolve_ecoregions_2017",
        artifact=PinnedArtifact(
            filename="Ecoregions2017.zip",
            url="https://storage.googleapis.com/teow2016/Ecoregions2017.zip",
            revision="2017/gcs-generation-1557190379773656",
            expected_bytes=149_248_653,
            sha256="be36d6209e443038d02e309f0447c6e7f2a62f5fe60c605ffe90d064952f2a60",
        ),
        relative_destination="ecoregions/2017/Ecoregions2017.zip",
        producer="RESOLVE Biodiversity and Wildlife Solutions",
        product="RESOLVE Ecoregions 2017",
        version="2017",
        license=CC_BY_4_0,
        role="ecoregion-polygons",
        metadata=(
            ("crs", "EPSG:4326"),
            ("primary_field", "ECO_ID"),
            ("dbf_encoding", "ISO-8859-1"),
            ("expected_feature_count", 847),
            ("citation", "https://doi.org/10.1093/biosci/bix014"),
            ("catalog", "https://ecoregions.appspot.com/"),
        ),
    ),
    FrameSource(
        key="worldcover_2021_grid",
        artifact=PinnedArtifact(
            filename="esa_worldcover_grid.geojson",
            url=(
                "https://esa-worldcover.s3.eu-central-1.amazonaws.com/"
                "esa_worldcover_grid.geojson"
            ),
            revision="grid/2023-03-06",
            expected_bytes=543_674,
            sha256="eeb5074bf182c411b3872b2494f6514401ecd9ba8ba0c353fe282f1e2b822f5b",
        ),
        relative_destination="worldcover/2021-v200/esa_worldcover_grid.geojson",
        producer="ESA WorldCover Consortium",
        product="ESA WorldCover Product Grid",
        version="2021-v200",
        license=CC_BY_4_0,
        role="worldcover-tile-index",
        metadata=(
            ("crs", "EPSG:4326"),
            ("tile_field", "ll_tile"),
            ("expected_feature_count", 2651),
            ("citation", _WORLDCOVER_DOI),
        ),
    ),
    _worldcover_tile(
        "S03E027",
        77_269_683,
        "79e03e9e79c7ab0b65f7c1767f6a6e0abdfc5973690eff8d60267b6687478b36",
    ),
    _worldcover_tile(
        "S03E030",
        124_518_426,
        "69734c1e0cb6298f0277136d4233042fad30c3d76a03f2c525c6dfad9a53e69d",
    ),
    _worldcover_tile(
        "N27E033",
        8_648_221,
        "e1175f97da47b5ac6b12ea27bb153b42a46aa08d16e5cbd26568dcd40d4ae171",
    ),
    _worldcover_tile(
        "N30E033",
        50_429_555,
        "ebf8b8bde6240429c9246abc936f17a5219699f8315057410480f7ea89b59bae",
    ),
    _worldcover_tile(
        "N33E033",
        31_235_439,
        "61c0d2597683862ceb9ecd472c37d74d02a5563e37a696a3f32156d86efd6ecd",
    ),
)

FRAME_SOURCES: Mapping[str, FrameSource] = MappingProxyType(
    {source.key: source for source in _SOURCES}
)

if len(FRAME_SOURCES) != len(_SOURCES):  # pragma: no cover - import-time invariant
    raise AssertionError("frame-source keys must be unique")
if len({source.relative_destination for source in _SOURCES}) != len(_SOURCES):
    raise AssertionError("frame-source destinations must be unique")


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ContractError(f"another frame operation holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


@contextmanager
def frame_operation_lock(root: str | Path, operation: str):
    if not operation or any(value not in "abcdefghijklmnopqrstuvwxyz-" for value in operation):
        raise ValueError("frame operation must use lowercase letters and hyphens")
    with _exclusive_lock(Path(root) / f".{operation}.lock"):
        yield


def select_frame_sources(keys: Iterable[str] | str | None = None) -> tuple[FrameSource, ...]:
    if keys is None:
        return _SOURCES
    requested = {keys} if isinstance(keys, str) else set(keys)
    if not requested:
        raise ContractError("at least one frame-source key is required")
    unknown = sorted(requested - set(FRAME_SOURCES))
    if unknown:
        raise ContractError(f"unknown frame-source keys: {unknown}")
    return tuple(source for source in _SOURCES if source.key in requested)


def fetch_frame_sources(
    root: str | Path,
    keys: Iterable[str] | str | None = None,
    *,
    fetcher: Callable[..., Path] = fetch_verified_artifact,
    verifier: Callable[[Path, PinnedArtifact], None] | None = None,
    **fetch_options: Any,
) -> tuple[FetchedFrameSource, ...]:
    """Fetch selected sources in registry order into deterministic destinations."""
    source_root = Path(root)
    with _exclusive_lock(source_root / ".fetch.lock"):
        fetched = []
        for source in select_frame_sources(keys):
            destination = source.destination(source_root)
            reused = destination.is_file()
            result = Path(fetcher(source.artifact, destination, **fetch_options))
            if result != destination:
                raise ContractError(
                    f"artifact fetcher returned {result}, expected {destination}"
                )
            (verifier or _verify_frame_source)(result, source.artifact)
            fetched.append(FetchedFrameSource(source, destination, reused))
        return tuple(fetched)


def _verify_frame_source(path: Path, artifact: PinnedArtifact) -> None:
    if not path.is_file():
        raise ContractError(f"frame source not found: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != artifact.expected_bytes:
        raise ContractError(
            f"frame source byte count {actual_bytes} != {artifact.expected_bytes}: {path}"
        )
    require_sha256(path, artifact.sha256)


def verify_frame_sources(root: str | Path) -> tuple[Path, ...]:
    """Verify every pinned frame input without performing network access."""
    verified = []
    for source in _SOURCES:
        path = source.destination(root)
        _verify_frame_source(path, source.artifact)
        verified.append(path)
    return tuple(verified)


def write_frame_sources_receipt(
    path: str | Path, fetched: Iterable[FetchedFrameSource]
) -> Path:
    """Write a deterministic lock receipt without transient reuse state."""
    destination = Path(path)
    records = tuple(fetched)
    payload = {
        "schema": "spectrajam-frame-sources-v1",
        "sources": [record.source.provenance(record.path) for record in records],
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    with _exclusive_lock(destination.parent / ".receipt.lock"):
        if destination.exists():
            if destination.read_bytes() != encoded:
                raise ContractError(
                    f"refusing to replace different frame-source receipt: {destination}"
                )
            return destination

        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        with part.open("wb") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(part, destination)
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return destination
