from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from .contracts import CANONICAL_S1_BANDS, CANONICAL_S2_BANDS, ContractError, PointYear


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    name: str
    endpoint: str
    collections: dict[str, str]
    checkpoint_source: str


MPC_V11 = ProviderProfile(
    name="mpc-v1.1",
    endpoint="https://planetarycomputer.microsoft.com/api/stac/v1",
    collections={"s2": "sentinel-2-l2a", "s1": "sentinel-1-rtc"},
    checkpoint_source="mpc",
)


@dataclass(frozen=True, slots=True)
class WorkTile:
    country: str
    spatial_block: str
    year: int
    bbox: tuple[float, float, float, float]
    sample_count: int

    @property
    def key(self) -> str:
        identity = json.dumps(
            asdict(self), sort_keys=True, separators=(",", ":")
        ).encode()
        digest = hashlib.sha256(identity).hexdigest()[:16]
        return f"{self.country.lower()}-{self.year}-{digest}"


def build_work_tiles(
    records: Iterable[PointYear], padding_degrees: float = 0.002
) -> list[WorkTile]:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for record in records:
        key = (record.country, record.spatial_block, record.year)
        if key not in groups:
            groups[key] = [
                record.longitude,
                record.latitude,
                record.longitude,
                record.latitude,
                1,
            ]
        else:
            aggregate = groups[key]
            aggregate[0] = min(aggregate[0], record.longitude)
            aggregate[1] = min(aggregate[1], record.latitude)
            aggregate[2] = max(aggregate[2], record.longitude)
            aggregate[3] = max(aggregate[3], record.latitude)
            aggregate[4] += 1
    tiles = []
    for (country, block, year), aggregate in sorted(groups.items()):
        min_lon, min_lat, max_lon, max_lat, count = aggregate
        tiles.append(
            WorkTile(
                country=country,
                spatial_block=block,
                year=year,
                bbox=(
                    min_lon - padding_degrees,
                    min_lat - padding_degrees,
                    max_lon + padding_degrees,
                    max_lat + padding_degrees,
                ),
                sample_count=int(count),
            )
        )
    return tiles


class STACCatalog:
    """Paginated work-tile STAC discovery with immutable raw snapshots.

    Discovery never applies an item-level cloud threshold. Cloud validity is a
    per-pixel preprocessing decision and is evaluated after materialization.
    """

    def __init__(
        self,
        profile: ProviderProfile = MPC_V11,
        request_retries: int = 2,
        backoff_factor: float = 0.5,
        backoff_max: float = 60.0,
    ):
        self.profile = profile
        self.request_retries = request_retries
        self.backoff_factor = backoff_factor
        self.backoff_max = backoff_max
        self._client = None

    def _open(self):
        if self._client is None:
            try:
                from pystac_client import Client
                from pystac_client.stac_api_io import StacApiIO
                from urllib3.util.retry import Retry
            except ImportError as error:
                raise RuntimeError("install spectrajam[data] to query STAC") from error
            retry = Retry(
                total=self.request_retries,
                connect=self.request_retries,
                read=self.request_retries,
                status=self.request_retries,
                backoff_factor=self.backoff_factor,
                backoff_max=self.backoff_max,
                status_forcelist=(408, 425, 429, 500, 502, 503, 504),
                allowed_methods=frozenset({"GET", "POST"}),
                respect_retry_after_header=True,
            )
            self._client = Client.open(
                self.profile.endpoint,
                stac_io=StacApiIO(max_retries=retry),
            )
        return self._client

    def search(self, tile: WorkTile, modality: str) -> dict[str, object]:
        if modality not in self.profile.collections:
            raise ContractError(f"profile {self.profile.name} has no {modality} collection")
        search = self._open().search(
            collections=[self.profile.collections[modality]],
            bbox=tile.bbox,
            datetime=f"{tile.year}-01-01T00:00:00Z/{tile.year}-12-31T23:59:59Z",
            max_items=None,
        )
        # pystac-client follows every rel=next page when items() is consumed.
        items = list(search.items())
        if not items:
            raise FileNotFoundError(
                f"no {modality} items for {tile.country}/{tile.spatial_block}/{tile.year}"
            )

        required_assets = (
            (*CANONICAL_S2_BANDS, "SCL")
            if modality == "s2"
            else tuple(value.lower() for value in CANONICAL_S1_BANDS)
        )
        snapshots = []
        rejected = []
        for item in sorted(items, key=lambda value: (str(value.datetime), value.id)):
            missing = set(required_assets) - set(item.assets)
            if missing:
                rejected.append({"id": item.id, "missing_assets": sorted(missing)})
                continue
            if modality == "s1":
                orbit = str(item.properties.get("sat:orbit_state", "")).lower()
                if orbit not in {"ascending", "descending"}:
                    rejected.append(
                        {
                            "id": item.id,
                            "reason": "unsupported_orbit_state",
                            "sat:orbit_state": orbit or None,
                        }
                    )
                    continue
            snapshots.append(
                {
                    "id": item.id,
                    "collection": item.collection_id,
                    "datetime": item.datetime.isoformat() if item.datetime else None,
                    "bbox": item.bbox,
                    "properties": {
                        key: item.properties.get(key)
                        for key in (
                            "s2:mgrs_tile",
                            "sat:orbit_state",
                            "sat:relative_orbit",
                            "processing:version",
                            "created",
                            "updated",
                        )
                        if key in item.properties
                    },
                    # Persist canonical unsigned hrefs. Sign MPC assets only at read time.
                    "assets": {key: item.assets[key].href for key in required_assets},
                    "raw_item": item.to_dict(),
                }
            )
        if not snapshots:
            raise ContractError(
                f"all {len(items)} STAC items were rejected for {modality}"
            )
        return {
            "schema_version": 1,
            "provider": asdict(self.profile),
            "work_tile": asdict(tile),
            "modality": modality,
            "band_order": (
                list(CANONICAL_S2_BANDS)
                if modality == "s2"
                else list(CANONICAL_S1_BANDS)
            ),
            "query": {
                "bbox": tile.bbox,
                "datetime": f"{tile.year}-01-01/{tile.year}-12-31",
                "collection": self.profile.collections[modality],
                "item_cloud_filter": None,
            },
            "items": snapshots,
            "rejected_items": rejected,
        }


def write_catalog_snapshot(path: str | Path, payload: dict[str, object]) -> tuple[str, int]:
    destination = Path(path)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    _atomic_write(destination, encoded)
    return hashlib.sha256(encoded).hexdigest(), len(payload["items"])


@dataclass(frozen=True, slots=True)
class CatalogSnapshotResult:
    path: str
    sha256: str
    item_count: int
    item_documents_written: int
    reused: bool


def _atomic_write(destination: Path, content: bytes) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f"{destination.name}.part.{os.getpid()}.{uuid.uuid4().hex}"
    )
    try:
        with temporary.open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
            installed = True
        except FileExistsError as error:
            if destination.read_bytes() != content:
                raise ContractError(
                    f"refusing to replace immutable artifact: {destination}"
                ) from error
            installed = False
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return installed
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.sha256")


def write_content_addressed_catalog_snapshot(
    path: str | Path,
    item_store: str | Path,
    payload: dict[str, object],
) -> CatalogSnapshotResult:
    """Persist one immutable query while de-duplicating raw STAC item documents."""
    destination = Path(path)
    expected = json.loads(json.dumps(payload))
    items = expected.get("items")
    if not isinstance(items, list) or not items:
        raise ContractError("catalog snapshot must contain at least one item")

    store = Path(item_store)
    written = 0
    for item in items:
        if not isinstance(item, dict):
            raise ContractError("catalog item snapshot must be a mapping")
        raw_item = item.pop("raw_item", None)
        if not isinstance(raw_item, dict):
            raise ContractError("catalog item is missing its raw STAC document")
        encoded_item = _canonical_json(raw_item)
        digest = hashlib.sha256(encoded_item).hexdigest()
        item_path = store / f"{digest}.json"
        if item_path.exists():
            if hashlib.sha256(item_path.read_bytes()).hexdigest() != digest:
                raise ContractError(f"content-addressed STAC item is corrupt: {item_path}")
        else:
            written += int(_atomic_write(item_path, encoded_item))
        item["raw_item_sha256"] = digest

    encoded_query = _canonical_json(expected)
    digest = hashlib.sha256(encoded_query).hexdigest()
    if destination.exists():
        existing = destination.read_bytes()
        if existing != encoded_query:
            raise ContractError(
                f"immutable catalog query already exists with different content: {destination}"
            )
        reused = True
    else:
        reused = not _atomic_write(destination, encoded_query)
    _atomic_write(_digest_path(destination), f"{digest}\n".encode())
    return CatalogSnapshotResult(
        path=str(destination),
        sha256=digest,
        item_count=len(items),
        item_documents_written=written,
        reused=reused,
    )


def catalog_snapshot_path(root: str | Path, tile: WorkTile, modality: str) -> Path:
    if modality not in {"s1", "s2"}:
        raise ContractError(f"unsupported modality: {modality}")
    return (
        Path(root)
        / "queries"
        / tile.country.lower()
        / str(tile.year)
        / tile.key
        / f"{modality}.json"
    )


def _validate_persisted_catalog(
    path: Path,
    item_store: Path,
    tile: WorkTile,
    modality: str,
    profile: ProviderProfile,
) -> tuple[bytes, dict[str, object]]:
    encoded = path.read_bytes()
    receipt_path = _digest_path(path)
    if not receipt_path.is_file():
        raise ContractError(f"catalog query checksum receipt is missing: {receipt_path}")
    expected_digest = receipt_path.read_text().strip().lower()
    actual_digest = hashlib.sha256(encoded).hexdigest()
    if (
        len(expected_digest) != 64
        or any(value not in "0123456789abcdef" for value in expected_digest)
        or actual_digest != expected_digest
    ):
        raise ContractError(f"catalog query checksum mismatch: {path}")
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ContractError(f"catalog query is not valid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ContractError(f"catalog query must be a mapping: {path}")
    work_tile = payload.get("work_tile")
    provider = payload.get("provider")
    query = payload.get("query")
    items = payload.get("items")
    expected_tile = json.loads(json.dumps(asdict(tile)))
    expected_provider = json.loads(json.dumps(asdict(profile)))
    expected_bands = (
        list(CANONICAL_S2_BANDS) if modality == "s2" else list(CANONICAL_S1_BANDS)
    )
    expected_query = {
        "bbox": list(tile.bbox),
        "datetime": f"{tile.year}-01-01/{tile.year}-12-31",
        "collection": profile.collections[modality],
        "item_cloud_filter": None,
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("modality") != modality
        or work_tile != expected_tile
        or provider != expected_provider
        or payload.get("band_order") != expected_bands
        or query != expected_query
        or not isinstance(items, list)
        or not items
    ):
        raise ContractError(f"catalog query identity or schema mismatch: {path}")
    for item in items:
        digest = item.get("raw_item_sha256") if isinstance(item, dict) else None
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(value not in "0123456789abcdef" for value in digest.lower())
        ):
            raise ContractError(f"catalog query has an invalid item digest: {path}")
        item_path = item_store / f"{digest}.json"
        if not item_path.is_file():
            raise ContractError(f"catalog item is missing or corrupt: {item_path}")
        raw_encoded = item_path.read_bytes()
        if hashlib.sha256(raw_encoded).hexdigest() != digest:
            raise ContractError(f"catalog item is missing or corrupt: {item_path}")
        try:
            raw_item = json.loads(raw_encoded)
        except json.JSONDecodeError as error:
            raise ContractError(f"catalog item is not valid JSON: {item_path}") from error
        raw_assets = raw_item.get("assets") if isinstance(raw_item, dict) else None
        raw_properties = raw_item.get("properties") if isinstance(raw_item, dict) else None
        summary_assets = item.get("assets") if isinstance(item, dict) else None
        summary_properties = item.get("properties") if isinstance(item, dict) else None
        if (
            not isinstance(raw_item, dict)
            or raw_item.get("id") != item.get("id")
            or raw_item.get("collection") != item.get("collection")
            or raw_item.get("bbox") != item.get("bbox")
            or not isinstance(raw_assets, dict)
            or not isinstance(summary_assets, dict)
            or not isinstance(raw_properties, dict)
            or not isinstance(summary_properties, dict)
        ):
            raise ContractError(f"catalog item summary does not match raw item: {path}")
        if any(
            raw_properties.get(key) != value
            for key, value in summary_properties.items()
        ):
            raise ContractError(f"catalog properties do not match raw item: {path}")
        for asset_key, href in summary_assets.items():
            raw_asset = raw_assets.get(asset_key)
            if not isinstance(raw_asset, dict) or raw_asset.get("href") != href:
                raise ContractError(
                    f"catalog asset summary does not match raw item: {path}/{asset_key}"
                )
    return encoded, payload


def discover_catalogs(
    records: Iterable[PointYear],
    output_root: str | Path,
    catalog: STACCatalog,
    modalities: Iterable[str] = ("s1", "s2"),
) -> list[CatalogSnapshotResult]:
    """Discover missing work-tile catalogs; completed immutable snapshots are reused."""
    root = Path(output_root)
    requested = tuple(sorted(set(modalities)))
    if not requested or any(value not in {"s1", "s2"} for value in requested):
        raise ContractError("modalities must contain s1 and/or s2")

    results = []
    for tile in build_work_tiles(records):
        for modality in requested:
            destination = catalog_snapshot_path(root, tile, modality)
            if destination.exists():
                if not _digest_path(destination).is_file():
                    payload = catalog.search(tile, modality)
                    results.append(
                        write_content_addressed_catalog_snapshot(
                            destination,
                            root / "items",
                            payload,
                        )
                    )
                    continue
                encoded, existing = _validate_persisted_catalog(
                    destination, root / "items", tile, modality, catalog.profile
                )
                results.append(
                    CatalogSnapshotResult(
                        path=str(destination),
                        sha256=hashlib.sha256(encoded).hexdigest(),
                        item_count=len(existing.get("items", [])),
                        item_documents_written=0,
                        reused=True,
                    )
                )
                continue
            payload = catalog.search(tile, modality)
            results.append(
                write_content_addressed_catalog_snapshot(
                    destination,
                    root / "items",
                    payload,
                )
            )
    return results
