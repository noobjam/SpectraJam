from __future__ import annotations

import hashlib
import json
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
        digest = hashlib.sha256(
            f"{self.country}:{self.spatial_block}:{self.year}".encode()
        ).hexdigest()[:16]
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

    def __init__(self, profile: ProviderProfile = MPC_V11, request_retries: int = 2):
        self.profile = profile
        self.request_retries = request_retries
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
                backoff_factor=0.5,
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
                f"all {len(items)} STAC items were missing required {modality} assets"
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
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_bytes(encoded)
    temporary.replace(destination)
    return hashlib.sha256(encoded).hexdigest(), len(payload["items"])
