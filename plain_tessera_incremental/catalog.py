from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any


S2_ASSETS = ("B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12", "SCL")
S1_ASSETS = ("vv", "vh")


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def snapshot_sha256(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(snapshot)).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_canonical_json(value))
    temporary.replace(path)


class MPCCatalog:
    """Unsigned, immutable STAC discovery; assets are signed only when read."""

    def __init__(
        self,
        endpoint: str,
        s2_collection: str,
        s1_collection: str,
        request_retries: int = 3,
    ):
        self.endpoint = endpoint
        self.collections = {"s2": s2_collection, "s1": s1_collection}
        self.request_retries = request_retries
        self._client = None

    def _open(self):
        if self._client is None:
            try:
                from pystac_client import Client
                from pystac_client.stac_api_io import StacApiIO
                from urllib3.util.retry import Retry
            except ImportError as error:
                raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
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
            self._client = Client.open(self.endpoint, stac_io=StacApiIO(max_retries=retry))
        return self._client

    def _search(
        self,
        modality: str,
        bbox_wgs84: tuple[float, float, float, float],
        start: date,
        end_exclusive: date,
    ) -> list[dict[str, Any]]:
        if modality not in self.collections:
            raise ValueError(f"unknown modality: {modality}")
        end_inclusive = datetime.combine(end_exclusive, datetime.min.time(), tzinfo=UTC) - timedelta(
            microseconds=1
        )
        start_time = datetime.combine(start, datetime.min.time(), tzinfo=UTC)
        search = self._open().search(
            collections=[self.collections[modality]],
            bbox=bbox_wgs84,
            datetime=f"{start_time.isoformat()}/{end_inclusive.isoformat()}",
            max_items=None,
        )
        required = S2_ASSETS if modality == "s2" else S1_ASSETS
        records: list[tuple[datetime, str, dict[str, Any]]] = []
        for item in search.items():
            if item.datetime is None:
                continue
            observed = item.datetime.astimezone(UTC)
            if not start_time <= observed < datetime.combine(
                end_exclusive, datetime.min.time(), tzinfo=UTC
            ):
                continue
            if any(asset not in item.assets for asset in required):
                continue
            records.append((observed, item.id, item.to_dict()))
        return [raw for _, _, raw in sorted(records, key=lambda value: (value[0], value[1]))]

    def load_or_create_snapshot(
        self,
        path: Path,
        work_tile_key: str,
        bbox_wgs84: tuple[float, float, float, float],
        start: date,
        end_exclusive: date,
    ) -> dict[str, Any]:
        query = {
            "endpoint": self.endpoint,
            "collections": self.collections,
            "bbox_wgs84": list(bbox_wgs84),
            "start": start.isoformat(),
            "end_exclusive": end_exclusive.isoformat(),
            "item_cloud_filter": None,
            "work_tile_key": work_tile_key,
        }
        if path.is_file():
            snapshot = json.loads(path.read_text())
            if snapshot.get("query") != query:
                raise RuntimeError(f"cached STAC query does not match this run: {path}")
            return snapshot
        snapshot = {
            "schema_version": 1,
            "query": query,
            "s2_items": self._search("s2", bbox_wgs84, start, end_exclusive),
            "s1_items": self._search("s1", bbox_wgs84, start, end_exclusive),
        }
        _atomic_json(path, snapshot)
        return snapshot


def unsigned_items(raw_items: list[dict[str, Any]]):
    try:
        import pystac
    except ImportError as error:
        raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
    return [pystac.Item.from_dict(raw) for raw in raw_items]


def signed_items(raw_items: list[dict[str, Any]]):
    try:
        import planetary_computer
    except ImportError as error:
        raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
    items = unsigned_items(raw_items)
    return [planetary_computer.sign(item) for item in items]
