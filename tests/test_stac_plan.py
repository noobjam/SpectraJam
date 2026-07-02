import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from spectrajam.contracts import (
    CANONICAL_S1_BANDS,
    CANONICAL_S2_BANDS,
    ContractError,
    PointYear,
)
from spectrajam.stac import (
    MPC_V11,
    STACCatalog,
    WorkTile,
    build_work_tiles,
    catalog_snapshot_path,
    discover_catalogs,
    read_catalog_snapshot,
    write_content_addressed_catalog_snapshot,
)


def _row(sample: str, lon: float, year: int) -> PointYear:
    return PointYear(
        sample_id=sample,
        candidate_id=sample,
        country="ISR",
        longitude=lon,
        latitude=31.0,
        spatial_block="block-a",
        stratum="desert",
        inclusion_probability=1.0,
        spatial_split="train",
        year_split="train",
        year=year,
    )


def test_stac_queries_are_grouped_by_work_tile_and_year() -> None:
    tiles = build_work_tiles([_row("a", 34.0, 2022), _row("b", 34.1, 2022), _row("c", 34.2, 2023)])
    assert len(tiles) == 2
    assert sorted(tile.sample_count for tile in tiles) == [1, 2]
    assert tiles[0].bbox[0] < 34.0


class _Asset:
    def __init__(self, href: str):
        self.href = href


class _Item:
    def __init__(self, assets, orbit="ascending"):
        self.id = "item"
        self.datetime = datetime(2023, 1, 1, tzinfo=UTC)
        self.collection_id = "collection"
        self.bbox = [1, 2, 3, 4]
        self.properties = {"sat:orbit_state": orbit} if orbit else {}
        self.assets = {key: _Asset(f"https://example/{key}") for key in assets}

    def to_dict(self):
        return {
            "type": "Feature",
            "id": self.id,
            "collection": self.collection_id,
            "bbox": self.bbox,
            "properties": self.properties,
            "assets": {
                key: {"href": asset.href} for key, asset in self.assets.items()
            },
        }


class _Search:
    def __init__(self, item):
        self.item = item

    def items(self):
        return iter([self.item])


class _Client:
    def __init__(self, item):
        self.item = item
        self.calls = 0

    def search(self, **_kwargs):
        self.calls += 1
        return _Search(self.item)


def test_catalog_carries_explicit_checkpoint_band_order() -> None:
    tile = WorkTile("RWA", "block", 2023, (29, -2, 30, -1), 10)
    s2_catalog = STACCatalog()
    s2_catalog._client = _Client(_Item([*CANONICAL_S2_BANDS, "SCL"]))
    assert s2_catalog.search(tile, "s2")["band_order"] == list(CANONICAL_S2_BANDS)

    s1_catalog = STACCatalog()
    s1_catalog._client = _Client(_Item([value.lower() for value in CANONICAL_S1_BANDS]))
    payload = s1_catalog.search(tile, "s1")
    assert payload["band_order"] == list(CANONICAL_S1_BANDS)
    assert "raw_item" in payload["items"][0]


def test_discovery_is_content_addressed_and_resumable(tmp_path) -> None:
    records = [
        _row("a", 34.0, 2023),
        replace(_row("b", 34.2, 2023), spatial_block="block-b"),
    ]
    catalog = STACCatalog()
    catalog._client = _Client(_Item([*CANONICAL_S2_BANDS, "SCL"]))

    first = discover_catalogs(records, tmp_path, catalog, modalities=("s2",))
    assert len(first) == 2
    assert sum(result.item_documents_written for result in first) == 1
    assert len(list((tmp_path / "items").glob("*.json"))) == 1
    assert all(not result.reused for result in first)
    assert all(Path(f"{result.path}.sha256").is_file() for result in first)

    resumed = discover_catalogs(records, tmp_path, catalog, modalities=("s2",))
    assert len(resumed) == 2
    assert all(result.reused for result in resumed)
    assert catalog._client.calls == 2

    moved = [replace(records[0], longitude=35.0), records[1]]
    refreshed = discover_catalogs(moved, tmp_path, catalog, modalities=("s2",))
    assert sum(result.reused for result in refreshed) == 1
    assert catalog._client.calls == 3

    changed_profile = replace(
        MPC_V11,
        collections={"s1": "sentinel-1-rtc", "s2": "different-s2-collection"},
    )
    changed_catalog = STACCatalog(changed_profile)
    changed_catalog._client = catalog._client
    with pytest.raises(ContractError, match="identity or schema"):
        discover_catalogs(records, tmp_path, changed_catalog, modalities=("s2",))

    query_path = Path(first[0].path)
    original_query = query_path.read_bytes()
    query = json.loads(original_query)
    query["items"][0]["assets"]["B04"] = "https://wrong.example/B04"
    query_path.write_text(json.dumps(query, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ContractError, match="checksum mismatch"):
        discover_catalogs(records, tmp_path, catalog, modalities=("s2",))

    tampered = query_path.read_bytes()
    Path(f"{query_path}.sha256").write_text(f"{hashlib.sha256(tampered).hexdigest()}\n")
    with pytest.raises(ContractError, match="asset summary"):
        discover_catalogs(records, tmp_path, catalog, modalities=("s2",))
    query_path.write_bytes(original_query)
    Path(f"{query_path}.sha256").write_text(
        f"{hashlib.sha256(original_query).hexdigest()}\n"
    )

    next((tmp_path / "items").glob("*.json")).write_text("corrupt")
    with pytest.raises(ContractError, match="missing or corrupt"):
        discover_catalogs(records, tmp_path, catalog, modalities=("s2",))


def test_immutable_query_refuses_different_content(tmp_path) -> None:
    catalog = STACCatalog()
    catalog._client = _Client(_Item([*CANONICAL_S2_BANDS, "SCL"]))
    tile = WorkTile("RWA", "block", 2023, (29, -2, 30, -1), 10)
    payload = catalog.search(tile, "s2")
    destination = tmp_path / "query.json"
    write_content_addressed_catalog_snapshot(destination, tmp_path / "items", payload)

    payload["query"]["collection"] = "different"
    with pytest.raises(ContractError, match="immutable"):
        write_content_addressed_catalog_snapshot(destination, tmp_path / "items", payload)


def test_read_catalog_snapshot_returns_sorted_typed_refs_with_full_properties(
    tmp_path,
) -> None:
    catalog = STACCatalog()
    catalog._client = _Client(_Item([*CANONICAL_S2_BANDS, "SCL"]))
    tile = WorkTile("RWA", "block", 2023, (29, -2, 30, -1), 10)
    later = catalog.search(tile, "s2")["items"][0]
    later["id"] = "item-later"
    later["datetime"] = "2023-02-01T00:00:00+00:00"
    later["raw_item"]["id"] = "item-later"
    later["raw_item"]["properties"]["private:full_property"] = "kept"
    earlier = deepcopy(later)
    earlier["id"] = "item-earlier"
    earlier["datetime"] = "2023-01-01T01:00:00+01:00"
    earlier["raw_item"]["id"] = "item-earlier"
    payload = catalog.search(tile, "s2")
    payload["items"] = [later, earlier]
    path = catalog_snapshot_path(tmp_path, tile, "s2")
    result = write_content_addressed_catalog_snapshot(path, tmp_path / "items", payload)

    snapshot = read_catalog_snapshot(tmp_path, tile, "s2")

    assert snapshot.path == path
    assert snapshot.sha256 == result.sha256
    assert snapshot.tile == tile
    assert snapshot.modality == "s2"
    assert [item.id for item in snapshot.items] == ["item-earlier", "item-later"]
    assert snapshot.items[0].acquired.tzinfo is UTC
    assert snapshot.items[0].acquired == datetime(2023, 1, 1, tzinfo=UTC)
    assert snapshot.items[0].assets["B04"] == "https://example/B04"
    assert snapshot.items[0].properties["private:full_property"] == "kept"
    assert snapshot.items[0].bbox == (1.0, 2.0, 3.0, 4.0)


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("not-a-datetime", "invalid acquisition datetime"),
        ("2023-01-01T00:00:00", "timezone"),
        ("2024-01-01T00:00:00+00:00", "does not match work-tile year"),
    ],
)
def test_read_catalog_snapshot_rejects_invalid_acquisition_datetime(
    tmp_path, value: str, message: str
) -> None:
    catalog = STACCatalog()
    catalog._client = _Client(_Item([*CANONICAL_S2_BANDS, "SCL"]))
    tile = WorkTile("RWA", "block", 2023, (29, -2, 30, -1), 10)
    payload = catalog.search(tile, "s2")
    payload["items"][0]["datetime"] = value
    path = catalog_snapshot_path(tmp_path, tile, "s2")
    write_content_addressed_catalog_snapshot(path, tmp_path / "items", payload)

    with pytest.raises(ContractError, match=message):
        read_catalog_snapshot(tmp_path, tile, "s2")


def test_s1_discovery_rejects_unknown_orbit_state() -> None:
    tile = WorkTile("ISR", "block", 2023, (34, 31, 35, 32), 1)
    catalog = STACCatalog()
    catalog._client = _Client(
        _Item([value.lower() for value in CANONICAL_S1_BANDS], orbit=None)
    )
    with pytest.raises(ContractError, match="rejected"):
        catalog.search(tile, "s1")


def test_missing_query_receipt_is_reconciled_by_requery(tmp_path) -> None:
    records = [_row("a", 34.0, 2023)]
    catalog = STACCatalog()
    catalog._client = _Client(_Item([*CANONICAL_S2_BANDS, "SCL"]))
    first = discover_catalogs(records, tmp_path, catalog, modalities=("s2",))[0]
    Path(f"{first.path}.sha256").unlink()

    resumed = discover_catalogs(records, tmp_path, catalog, modalities=("s2",))[0]
    assert resumed.reused
    assert Path(f"{first.path}.sha256").is_file()
    assert catalog._client.calls == 2
