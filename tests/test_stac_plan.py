from datetime import UTC, datetime

from spectrajam.contracts import CANONICAL_S1_BANDS, CANONICAL_S2_BANDS, PointYear
from spectrajam.stac import STACCatalog, WorkTile, build_work_tiles


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
    def __init__(self, assets):
        self.id = "item"
        self.datetime = datetime(2023, 1, 1, tzinfo=UTC)
        self.collection_id = "collection"
        self.bbox = [1, 2, 3, 4]
        self.properties = {}
        self.assets = {key: _Asset(f"https://example/{key}") for key in assets}

    def to_dict(self):
        return {"type": "Feature", "id": self.id, "assets": {key: {} for key in self.assets}}


class _Search:
    def __init__(self, item):
        self.item = item

    def items(self):
        return iter([self.item])


class _Client:
    def __init__(self, item):
        self.item = item

    def search(self, **_kwargs):
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
