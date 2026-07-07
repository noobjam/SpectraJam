from __future__ import annotations

from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
import shapely

from plain_tessera_incremental import PREPROCESSING_VERSION
from plain_tessera_incremental.catalog import MPCCatalog, S2_ASSETS
from plain_tessera_incremental.config import load_config
from plain_tessera_incremental.geometry import (
    expand_projected_bounds,
    pixel_cells_for_geometry,
    positive_area_pixel_count,
)
from plain_tessera_incremental.pipeline import prepare_field_pixels


ROOT = Path(__file__).parents[1]


def test_v2_defaults_use_separate_output_and_500m_query_halo() -> None:
    config = load_config(ROOT / "config.yaml")

    assert config.output_dir.name == "harvard_tessera_incremental_v2"
    assert config.stac_query_halo_m == 500
    assert config.materialize_workers == 8
    assert PREPROCESSING_VERSION.endswith("-v2")
    replace(config, stac_query_halo_m=250).validate()
    with pytest.raises(ValueError, match="query_halo_m"):
        replace(config, stac_query_halo_m=-1).validate()
    with pytest.raises(ValueError, match="materialize worker"):
        replace(config, materialize_workers=0).validate()


def test_projected_query_halo_expands_bounds_without_changing_shape() -> None:
    original = (1000.0, 2000.0, 21_000.0, 22_000.0)

    assert expand_projected_bounds(original, 500) == (
        500.0,
        1500.0,
        21_500.0,
        22_500.0,
    )
    assert original == (1000.0, 2000.0, 21_000.0, 22_000.0)


def test_positive_area_count_does_not_change_center_membership() -> None:
    narrow_strip = shapely.box(9.0, 1.0, 11.0, 19.0)

    center_cells = pixel_cells_for_geometry(narrow_strip, epsg=32631, resolution_m=10)

    assert center_cells == ()
    assert positive_area_pixel_count(narrow_strip, resolution_m=10, block_pixels=1) == 4


def test_positive_area_count_skips_empty_space_between_multipart_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = shapely.union_all(
        [
            shapely.box(1.0, 1.0, 2.0, 2.0),
            shapely.box(10_001.0, 1.0, 10_002.0, 2.0),
        ]
    )
    original_box = shapely.box
    generated_cells = 0

    def tracked_box(*args, **kwargs):
        nonlocal generated_cells
        cells = original_box(*args, **kwargs)
        generated_cells += int(cells.size)
        return cells

    monkeypatch.setattr(shapely, "box", tracked_box)

    assert positive_area_pixel_count(geometry, resolution_m=10) == 2
    assert generated_cells == 2


def test_positive_area_count_excludes_boundary_only_hole_cell() -> None:
    geometry = shapely.difference(
        shapely.box(0.0, 0.0, 30.0, 30.0),
        shapely.box(10.0, 10.0, 20.0, 20.0),
    )

    assert positive_area_pixel_count(geometry, resolution_m=10) == 8


def test_positive_area_count_shortcuts_large_covered_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = shapely.box(0.0, 0.0, 10_000.0, 10_000.0)
    original_box = shapely.box
    generated_cells = 0

    def tracked_box(*args, **kwargs):
        nonlocal generated_cells
        cells = original_box(*args, **kwargs)
        generated_cells += int(cells.size)
        return cells

    monkeypatch.setattr(shapely, "box", tracked_box)

    assert positive_area_pixel_count(geometry, resolution_m=10) == 1_000_000
    assert generated_cells == 0


def test_catalog_splits_wrapped_bbox_and_deduplicates_items() -> None:
    observed = datetime(2025, 1, 15, 12, tzinfo=UTC)

    class FakeItem:
        def __init__(self, item_id: str):
            self.id = item_id
            self.datetime = observed
            self.assets = {name: object() for name in S2_ASSETS}

        def to_dict(self, transform_hrefs: bool = False):
            assert transform_hrefs is False
            return {"id": self.id, "properties": {"datetime": observed.isoformat()}}

    shared = FakeItem("shared")

    class FakeSearch:
        def __init__(self, items):
            self._items = items

        def items(self):
            return iter(self._items)

    class FakeClient:
        def __init__(self):
            self.bboxes = []

        def search(self, **kwargs):
            bbox = tuple(kwargs["bbox"])
            self.bboxes.append(bbox)
            if bbox[0] >= 0:
                return FakeSearch([shared, FakeItem("east")])
            return FakeSearch([shared, FakeItem("west")])

    client = FakeClient()
    catalog = MPCCatalog(
        "https://example.invalid/stac",
        "sentinel-2-l2a",
        "sentinel-1-rtc",
        request_retries=0,
    )
    catalog._client = client

    items = catalog._search(
        "s2",
        (179.8, -1.0, -179.8, 1.0),
        date(2025, 1, 1),
        date(2025, 2, 1),
    )

    assert client.bboxes == [
        (179.8, -1.0, 180.0, 1.0),
        (-180.0, -1.0, -179.8, 1.0),
    ]
    assert [item["id"] for item in items] == ["east", "shared", "west"]


def test_catalog_rejects_conflicting_duplicate_item_documents() -> None:
    observed = datetime(2025, 1, 15, 12, tzinfo=UTC)

    class FakeItem:
        id = "shared"
        datetime = observed
        assets = {name: object() for name in S2_ASSETS}

        def __init__(self, version: int):
            self.version = version

        def to_dict(self, transform_hrefs: bool = False):
            assert transform_hrefs is False
            return {"id": self.id, "properties": {"version": self.version}}

    class FakeSearch:
        def __init__(self, item):
            self.item = item

        def items(self):
            return iter([self.item])

    class FakeClient:
        def __init__(self):
            self.version = 0

        def search(self, **kwargs):
            self.version += 1
            return FakeSearch(FakeItem(self.version))

    catalog = MPCCatalog(
        "https://example.invalid/stac",
        "sentinel-2-l2a",
        "sentinel-1-rtc",
        request_retries=0,
    )
    catalog._client = FakeClient()

    with pytest.raises(RuntimeError, match="conflicting documents.*shared"):
        catalog._search(
            "s2",
            (179.8, -1.0, -179.8, 1.0),
            date(2025, 1, 1),
            date(2025, 2, 1),
        )


def test_fields_include_projected_diagnostics_and_keep_center_memberships() -> None:
    config = load_config(ROOT / "config.yaml")
    geometry = shapely.box(3.0, 1.0, 3.0003, 1.0003)
    source = pd.DataFrame(
        {
            "LONGITUDE": [3.00015],
            "LATITUDE": [1.00015],
            "QUADKEY": ["q"],
            "landcover": ["crop"],
            "wkt": [geometry.wkt],
            "id": [622],
        }
    )

    fields, pixels, memberships = prepare_field_pixels(source, config)
    _, unbuffered_pixels, unbuffered_memberships = prepare_field_pixels(
        source, replace(config, stac_query_halo_m=0)
    )
    field = fields.iloc[0]

    assert field["area_m2"] > 0
    assert field["bbox_width_m"] > 0
    assert field["bbox_height_m"] > 0
    assert int(field["center_pixel_count"]) == int(field["pixel_count"])
    assert int(field["center_pixel_count"]) == len(pixels)
    assert len(memberships) == len(pixels)
    assert int(field["positive_area_pixel_count"]) >= int(field["center_pixel_count"])
    assert set(unbuffered_pixels["pixel_id"]) == set(pixels["pixel_id"])
    assert set(unbuffered_memberships["pixel_id"]) == set(memberships["pixel_id"])
