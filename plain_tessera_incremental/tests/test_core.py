from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import shapely

from plain_tessera_incremental.catalog import (
    S2_ASSETS,
    detached_item_dicts,
    signed_item_dicts,
    unsigned_items,
)
from plain_tessera_incremental.config import load_config
from plain_tessera_incremental.geometry import (
    PixelCell,
    RasterChunk,
    RasterWindow,
    pixel_cells_for_geometry,
)
from plain_tessera_incremental.inference import (
    WindowEmbeddings,
    bucket_size,
    build_resample_indices,
    day_of_year,
)
from plain_tessera_incremental.materialize import (
    MPCMaterializer,
    NoSpatialCoverageError,
    PixelTimelines,
    _items_intersecting_grid,
    scale_s1_amplitude,
    select_s1_daily_mosaic,
    select_s2_daily_mosaic,
)
from plain_tessera_incremental.pipeline import preflight, prepare_field_pixels
from plain_tessera_incremental.storage import write_embedding_shard
from plain_tessera_incremental.windows import PrefixWindow
from plain_tessera_incremental.windows import build_prefix_windows


class WindowTests(unittest.TestCase):
    def test_exact_four_cumulative_prefixes(self) -> None:
        windows = build_prefix_windows(
            "2024-09-01",
            ["2025-01-01", "2025-05-01", "2025-09-01", "2026-01-01"],
        )
        self.assertEqual([window.duration_days for window in windows], [122, 242, 365, 487])
        self.assertEqual(windows[-1].end_exclusive.isoformat(), "2026-01-01")

    def test_requires_four_cutoffs(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly four"):
            build_prefix_windows("2024-09-01", ["2025-01-01"])

    def test_pilot_config_has_the_full_window_contract(self) -> None:
        config = load_config(
            Path(__file__).parents[1] / "config_worldcover_noncrop_pilot_w2.yaml"
        )
        self.assertEqual([window.window_id for window in config.windows], ["w1", "w2", "w3", "w4"])


class GeometryTests(unittest.TestCase):
    def test_pixel_center_rule_and_global_ids(self) -> None:
        geometry = shapely.box(0, 0, 20, 20)
        cells = pixel_cells_for_geometry(geometry, epsg=32631, resolution_m=10)
        self.assertEqual(
            {(cell.x_index, cell.y_index) for cell in cells},
            {(0, 0), (1, 0), (0, 1), (1, 1)},
        )
        self.assertEqual(len({cell.pixel_id for cell in cells}), 4)

    def test_polygon_hole_removes_covered_center(self) -> None:
        outer = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
        hole = [(10, 10), (20, 10), (20, 20), (10, 20), (10, 10)]
        geometry = shapely.Polygon(outer, [hole])
        cells = pixel_cells_for_geometry(geometry, epsg=32631, resolution_m=10)
        self.assertNotIn((1, 1), {(cell.x_index, cell.y_index) for cell in cells})
        self.assertEqual(len(cells), 8)

    def test_chunk_indices_are_top_down(self) -> None:
        chunk = RasterChunk(32631, 0, 0, cells=256, resolution_m=10)
        self.assertEqual(chunk.local_indices(PixelCell(32631, 0, 0)), (255, 0))
        self.assertEqual(chunk.local_indices(PixelCell(32631, 255, 255)), (0, 255))

    def test_raster_window_is_tight_around_requested_pixels(self) -> None:
        pixels = (PixelCell(32631, 10, 20), PixelCell(32631, 12, 23))
        window = RasterWindow.from_pixels(pixels)
        self.assertEqual((window.width, window.height), (3, 4))
        self.assertEqual(window.local_indices(pixels[0]), (3, 0))
        self.assertEqual(window.local_indices(pixels[1]), (0, 2))

    def test_wkt_remains_authoritative_when_auxiliary_coordinate_is_outside(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.yaml")
        geometry = shapely.box(3.0, 1.0, 3.0003, 1.0003)
        source = pd.DataFrame(
            {
                "LONGITUDE": [4.0],
                "LATITUDE": [2.0],
                "QUADKEY": ["q"],
                "landcover": ["crop"],
                "wkt": [geometry.wkt],
                "id": [622],
            }
        )

        outside_fields, outside_pixels, outside_memberships = prepare_field_pixels(
            source, config
        )
        source.loc[0, ["LONGITUDE", "LATITUDE"]] = [3.00015, 1.00015]
        inside_fields, inside_pixels, _ = prepare_field_pixels(source, config)

        self.assertEqual(outside_fields.loc[0, "geometry_status"], "valid")
        self.assertEqual(
            outside_fields.loc[0, "coordinate_status"], "outside_wkt_bounds"
        )
        self.assertEqual(inside_fields.loc[0, "coordinate_status"], "within_wkt_bounds")
        self.assertEqual(
            int(outside_fields.loc[0, "utm_epsg"]),
            int(inside_fields.loc[0, "utm_epsg"]),
        )
        self.assertEqual(set(outside_pixels["pixel_id"]), set(inside_pixels["pixel_id"]))
        self.assertGreater(int(outside_fields.loc[0, "pixel_count"]), 0)
        self.assertFalse(outside_pixels.empty)
        self.assertFalse(outside_memberships.empty)

    def test_preflight_reports_task_and_embedding_cardinality(self) -> None:
        config = load_config(Path(__file__).parents[1] / "config.yaml")
        source = pd.DataFrame(
            {
                "LONGITUDE": [3.00015],
                "LATITUDE": [1.00015],
                "QUADKEY": ["q"],
                "landcover": ["crop"],
                "wkt": [shapely.box(3.0, 1.0, 3.0003, 1.0003).wkt],
                "id": [622],
            }
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_path = root / "input.parquet"
            source.to_parquet(input_path)
            checkpoint = root / "checkpoint.pt"
            checkpoint.write_bytes(b"fixture")
            config = replace(
                config,
                input_parquet=input_path,
                checkpoint_path=checkpoint,
                checkpoint_sha256=None,
                output_dir=root / "output",
                device="cpu",
            )
            with patch("plain_tessera_incremental.pipeline.runtime_identity", return_value={}):
                result = preflight(config)
        self.assertGreater(result["unique_pixel_count"], 0)
        self.assertGreaterEqual(result["estimated_task_count"], 1)
        self.assertEqual(result["expected_embedding_rows"], result["field_pixel_membership_count"] * 4)


class PreprocessingTests(unittest.TestCase):
    def test_signed_stack_items_do_not_resolve_stac_root_links(self) -> None:
        import pystac

        raw = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "scene",
            "geometry": {"type": "Point", "coordinates": [3.0, 1.0]},
            "bbox": [3.0, 1.0, 3.0, 1.0],
            "properties": {"datetime": "2024-09-01T00:00:00Z"},
            "links": [
                {
                    "rel": "root",
                    "href": "https://unreachable.invalid/api/stac/v1",
                }
            ],
            "assets": {"B04": {"href": "https://example.invalid/B04.tif"}},
        }

        def sign_without_network(item):
            item["assets"]["B04"]["href"] += "?sig=fresh"
            return item

        with (
            patch("planetary_computer.sign", side_effect=sign_without_network),
            patch.object(
                pystac.Link,
                "resolve_stac_object",
                side_effect=AssertionError("STAC root must not be resolved"),
            ),
        ):
            serialized = signed_item_dicts(detached_item_dicts(unsigned_items([raw])))

        self.assertEqual(serialized[0]["id"], "scene")
        self.assertEqual(serialized[0]["links"][0]["href"], raw["links"][0]["href"])
        self.assertEqual(
            serialized[0]["assets"]["B04"]["href"],
            "https://example.invalid/B04.tif?sig=fresh",
        )

    def test_raster_retry_resigns_plain_items_without_logging_sas_query(self) -> None:
        raw = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "scene",
            "geometry": {"type": "Point", "coordinates": [3.0, 1.0]},
            "bbox": [3.0, 1.0, 3.0, 1.0],
            "properties": {"datetime": "2024-09-01T00:00:00Z"},
            "links": [],
            "assets": {"B04": {"href": "https://example.invalid/B04.tif"}},
        }
        materializer = MPCMaterializer(stack_chunksize=1, read_retries=1)
        secret_error = RuntimeError("https://blob.invalid/a.tif?sig=secret")

        with (
            patch(
                "plain_tessera_incremental.materialize.signed_item_dicts",
                return_value=[raw],
            ) as signer,
            patch("plain_tessera_incremental.materialize.time.sleep"),
            patch("stackstac.stack", side_effect=secret_error) as stack,
            self.assertLogs("plain_tessera_incremental.materialize", "WARNING") as logs,
            self.assertRaisesRegex(RuntimeError, "last error type: RuntimeError") as raised,
        ):
            materializer._stack(
                unsigned_items([raw]),
                ["B04"],
                RasterWindow(32631, 0, 0, 1, 1, 10),
                "nearest",
                rescale=False,
            )

        self.assertEqual(signer.call_count, 2)
        self.assertEqual(stack.call_count, 2)
        self.assertIsInstance(stack.call_args.args[0][0], dict)
        self.assertNotIn("sig=secret", str(raised.exception))
        self.assertNotIn("sig=secret", "\n".join(logs.output))
        self.assertIn("https://blob.invalid/a.tif?<redacted>", str(raised.exception))

    def test_materializer_skips_work_tile_scene_outside_tight_pixel_grid(self) -> None:
        raw = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "S2C-edge-scene",
            "geometry": None,
            "bbox": [29.4, -1.75, 29.5, -1.65],
            "properties": {
                "datetime": "2025-02-05T08:11:51.025000Z",
                "proj:code": "EPSG:32735",
            },
            "links": [],
            "assets": {
                name: {
                    "href": f"https://example.invalid/{name}.tif",
                    "proj:bbox": [699960.0, 9690220.0, 809760.0, 9800020.0],
                }
                for name in S2_ASSETS
            },
        }
        materializer = MPCMaterializer(stack_chunksize=1, read_retries=0)

        with patch.object(
            materializer,
            "_stack",
            side_effect=AssertionError("non-overlapping scene must not be stacked"),
        ) as stack:
            timelines = materializer.materialize(
                [raw],
                [],
                RasterWindow(32735, 77000, 981000, 1, 1, 10),
                ("pixel",),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )

        stack.assert_not_called()
        self.assertEqual(timelines.s2_values.shape, (0, 1, 10))
        self.assertEqual(timelines.s2_days.shape, (0,))

    def test_materializer_skips_stackstac_zero_spatial_coverage_without_retry(self) -> None:
        raw = {
            "type": "Feature",
            "stac_version": "1.0.0",
            "id": "edge-scene",
            "geometry": None,
            "bbox": [29.4, -1.75, 29.5, -1.65],
            "properties": {"datetime": "2025-02-05T08:11:51.025000Z"},
            "links": [],
            "assets": {},
        }
        materializer = MPCMaterializer(stack_chunksize=1, read_retries=3)

        with patch.object(
            materializer,
            "_stack",
            side_effect=NoSpatialCoverageError("no spatial coverage"),
        ) as stack:
            timelines = materializer.materialize(
                [raw],
                [],
                RasterWindow(32735, 77000, 981000, 1, 1, 10),
                ("pixel",),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )

        self.assertEqual(stack.call_count, 1)
        self.assertEqual(timelines.s2_days.shape, (0,))

    def test_wrapped_longitude_bounds_are_conservatively_retained(self) -> None:
        item = SimpleNamespace(
            bbox=(179.9, -1.0, -179.9, 1.0),
            assets={},
            properties={},
        )
        retained = _items_intersecting_grid(
            [item],
            [],
            RasterWindow(32660, 0, 0, 1, 1, 10),
            (179.8, -1.0, -179.8, 1.0),
        )
        self.assertEqual(retained, [item])

    def test_s1_scaled_db_reference_values(self) -> None:
        values = scale_s1_amplitude(np.array([1.0, 0.1, 0.0, np.nan], np.float32))
        np.testing.assert_array_equal(values, np.array([10000, 6000, 0, 0], np.int16))

    def test_s2_first_valid_item_and_offset(self) -> None:
        scl = np.array([[9, 4], [4, 5]], np.float32)
        bands = np.zeros((2, 10, 2), np.float32)
        bands[0, :, 1] = 1200
        bands[1, :, 0] = 2500
        bands[1, :, 1] = 3000
        values, valid = select_s2_daily_mosaic(scl, bands)
        self.assertTrue(valid.all())
        np.testing.assert_array_equal(values[0], np.full(10, 1500, np.uint16))
        np.testing.assert_array_equal(values[1], np.full(10, 200, np.uint16))

    def test_s1_first_nonzero_item(self) -> None:
        bands = np.zeros((2, 2, 1), np.float32)
        bands[0, 0, 0] = 1.0
        bands[1, 1, 0] = 0.1
        values, valid = select_s1_daily_mosaic(bands)
        self.assertTrue(bool(valid[0]))
        np.testing.assert_array_equal(values[0], np.array([10000, 6000], np.int16))


class InferencePreparationTests(unittest.TestCase):
    def test_bucket_resampling_matches_upstream_rules(self) -> None:
        np.testing.assert_array_equal(build_resample_indices(3, 3), [0, 1, 2])
        np.testing.assert_array_equal(build_resample_indices(4, 2), [1, 3])
        np.testing.assert_array_equal(build_resample_indices(2, 4), [0, 1, 0, 1])
        self.assertEqual(bucket_size(0), 8)
        self.assertEqual(bucket_size(9), 16)
        self.assertEqual(bucket_size(999), 256)

    def test_day_of_year_crosses_new_year_without_wrapping_order(self) -> None:
        days = np.array(
            [
                (np.datetime64("2024-12-31") - np.datetime64("1970-01-01")).astype(int),
                (np.datetime64("2025-01-01") - np.datetime64("1970-01-01")).astype(int),
            ]
        )
        np.testing.assert_array_equal(day_of_year(days), [366, 1])

    def test_timeline_cache_round_trip(self) -> None:
        timelines = PixelTimelines(
            pixel_ids=("p1",),
            s2_values=np.ones((1, 1, 10), np.uint16),
            s2_valid=np.ones((1, 1), bool),
            s2_days=np.array([1], np.int32),
            s1a_values=np.ones((1, 1, 2), np.int16),
            s1a_valid=np.ones((1, 1), bool),
            s1a_days=np.array([1], np.int32),
            s1d_values=np.empty((0, 1, 2), np.int16),
            s1d_valid=np.empty((0, 1), bool),
            s1d_days=np.empty(0, np.int32),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "timeline.npz"
            timelines.save(path)
            loaded = PixelTimelines.load(path)
        self.assertEqual(loaded.pixel_ids, timelines.pixel_ids)
        np.testing.assert_array_equal(loaded.s2_values, timelines.s2_values)


class StorageTests(unittest.TestCase):
    def test_null_embedding_round_trips_without_fake_vector(self) -> None:
        import pyarrow.parquet as pq

        memberships = pd.DataFrame(
            {
                "pixel_position": [0, 1],
                "field_uid": ["f1", "f2"],
                "source_id": ["1", "2"],
                "landcover": ["crop", "crop"],
                "quadkey": ["q", "q"],
                "pixel_id": ["p1", "p2"],
                "utm_epsg": [32631, 32631],
                "pixel_x_index": [1, 2],
                "pixel_y_index": [3, 4],
                "pixel_longitude": [0.1, 0.2],
                "pixel_latitude": [1.1, 1.2],
            }
        )
        results = WindowEmbeddings(
            embeddings=np.vstack(
                [np.ones((1, 128), np.float32), np.full((1, 128), np.nan, np.float32)]
            ),
            outcome=np.array(["complete", "empty_window"]),
            s2_valid_count=np.array([1, 0], np.int32),
            s1_valid_count=np.array([1, 0], np.int32),
            s2_input_count=np.array([8, 0], np.int32),
            s1_input_count=np.array([8, 0], np.int32),
            s2_source_count=1,
            s1_source_count=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "embeddings.parquet"
            write_embedding_shard(
                memberships,
                results,
                PrefixWindow("w1", 1, date(2024, 9, 1), date(2025, 1, 1)),
                path,
                "run",
                "task",
                "task-fingerprint",
            )
            values = pq.read_table(path, columns=["embedding"])["embedding"].to_pylist()
        self.assertEqual(len(values[0]), 128)
        self.assertIsNone(values[1])


if __name__ == "__main__":
    unittest.main()
