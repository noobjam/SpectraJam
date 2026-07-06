from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import shapely

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
    PixelTimelines,
    scale_s1_amplitude,
    select_s1_daily_mosaic,
    select_s2_daily_mosaic,
)
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


class PreprocessingTests(unittest.TestCase):
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
