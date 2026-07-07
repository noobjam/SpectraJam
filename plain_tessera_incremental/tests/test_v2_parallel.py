from __future__ import annotations

import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from plain_tessera_incremental.geometry import RasterWindow
from plain_tessera_incremental.materialize import S2_BANDS, MPCMaterializer, epoch_day


def _item(identifier: str, observed: date, orbit: str | None = None) -> SimpleNamespace:
    properties = {} if orbit is None else {"sat:orbit_state": orbit}
    return SimpleNamespace(
        id=identifier,
        datetime=datetime(observed.year, observed.month, observed.day, tzinfo=UTC),
        properties=properties,
        assets={},
        bbox=(-180.0, -90.0, 180.0, 90.0),
    )


class ParallelMaterializationTests(unittest.TestCase):
    def test_concurrent_stack_serializes_signing_but_not_raster_reads(self) -> None:
        state_lock = threading.Lock()
        start = threading.Barrier(2)
        compute = threading.Barrier(2)
        signing_active = 0
        signing_peak = 0
        compute_active = 0
        compute_peak = 0

        class FakeStack:
            def __init__(self, assets):
                self.coords = {"band": SimpleNamespace(values=np.asarray(assets, dtype=object))}
                self.sizes = {"time": 1, "y": 1, "x": 1}
                self.values = np.ones((1, len(assets), 1, 1), dtype=np.float32)

            def transpose(self, *dimensions):
                return self

            def compute(self, **kwargs):
                nonlocal compute_active, compute_peak
                with state_lock:
                    compute_active += 1
                    compute_peak = max(compute_peak, compute_active)
                try:
                    compute.wait(timeout=2)
                finally:
                    with state_lock:
                        compute_active -= 1
                return self

        def sign(raw_items):
            nonlocal signing_active, signing_peak
            with state_lock:
                signing_active += 1
                signing_peak = max(signing_peak, signing_active)
            try:
                time.sleep(0.05)
                return raw_items
            finally:
                with state_lock:
                    signing_active -= 1

        def run_stack(identifier):
            start.wait(timeout=2)
            return materializer._stack(
                [identifier],
                ["B04"],
                RasterWindow(32631, 50_000, 0, 1, 1, 10),
                "nearest",
                rescale=False,
            )

        materializer = MPCMaterializer(read_retries=0, group_workers=2)
        with (
            patch(
                "plain_tessera_incremental.materialize.detached_item_dicts",
                side_effect=lambda items: [{"id": str(items[0])}],
            ),
            patch(
                "plain_tessera_incremental.materialize.signed_item_dicts",
                side_effect=sign,
            ) as signer,
            patch(
                "stackstac.stack",
                side_effect=lambda *args, **kwargs: FakeStack(kwargs["assets"]),
            ),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            results = list(executor.map(run_stack, ("first", "second")))

        self.assertEqual(signer.call_count, 2)
        self.assertEqual(signing_peak, 1)
        self.assertEqual(compute_peak, 2)
        self.assertEqual([result.shape for result in results], [(1, 1, 1, 1)] * 2)

    def test_group_parallelism_is_bounded_and_output_order_is_chronological(self) -> None:
        start = date(2024, 9, 1)
        s2_items = [_item(f"s2-{ordinal}", start + timedelta(days=ordinal)) for ordinal in range(3)]
        s1_items = [
            _item(f"s1-{orbit}-{ordinal}", start + timedelta(days=ordinal), orbit)
            for ordinal in range(2)
            for orbit in ("ascending", "descending")
        ]
        barrier = threading.Barrier(3)
        lock = threading.Lock()
        active = 0
        peak = 0
        s2_completion: list[str] = []
        release_middle = threading.Event()
        release_earliest = threading.Event()

        def stack(items, assets, grid, resampling, rescale):
            nonlocal active, peak
            identifier = items[0].id
            with lock:
                active += 1
                peak = max(peak, active)
            try:
                if list(assets) == ["SCL"]:
                    barrier.wait(timeout=2)
                    return np.full((1, 1, 1, 1), 4, dtype=np.float32)
                if tuple(assets) == tuple(S2_BANDS):
                    ordinal = int(identifier.rsplit("-", 1)[1])
                    if ordinal == 2:
                        with lock:
                            s2_completion.append(identifier)
                        release_middle.set()
                    elif ordinal == 1:
                        release_middle.wait(timeout=2)
                        with lock:
                            s2_completion.append(identifier)
                        release_earliest.set()
                    else:
                        release_earliest.wait(timeout=2)
                        with lock:
                            s2_completion.append(identifier)
                    return np.full((1, 10, 1, 1), 1100 + ordinal, dtype=np.float32)
                return np.ones((1, 2, 1, 1), dtype=np.float32)
            finally:
                with lock:
                    active -= 1

        materializer = MPCMaterializer(read_retries=0, group_workers=3)
        with (
            patch(
                "plain_tessera_incremental.materialize.unsigned_items",
                side_effect=[s2_items, s1_items],
            ),
            patch.object(materializer, "_stack", side_effect=stack),
        ):
            timelines = materializer.materialize(
                [],
                [],
                RasterWindow(32631, 50_000, 0, 1, 1, 10),
                ("pixel",),
                np.array([0], dtype=np.int64),
                np.array([0], dtype=np.int64),
            )

        self.assertEqual(peak, 3)
        self.assertEqual(s2_completion, ["s2-2", "s2-1", "s2-0"])
        np.testing.assert_array_equal(
            timelines.s2_days,
            [epoch_day(start + timedelta(days=ordinal)) for ordinal in range(3)],
        )
        np.testing.assert_array_equal(
            timelines.s1a_days,
            [epoch_day(start + timedelta(days=ordinal)) for ordinal in range(2)],
        )
        np.testing.assert_array_equal(timelines.s1d_days, timelines.s1a_days)
        np.testing.assert_array_equal(timelines.s2_values[:, 0, 0], [100, 101, 102])

    def test_failure_cancels_queued_groups_and_redacts_signed_url(self) -> None:
        start = date(2024, 9, 1)
        s2_items = [
            _item(f"s2-{ordinal:02d}", start + timedelta(days=ordinal)) for ordinal in range(16)
        ]
        release_running_group = threading.Event()
        started: list[str] = []

        def stack(items, assets, grid, resampling, rescale):
            identifier = items[0].id
            started.append(identifier)
            if identifier == "s2-00":
                raise RuntimeError("https://blob.invalid/a.tif?sig=top-secret")
            release_running_group.wait(timeout=2)
            if list(assets) == ["SCL"]:
                return np.full((1, 1, 1, 1), 4, dtype=np.float32)
            return np.full((1, 10, 1, 1), 1100, dtype=np.float32)

        timer = threading.Timer(0.2, release_running_group.set)
        timer.start()
        materializer = MPCMaterializer(read_retries=0, group_workers=1)
        try:
            with (
                patch(
                    "plain_tessera_incremental.materialize.unsigned_items",
                    side_effect=[s2_items, []],
                ),
                patch.object(materializer, "_stack", side_effect=stack),
                self.assertRaisesRegex(
                    RuntimeError, "parallel raster materialization failed"
                ) as raised,
            ):
                materializer.materialize(
                    [],
                    [],
                    RasterWindow(32631, 50_000, 0, 1, 1, 10),
                    ("pixel",),
                    np.array([0], dtype=np.int64),
                    np.array([0], dtype=np.int64),
                )
        finally:
            release_running_group.set()
            timer.cancel()

        message = str(raised.exception)
        self.assertIn("https://blob.invalid/a.tif?<redacted>", message)
        self.assertNotIn("top-secret", message)
        self.assertLess(len(set(started)), len(s2_items))

    def test_group_workers_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "group_workers must be positive"):
            MPCMaterializer(group_workers=0)


if __name__ == "__main__":
    unittest.main()
