from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import requests

from spectrajam.contracts import CANONICAL_S2_BANDS, PointYear, stable_sample_id
from spectrajam.ledger import AcquisitionLedger, Task
from spectrajam.materialize import (
    TransientMaterializationError,
    _compatibility_coordinates,
    _heartbeat_sleep,
    _sample_asset,
    materialize_group_records,
    point_shard_path,
    preflight_materialization,
    run_materialization,
    sample_cog_points,
)
from spectrajam.pointstore import read_point_store
from spectrajam.preprocessing import amplitude_to_mpc_s1, harmonize_mpc_s2
from spectrajam.stac import CatalogItemRef, CatalogSnapshot, WorkTile


def _task(
    candidate: str,
    modality: str,
    longitude: float = 30.0,
    latitude: float = -2.0,
) -> Task:
    sample_id = stable_sample_id("RWA", candidate, 2023)
    return Task(
        task_id=f"task-{candidate}-{modality}",
        sample_id=sample_id,
        country="RWA",
        longitude=longitude,
        latitude=latitude,
        spatial_block="block-1",
        year=2023,
        modality=modality,
        attempts=1,
        max_attempts=3,
    )


def _item(
    item_id: str,
    acquired: str,
    digest_character: str,
    modality: str,
    *,
    orbit: str = "ascending",
    bbox: tuple[float, float, float, float] = (29.0, -3.0, 31.0, -1.0),
) -> CatalogItemRef:
    asset_keys = (*CANONICAL_S2_BANDS, "SCL") if modality == "s2" else ("vv", "vh")
    return CatalogItemRef(
        id=item_id,
        acquired=datetime.fromisoformat(acquired).astimezone(UTC),
        raw_item_sha256=digest_character * 64,
        assets={key: f"unsigned://{item_id}/{key}" for key in asset_keys},
        bbox=bbox,
        properties={"sat:orbit_state": orbit},
    )


def _snapshot(modality: str, items: tuple[CatalogItemRef, ...]) -> CatalogSnapshot:
    tile = WorkTile("RWA", "block-1", 2023, (29.9, -2.1, 30.1, -1.9), 2)
    return CatalogSnapshot(
        path=Path("catalog.json"),
        sha256=("a" if modality == "s2" else "b") * 64,
        tile=tile,
        modality=modality,
        items=items,
    )


def test_s2_uses_first_valid_scl_item_and_reads_only_its_bands() -> None:
    first = _item("first", "2023-06-01T08:00:00+00:00", "1", "s2")
    second = _item("second", "2023-06-01T09:00:00+00:00", "2", "s2")
    cloudy = _item("cloudy", "2023-06-02T08:00:00+00:00", "3", "s2")
    tasks = [_task("one", "s2", 30.0), _task("two", "s2", 30.1)]
    calls: list[tuple[str, int, str]] = []

    def sampler(href, coordinates, resampling, kind, _heartbeat, *_grid):
        calls.append((href, len(coordinates), resampling))
        if href.endswith("first/SCL"):
            return np.array([8, 4], dtype=np.float64)
        if href.endswith("second/SCL"):
            assert len(coordinates) == 1
            return np.array([5], dtype=np.float64)
        if href.endswith("cloudy/SCL"):
            return np.array([9, 9], dtype=np.float64)
        band = href.rsplit("/", 1)[1]
        index = CANONICAL_S2_BANDS.index(band)
        base = 1200 if "second/" in href else 1100
        return np.full(len(coordinates), base + index, dtype=np.float64)

    result = materialize_group_records(
        tasks,
        {"block-1": _snapshot("s2", (first, second, cloudy))},
        sampler=sampler,
        signer=lambda value: value,
        max_attempts=1,
    )

    one, two = (result.rows_by_task[task.task_id] for task in tasks)
    assert len(one) == len(two) == 2
    assert one[0]["source_item_id"] == "second"
    assert one[0]["scl"] == 5
    assert one[0]["bands"] == list(range(200, 210))
    assert two[0]["source_item_id"] == "first"
    assert two[0]["scl"] == 4
    assert two[0]["bands"] == list(range(100, 110))
    assert one[1]["source_item_id"] == two[1]["source_item_id"] == "cloudy"
    assert one[1]["valid"] is two[1]["valid"] is False
    assert one[1]["bands"] == two[1]["bands"] == [0] * 10
    assert not any("cloudy/B" in href for href, _count, _resampling in calls)
    assert sum("first/B" in href for href, _count, _resampling in calls) == 10
    assert sum("second/B" in href for href, _count, _resampling in calls) == 10


def test_s1_selects_first_nonzero_same_day_orbit_item() -> None:
    first = _item("first", "2023-04-01T03:00:00+00:00", "4", "s1")
    second = _item("second", "2023-04-01T04:00:00+00:00", "5", "s1")
    tasks = [_task("one", "s1", 30.0), _task("two", "s1", 30.1)]
    resampling_modes = []

    def sampler(href, coordinates, resampling, _kind, _heartbeat, *_grid):
        resampling_modes.append(resampling)
        if "first/" in href:
            values = [0.0, 0.1] if href.endswith("/vv") else [0.0, 0.2]
            return np.asarray(values[: len(coordinates)], dtype=np.float64)
        assert len(coordinates) == 1
        return np.asarray([0.3 if href.endswith("/vv") else 0.4])

    result = materialize_group_records(
        tasks,
        {"block-1": _snapshot("s1", (first, second))},
        sampler=sampler,
        signer=lambda value: value,
        max_attempts=1,
    )

    one, two = (result.rows_by_task[task.task_id][0] for task in tasks)
    assert one["source_item_id"] == "second"
    assert one["bands"] == amplitude_to_mpc_s1([0.3, 0.4]).tolist()
    assert two["source_item_id"] == "first"
    assert two["bands"] == amplitude_to_mpc_s1([0.1, 0.2]).tolist()
    assert one["orbit"] == two["orbit"] == 1
    assert one["valid"] is two["valid"] is True
    assert set(resampling_modes) == {"nearest"}


def test_s2_keeps_scl_selected_item_when_one_band_is_nodata() -> None:
    first = _item("first", "2023-08-01T08:00:00+00:00", "8", "s2")
    second = _item("second", "2023-08-01T09:00:00+00:00", "9", "s2")
    task = _task("one", "s2")

    def sampler(href, coordinates, _resampling, _kind, _heartbeat, *_grid):
        if href.endswith("/SCL"):
            return np.array([4.0])
        if href.endswith("first/B04"):
            return np.array([np.nan])
        return np.full(len(coordinates), 1600.0)

    result = materialize_group_records(
        [task],
        {"block-1": _snapshot("s2", (first, second))},
        sampler=sampler,
        signer=lambda value: value,
        max_attempts=1,
    )

    row = result.rows_by_task[task.task_id][0]
    assert row["source_item_id"] == "first"
    assert row["bands"] == [0, *([600] * 9)]
    assert row["valid"] is True


def test_s1_mosaics_vv_and_vh_independently() -> None:
    first = _item("first", "2023-09-01T03:00:00+00:00", "c", "s1")
    second = _item("second", "2023-09-01T04:00:00+00:00", "d", "s1")
    task = _task("one", "s1")

    def sampler(href, coordinates, _resampling, _kind, _heartbeat, *_grid):
        values = {
            "unsigned://first/vv": 0.1,
            "unsigned://first/vh": np.nan,
            "unsigned://second/vv": np.nan,
            "unsigned://second/vh": 0.2,
        }
        return np.full(len(coordinates), values[href], dtype=np.float64)

    result = materialize_group_records(
        [task],
        {"block-1": _snapshot("s1", (first, second))},
        sampler=sampler,
        signer=lambda value: value,
        max_attempts=1,
    )

    row = result.rows_by_task[task.task_id][0]
    assert row["bands"] == amplitude_to_mpc_s1([0.1, 0.2]).tolist()
    assert row["source_item_id"] == "mosaic:vv=first;vh=second"
    assert len(row["source_item_sha256"]) == 64
    document = result.provenance_documents[row["source_item_sha256"]]
    assert json.loads(document)["components"] == {
        "vv": {"id": "first", "raw_item_sha256": "c" * 64},
        "vh": {"id": "second", "raw_item_sha256": "d" * 64},
    }
    assert row["valid"] is True


def test_asset_retry_always_signs_the_canonical_unsigned_href() -> None:
    item = _item("item", "2023-05-01T00:00:00+00:00", "6", "s2")
    task = _task("one", "s2")
    signed_inputs: list[str] = []
    attempts = 0

    class RasterioIOError(OSError):
        pass

    def signer(value):
        signed_inputs.append(value)
        return f"{value}?token={len(signed_inputs)}"

    def sampler(href, coordinates, _resampling, _kind, _heartbeat, *_grid):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RasterioIOError("truncated range read")
        if "/SCL?" in href:
            return np.array([4.0])
        return np.full(len(coordinates), 1500.0)

    result = materialize_group_records(
        [task],
        {"block-1": _snapshot("s2", (item,))},
        sampler=sampler,
        signer=signer,
        max_attempts=2,
        base_delay_seconds=0,
        max_delay_seconds=0,
    )

    assert result.rows_by_task[task.task_id][0]["valid"] is True
    assert signed_inputs[0] == signed_inputs[1] == "unsigned://item/SCL"
    assert all("token=" not in value for value in signed_inputs)


def test_final_asset_retry_preserves_retry_after() -> None:
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "17"

    def sampler(*_args):
        raise requests.HTTPError(response=response)

    with pytest.raises(TransientMaterializationError) as captured:
        _sample_asset(
            item_id="item",
            asset_key="B04",
            unsigned_href="unsigned://item/B04",
            coordinates=[(30.0, -2.0)],
            resampling="bilinear",
            kind="s2",
            sampler=sampler,
            signer=lambda value: value,
            max_attempts=1,
            base_delay_seconds=0,
            max_delay_seconds=0,
            heartbeat=lambda: None,
            task_ids=["task"],
            target_crs="EPSG:32735",
            target_resolution=10,
        )
    assert captured.value.retry_after_seconds == 17


def test_retry_wait_renews_heartbeat_in_bounded_chunks(monkeypatch) -> None:
    clock = [0.0]
    sleeps = []
    heartbeats = []

    monkeypatch.setattr("spectrajam.materialize.time.monotonic", lambda: clock[0])

    def fake_sleep(seconds):
        sleeps.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr("spectrajam.materialize.time.sleep", fake_sleep)
    _heartbeat_sleep(95, lambda: heartbeats.append(clock[0]))

    assert sleeps == [30.0, 30.0, 30.0, 5.0]
    assert heartbeats == [30.0, 60.0, 90.0, 95.0]


def test_sample_cog_points_nearest_bilinear_and_nodata(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "asset.tif"
    values = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0, 16, 1, 1),
        nodata=0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dataset:
        dataset.write(values, 1)

    nearest = sample_cog_points(str(path), [(1.5, 14.5)], "nearest", "s2")
    bilinear = sample_cog_points(str(path), [(2.5, 13.5)], "bilinear", "s2")
    nodata = sample_cog_points(str(path), [(0.5, 15.5)], "nearest", "s2")

    assert nearest.tolist() == [17.0]
    assert bilinear.tolist() == [34.0]
    assert np.isnan(nodata[0])

    coarse_path = tmp_path / "coarse.tif"
    with rasterio.open(
        coarse_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dataset:
        dataset.write(np.array([[0, 20], [30, 40]], dtype=np.uint16), 1)
    warped = sample_cog_points(
        str(coarse_path),
        [(0.75, 1.25), (0.25, 0.75)],
        "bilinear",
        "s2",
        target_crs="EPSG:4326",
        target_resolution=0.5,
    )
    assert np.isnan(warped[0])  # GDAL requires the source centre pixel to be valid.
    assert warped[1] == 30.0  # GDAL still samples a valid footprint at a raster edge.

    scl_path = tmp_path / "scl.tif"
    with rasterio.open(
        scl_path,
        "w",
        driver="GTiff",
        width=16,
        height=16,
        count=1,
        dtype="uint8",
        crs="EPSG:4326",
        transform=from_origin(0, 16, 1, 1),
        nodata=0,
        tiled=True,
        blockxsize=16,
        blockysize=16,
    ) as dataset:
        dataset.write(np.zeros((16, 16), dtype=np.uint8), 1)
    scl_zero = sample_cog_points(str(scl_path), [(0.5, 15.5)], "nearest", "scl")
    assert scl_zero.tolist() == [0.0]


def test_warp_uses_native_dtype_before_float64_preprocessing(tmp_path: Path) -> None:
    rasterio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin

    path = tmp_path / "quantized.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dataset:
        dataset.write(np.array([[10, 21], [32, 51]], dtype=np.uint16), 1)
    sampled = sample_cog_points(
        str(path),
        [(0.75, 1.25)],
        "bilinear",
        "s2",
        target_crs="EPSG:4326",
        target_resolution=0.5,
    )
    assert sampled.tolist() == [19.0]  # Native uint16 VRT quantizes 18.75 first.

    threshold_path = tmp_path / "threshold.tif"
    with rasterio.open(
        threshold_path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint16",
        crs="EPSG:4326",
        transform=from_origin(0, 2, 1, 1),
        nodata=0,
    ) as dataset:
        dataset.write(np.array([[999, 1000], [1000, 1000]], dtype=np.uint16), 1)
    threshold = sample_cog_points(
        str(threshold_path),
        [(0.25, 0.75)],
        "bilinear",
        "s2",
        target_crs="EPSG:4326",
        target_resolution=0.5,
    )
    assert threshold.tolist() == [1000.0]
    assert harmonize_mpc_s2(threshold, datetime(2023, 1, 1)).tolist() == [0]


@pytest.mark.parametrize(
    ("country", "target_crs", "easting", "northing"),
    [
        ("ISR", "EPSG:32636", 683500.0, 3611900.0),
        ("RWA", "EPSG:32735", 700100.0, 9777900.0),
    ],
)
def test_compatibility_grid_recovers_lattice_edges_after_csv_roundtrip(
    country, target_crs, easting, northing
) -> None:
    from rasterio.warp import transform

    longitude, latitude = transform(target_crs, "EPSG:4326", [easting], [northing])
    task = Task(
        task_id=f"task-{country}",
        sample_id=f"sample-{country}",
        country=country,
        longitude=round(longitude[0], 10),
        latitude=round(latitude[0], 10),
        spatial_block="block",
        year=2023,
        modality="s2",
        attempts=1,
        max_attempts=3,
    )
    snapped = _compatibility_coordinates([task])[task.task_id]
    x_value, y_value = transform("EPSG:4326", target_crs, [snapped[0]], [snapped[1]])
    assert x_value[0] == pytest.approx(easting + 5, abs=1e-3)
    assert y_value[0] == pytest.approx(northing - 5, abs=1e-3)


def _point(candidate: str, longitude: float) -> PointYear:
    return PointYear(
        sample_id=stable_sample_id("RWA", candidate, 2023),
        candidate_id=candidate,
        country="RWA",
        longitude=longitude,
        latitude=-2.0,
        spatial_block="block-1",
        stratum="test",
        inclusion_probability=1.0,
        spatial_split="train",
        year_split="train",
        year=2023,
    )


def test_runner_publishes_verified_shard_and_resolves_no_source(
    tmp_path: Path, monkeypatch
) -> None:
    records = [_point("inside", 30.0), _point("outside", 40.0)]
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        records,
        ["s2"],
        max_attempts=2,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    item = _item("item", "2023-07-01T00:00:00+00:00", "7", "s2")
    snapshot = _snapshot("s2", (item,))
    monkeypatch.setattr(
        "spectrajam.stac.read_catalog_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )

    def sampler(href, coordinates, _resampling, _kind, _heartbeat, *_grid):
        if href.endswith("/SCL"):
            return np.full(len(coordinates), 4.0)
        return np.full(len(coordinates), 1500.0)

    result = run_materialization(
        ledger=ledger,
        tile_by_identity={("RWA", "block-1", 2023): snapshot.tile},
        catalog_root=tmp_path / "catalog",
        output_root=tmp_path / "points",
        profile=object(),
        worker_id="worker-a",
        batch_points=2,
        lease_seconds=180,
        max_attempts=1,
        base_delay_seconds=0,
        max_delay_seconds=0,
        sampler=sampler,
        signer=lambda value: value,
    )

    assert result.complete == 1
    assert result.no_source_observation == 1
    assert ledger.outcomes() == {
        "complete": 1,
        "insufficient_valid_observations": 0,
        "no_source_observation": 1,
        "terminal_data_error": 0,
    }
    ledger.assert_complete()

    inside_task = _task("inside", "s2")
    shard = point_shard_path(tmp_path / "points", inside_task)
    table = read_point_store(shard, "s2")
    assert table.num_rows == 1
    assert table.to_pylist()[0]["bands"] == [500] * 10
    assert (
        run_materialization(
            ledger=ledger,
            tile_by_identity={("RWA", "block-1", 2023): snapshot.tile},
            catalog_root=tmp_path / "catalog",
            output_root=tmp_path / "points",
            profile=object(),
            worker_id="worker-b",
            batch_points=2,
            lease_seconds=180,
            max_attempts=1,
            base_delay_seconds=0,
            max_delay_seconds=0,
            sampler=sampler,
            signer=lambda value: value,
        ).claimed
        == 0
    )


def test_preflight_opens_representative_s1_and_s2_assets_before_claims(
    tmp_path: Path, monkeypatch
) -> None:
    s2 = _snapshot("s2", (_item("s2-item", "2023-01-01T00:00:00+00:00", "e", "s2"),))
    s1 = _snapshot("s1", (_item("s1-item", "2023-01-01T00:00:00+00:00", "f", "s1"),))
    monkeypatch.setattr(
        "spectrajam.stac.read_catalog_snapshot",
        lambda _root, _tile, modality, _profile: {"s1": s1, "s2": s2}[modality],
    )
    opened = []

    def sampler(href, coordinates, resampling, kind, _heartbeat, *_grid):
        opened.append((href, resampling, kind))
        return np.zeros(len(coordinates), dtype=np.float64)

    digest = preflight_materialization(
        tile_by_identity={("RWA", "block-1", 2023): s2.tile},
        catalog_root=tmp_path / "catalog",
        output_root=tmp_path / "points",
        profile=object(),
        verify_remote_assets=True,
        remote_sampler=sampler,
        remote_signer=lambda value: value,
    )

    assert len(digest) == 64
    assert opened == [
        ("unsigned://s1-item/vv", "nearest", "s1"),
        ("unsigned://s1-item/vh", "nearest", "s1"),
        ("unsigned://s2-item/SCL", "nearest", "scl"),
        ("unsigned://s2-item/B04", "bilinear", "s2"),
    ]


def test_runner_scopes_a_bad_asset_to_its_applicable_tasks(tmp_path: Path, monkeypatch) -> None:
    records = [_point("west", 30.0), _point("east", 30.1)]
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        records,
        ["s2"],
        max_attempts=1,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    bad = _item(
        "bad",
        "2023-07-01T00:00:00+00:00",
        "a",
        "s2",
        bbox=(29.99, -2.01, 30.01, -1.99),
    )
    good = _item(
        "good",
        "2023-07-01T00:00:00+00:00",
        "b",
        "s2",
        bbox=(30.09, -2.01, 30.11, -1.99),
    )
    snapshot = _snapshot("s2", (bad, good))
    monkeypatch.setattr(
        "spectrajam.stac.read_catalog_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )

    class RasterioIOError(OSError):
        pass

    def sampler(href, coordinates, _resampling, _kind, _heartbeat, *_grid):
        if "bad/" in href:
            raise RasterioIOError("truncated read")
        if href.endswith("/SCL"):
            return np.full(len(coordinates), 4.0)
        return np.full(len(coordinates), 1500.0)

    result = run_materialization(
        ledger=ledger,
        tile_by_identity={("RWA", "block-1", 2023): snapshot.tile},
        catalog_root=tmp_path / "catalog",
        output_root=tmp_path / "points",
        profile=object(),
        worker_id="worker-a",
        batch_points=2,
        lease_seconds=180,
        max_attempts=1,
        base_delay_seconds=0,
        max_delay_seconds=0,
        sampler=sampler,
        signer=lambda value: value,
    )

    assert result.failed == 1
    assert result.complete == 1
    assert ledger.summary()["failed"] == 1
    assert ledger.summary()["succeeded"] == 1


def test_runner_publishes_resolvable_s1_composite_provenance(tmp_path: Path, monkeypatch) -> None:
    record = _point("mixed", 30.0)
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [record],
        ["s1"],
        max_attempts=1,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    first = _item("first", "2023-09-01T03:00:00+00:00", "c", "s1")
    second = _item("second", "2023-09-01T04:00:00+00:00", "d", "s1")
    snapshot = _snapshot("s1", (first, second))
    monkeypatch.setattr(
        "spectrajam.stac.read_catalog_snapshot",
        lambda *_args, **_kwargs: snapshot,
    )

    def sampler(href, coordinates, _resampling, _kind, _heartbeat, *_grid):
        values = {
            "unsigned://first/vv": 0.1,
            "unsigned://first/vh": np.nan,
            "unsigned://second/vv": np.nan,
            "unsigned://second/vh": 0.2,
        }
        return np.full(len(coordinates), values[href], dtype=np.float64)

    result = run_materialization(
        ledger=ledger,
        tile_by_identity={("RWA", "block-1", 2023): snapshot.tile},
        catalog_root=tmp_path / "catalog",
        output_root=tmp_path / "points",
        profile=object(),
        worker_id="worker-a",
        batch_points=1,
        lease_seconds=180,
        max_attempts=1,
        base_delay_seconds=0,
        max_delay_seconds=0,
        sampler=sampler,
        signer=lambda value: value,
    )

    assert result.complete == 1
    shard = point_shard_path(tmp_path / "points", _task("mixed", "s1"))
    row = read_point_store(shard, "s1").to_pylist()[0]
    provenance = tmp_path / "points" / "provenance" / f"{row['source_item_sha256']}.json"
    assert provenance.is_file()
    assert json.loads(provenance.read_bytes())["schema"] == ("spectrajam-s1-mosaic-provenance-v1")
