"""Sparse, asset-major MPC COG materialization into immutable point shards."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import random
import socket
import time
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal, Protocol

import requests

from .contracts import CANONICAL_S2_BANDS, ContractError, sha256_file
from .ledger import AcquisitionLedger, Artifact, Task
from .pointstore import (
    S1_ORBIT_ASCENDING,
    S1_ORBIT_DESCENDING,
    read_point_store,
    write_point_store,
)
from .preprocessing import amplitude_to_mpc_s1, harmonize_mpc_s2, valid_scl
from .retry import RETRYABLE_STATUS_CODES, is_retryable, retry_after_seconds

Resampling = Literal["nearest", "bilinear"]
RasterKind = Literal["s2", "scl", "s1"]
Coordinate = tuple[float, float]
_COUNTRY_TARGET_CRS = {"RWA": "EPSG:32735", "ISR": "EPSG:32636"}
_TARGET_RESOLUTION_METERS = 10.0


class MaterializationError(RuntimeError):
    def __init__(
        self,
        message: str,
        task_ids: Sequence[str] = (),
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.task_ids = frozenset(task_ids)
        self.retry_after_seconds = retry_after


class TransientMaterializationError(MaterializationError):
    """A remote read failed after bounded, independently signed attempts."""


class TerminalMaterializationError(MaterializationError):
    """The persisted catalog or opened raster violates the data contract."""


class AssetSampler(Protocol):
    def __call__(
        self,
        signed_href: str,
        coordinates: Sequence[Coordinate],
        resampling: Resampling,
        kind: RasterKind,
        heartbeat: Callable[[], None],
        target_crs: str,
        target_resolution: float,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class GroupRecords:
    rows_by_task: Mapping[str, tuple[dict[str, object], ...]]
    query_sha256_by_task: Mapping[str, str]
    provenance_documents: Mapping[str, bytes]


@dataclass(frozen=True, slots=True)
class MaterializationRunResult:
    groups: int
    claimed: int
    complete: int
    insufficient_valid_observations: int
    no_source_observation: int
    requeued: int
    failed: int


@dataclass(slots=True)
class _Counters:
    groups: int = 0
    claimed: int = 0
    complete: int = 0
    insufficient_valid_observations: int = 0
    no_source_observation: int = 0
    requeued: int = 0
    failed: int = 0

    def result(self) -> MaterializationRunResult:
        return MaterializationRunResult(
            groups=self.groups,
            claimed=self.claimed,
            complete=self.complete,
            insufficient_valid_observations=self.insufficient_valid_observations,
            no_source_observation=self.no_source_observation,
            requeued=self.requeued,
            failed=self.failed,
        )


def default_worker_id() -> str:
    """Return a process-unique lease owner; never reuse a stale worker identity."""
    return f"{socket.gethostname()}-{os.getpid()}-{uuid.uuid4().hex[:10]}"


def materializer_contract_sha256() -> str:
    """Bind resumable output to the exact data-contract implementation."""
    import rasterio

    from . import pointstore, preprocessing, stac

    payload = {
        "schema": "spectrajam-materializer-contract-v1",
        "provider_profile": "mpc-v1.1",
        "target_crs": _COUNTRY_TARGET_CRS,
        "target_resolution_meters": _TARGET_RESOLUTION_METERS,
        "implementation": {
            "materialize": sha256_file(__file__),
            "pointstore": sha256_file(pointstore.__file__),
            "preprocessing": sha256_file(preprocessing.__file__),
            "stac": sha256_file(stac.__file__),
        },
        "runtime": {
            "numpy": importlib.metadata.version("numpy"),
            "planetary-computer": importlib.metadata.version("planetary-computer"),
            "pyarrow": importlib.metadata.version("pyarrow"),
            "rasterio": importlib.metadata.version("rasterio"),
            "gdal": rasterio.__gdal_version__,
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def preflight_materialization(
    *,
    tile_by_identity: Mapping[tuple[str, str, int], Any],
    catalog_root: str | Path,
    output_root: str | Path,
    profile: Any,
    modalities: Sequence[str] = ("s1", "s2"),
    verify_remote_assets: bool = False,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 30.0,
    remote_sampler: AssetSampler | None = None,
    remote_signer: Callable[[str], str] | None = None,
) -> str:
    """Validate every immutable input and output publication before claiming work."""
    from .stac import read_catalog_snapshot

    requested = tuple(sorted(set(modalities)))
    if not requested or any(value not in {"s1", "s2"} for value in requested):
        raise ContractError("materialization modalities must contain s1 and/or s2")
    inventory = []
    representatives: dict[str, tuple[Any, Any]] = {}
    for identity, tile in sorted(tile_by_identity.items()):
        for modality in requested:
            snapshot = read_catalog_snapshot(catalog_root, tile, modality, profile)
            representatives.setdefault(modality, (snapshot.items[0], tile))
            inventory.append(
                {
                    "country": identity[0],
                    "spatial_block": identity[1],
                    "year": identity[2],
                    "modality": modality,
                    "query_sha256": snapshot.sha256,
                }
            )

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    probe = output / f".spectrajam-write-probe-{os.getpid()}-{uuid.uuid4().hex}"
    try:
        with probe.open("xb") as stream:
            stream.write(b"spectrajam\n")
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        probe.unlink(missing_ok=True)

    if verify_remote_assets:
        sampler = remote_sampler or sample_cog_points
        signer = remote_signer or sign_mpc_href
        checks = {
            "s2": (("SCL", "nearest", "scl"), ("B04", "bilinear", "s2")),
            "s1": (("vv", "nearest", "s1"), ("vh", "nearest", "s1")),
        }
        for modality in requested:
            item, tile = representatives[modality]
            assets = _item_assets(item, tuple(check[0] for check in checks[modality]))
            coordinate = (
                (item.bbox[0] + item.bbox[2]) / 2,
                (item.bbox[1] + item.bbox[3]) / 2,
            )
            for asset_key, resampling, kind in checks[modality]:
                _sample_asset(
                    item_id=item.id,
                    asset_key=asset_key,
                    unsigned_href=assets[asset_key],
                    coordinates=[coordinate],
                    resampling=resampling,
                    kind=kind,
                    sampler=sampler,
                    signer=signer,
                    max_attempts=max_attempts,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    heartbeat=lambda: None,
                    task_ids=[],
                    target_crs=_COUNTRY_TARGET_CRS[tile.country],
                    target_resolution=_TARGET_RESOLUTION_METERS,
                )
    encoded = json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def sign_mpc_href(unsigned_href: str) -> str:
    """Sign one canonical unsigned MPC href immediately before it is opened."""
    try:
        import planetary_computer
    except ImportError as error:  # pragma: no cover - exercised without data extra
        raise RuntimeError(
            "planetary-computer is required for MPC reads; install spectrajam[data]"
        ) from error
    signed = planetary_computer.sign(unsigned_href)
    if not isinstance(signed, str) or not signed:
        raise TerminalMaterializationError("MPC signer did not return an asset href")
    return signed


def _compatibility_coordinates(tasks: Sequence[Task]) -> dict[str, Coordinate]:
    """Snap anchors to the common stackstac-style 10 m target-pixel centres."""
    from rasterio.warp import transform

    countries = {task.country for task in tasks}
    if len(countries) != 1:
        raise ContractError("compatibility-grid tasks must share one country")
    country = next(iter(countries))
    try:
        target_crs = _COUNTRY_TARGET_CRS[country]
    except KeyError as error:
        raise ContractError(f"unsupported compatibility-grid country: {country}") from error
    x_values, y_values = transform(
        "EPSG:4326",
        target_crs,
        [task.longitude for task in tasks],
        [task.latitude for task in tasks],
    )
    resolution = _TARGET_RESOLUTION_METERS
    centered_x = [
        math.floor(round(value, 3) / resolution) * resolution + resolution / 2 for value in x_values
    ]
    # North-up raster row lookup assigns an exact horizontal grid edge to the
    # pixel immediately south of it, hence ceil(y/resolution) - half a pixel.
    centered_y = [
        math.ceil(round(value, 3) / resolution) * resolution - resolution / 2 for value in y_values
    ]
    longitude, latitude = transform(target_crs, "EPSG:4326", centered_x, centered_y)
    return {
        task.task_id: (lon, lat) for task, lon, lat in zip(tasks, longitude, latitude, strict=True)
    }


def _read_pixels_by_block(
    dataset: Any,
    pixels: set[tuple[int, int]],
    *,
    preserve_nodata: bool = False,
    heartbeat: Callable[[], None] = lambda: None,
) -> dict[tuple[int, int], float]:
    """Read each touched internal raster block once and return requested pixels."""
    import numpy as np
    from rasterio.windows import Window

    if not pixels:
        return {}
    block_height, block_width = dataset.block_shapes[0]
    grouped: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for row, column in pixels:
        grouped[(row // block_height, column // block_width)].append((row, column))

    result: dict[tuple[int, int], float] = {}
    for block_row, block_column in sorted(grouped):
        heartbeat()
        row_offset = block_row * block_height
        column_offset = block_column * block_width
        height = min(block_height, dataset.height - row_offset)
        width = min(block_width, dataset.width - column_offset)
        block = dataset.read(
            1,
            window=Window(column_offset, row_offset, width, height),
            masked=not preserve_nodata,
        )
        for row, column in grouped[(block_row, block_column)]:
            value = block[row - row_offset, column - column_offset]
            result[(row, column)] = math.nan if np.ma.is_masked(value) else float(value)
    return result


def sample_cog_points(
    signed_href: str,
    coordinates: Sequence[Coordinate],
    resampling: Resampling,
    kind: RasterKind,
    heartbeat: Callable[[], None] = lambda: None,
    target_crs: str = "EPSG:4326",
    target_resolution: float = 1.0,
) -> Any:
    """Sample a COG through GDAL's common-grid warp, one touched VRT block once."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling as RasterioResampling
    from rasterio.transform import from_origin
    from rasterio.vrt import WarpedVRT
    from rasterio.warp import transform, transform_bounds

    if resampling not in {"nearest", "bilinear"}:
        raise ContractError(f"unsupported resampling mode: {resampling}")
    if not math.isfinite(target_resolution) or target_resolution <= 0:
        raise ContractError("target_resolution must be positive and finite")
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_CONNECTTIMEOUT=15,
        GDAL_HTTP_TIMEOUT=60,
        GDAL_HTTP_MAX_RETRY=0,
    ):
        with rasterio.open(signed_href) as dataset:
            if dataset.count != 1 or dataset.crs is None:
                raise TerminalMaterializationError(
                    "MPC asset must be a single-band georeferenced raster"
                )
            dtype = dataset.dtypes[0]
            if kind == "s2" and dtype != "uint16":
                raise TerminalMaterializationError(
                    f"MPC S2 spectral asset has unexpected dtype {dtype}"
                )
            if kind == "scl" and dtype != "uint8":
                raise TerminalMaterializationError(f"MPC SCL asset has unexpected dtype {dtype}")
            if kind == "s1" and dtype not in {"float32", "float64"}:
                raise TerminalMaterializationError(f"MPC S1 asset has unexpected dtype {dtype}")
            expected_nodata = -32768.0 if kind == "s1" else 0.0
            if dataset.nodata != expected_nodata:
                raise TerminalMaterializationError(
                    f"MPC {kind} asset has unexpected nodata {dataset.nodata}"
                )
            if tuple(dataset.scales) != (1.0,) or tuple(dataset.offsets) != (0.0,):
                raise TerminalMaterializationError(
                    f"MPC {kind} asset has unexpected scale/offset metadata"
                )
            if kind == "s1":
                tags = dataset.tags(1)
                if (
                    tags.get("SARPixelContent", "").lower() != "intensity"
                    or tags.get("Scale", "").lower() != "linear"
                ):
                    raise TerminalMaterializationError(
                        "MPC S1 asset is not tagged as linear intensity"
                    )

            if not coordinates:
                return np.empty(0, dtype=np.float64)
            left, bottom, right, top = transform_bounds(
                dataset.crs,
                target_crs,
                *dataset.bounds,
                densify_pts=21,
            )
            resolution = float(target_resolution)
            left = math.floor(left / resolution) * resolution
            bottom = math.floor(bottom / resolution) * resolution
            right = math.ceil(right / resolution) * resolution
            top = math.ceil(top / resolution) * resolution
            width = max(1, int(round((right - left) / resolution)))
            height = max(1, int(round((top - bottom) / resolution)))
            warp_resampling = (
                RasterioResampling.nearest
                if resampling == "nearest"
                else RasterioResampling.bilinear
            )
            with WarpedVRT(
                dataset,
                crs=target_crs,
                transform=from_origin(left, top, resolution, resolution),
                width=width,
                height=height,
                src_nodata=expected_nodata,
                nodata=expected_nodata,
                resampling=warp_resampling,
            ) as warped:
                longitude, latitude = zip(*coordinates, strict=True)
                x_values, y_values = transform(
                    "EPSG:4326",
                    target_crs,
                    list(longitude),
                    list(latitude),
                )
                inverse = ~warped.transform
                requested: list[tuple[int, int] | None] = []
                pixels: set[tuple[int, int]] = set()
                for x, y in zip(x_values, y_values, strict=True):
                    column, row = inverse * (x, y)
                    pixel = (math.floor(row), math.floor(column))
                    if 0 <= pixel[0] < warped.height and 0 <= pixel[1] < warped.width:
                        requested.append(pixel)
                        pixels.add(pixel)
                    else:
                        requested.append(None)
                values = _read_pixels_by_block(
                    warped,
                    pixels,
                    preserve_nodata=kind == "scl",
                    heartbeat=heartbeat,
                )
                return np.asarray(
                    [math.nan if pixel is None else values[pixel] for pixel in requested],
                    dtype=np.float64,
                )


def _http_status(error: BaseException) -> int | None:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code
    message = str(error).lower()
    for status in (*sorted(RETRYABLE_STATUS_CODES), 401, 403, 404):
        if f"{status}" in message and ("http" in message or "response" in message):
            return status
    return None


def _transient_asset_error(error: BaseException) -> bool:
    if isinstance(error, (ContractError, TerminalMaterializationError)):
        return False
    if is_retryable(error) or isinstance(error, (TimeoutError, ConnectionError, EOFError)):
        return True
    status = _http_status(error)
    if status is not None:
        return status in RETRYABLE_STATUS_CODES | {401, 403}
    # Rasterio/GDAL wraps connection resets, expired range reads, and truncated
    # COG responses in RasterioIOError/OSError without preserving requests types.
    return error.__class__.__name__ in {
        "RasterioIOError",
        "CPLE_OpenFailedError",
        "CPLE_AppDefinedError",
    } or isinstance(error, OSError)


def _retry_delay(
    attempt: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    error: BaseException,
) -> float:
    cap = min(max_delay_seconds, base_delay_seconds * (2**attempt))
    delay = random.uniform(0.0, cap) if cap > 0 else 0.0
    retry_after = retry_after_seconds(error)
    return max(delay, retry_after or 0.0)


def _heartbeat_sleep(
    delay_seconds: float,
    heartbeat: Callable[[], None],
    chunk_seconds: float = 30.0,
) -> None:
    deadline = time.monotonic() + max(0.0, delay_seconds)
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return
        time.sleep(min(chunk_seconds, remaining))
        heartbeat()


def _sample_asset(
    *,
    item_id: str,
    asset_key: str,
    unsigned_href: str,
    coordinates: Sequence[Coordinate],
    resampling: Resampling,
    kind: RasterKind,
    sampler: AssetSampler,
    signer: Callable[[str], str],
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    heartbeat: Callable[[], None],
    task_ids: Sequence[str],
    target_crs: str,
    target_resolution: float,
) -> Any:
    for attempt in range(max_attempts):
        try:
            heartbeat()
            # Always restart from the canonical unsigned href. Reusing a signed
            # href can preserve an expired SAS token across retries.
            signed_href = signer(unsigned_href)
            return sampler(
                signed_href,
                coordinates,
                resampling,
                kind,
                heartbeat,
                target_crs,
                target_resolution,
            )
        except Exception as error:
            transient = _transient_asset_error(error)
            if not transient:
                if isinstance(error, (ContractError, TerminalMaterializationError)):
                    if isinstance(error, TerminalMaterializationError):
                        raise TerminalMaterializationError(str(error), task_ids) from None
                    raise TerminalMaterializationError(
                        f"terminal raster contract failure for {item_id}/{asset_key}",
                        task_ids,
                    ) from None
                raise TerminalMaterializationError(
                    f"terminal raster read failure for {item_id}/{asset_key}",
                    task_ids,
                ) from None
            if attempt + 1 >= max_attempts:
                raise TransientMaterializationError(
                    f"transient raster read exhausted for {item_id}/{asset_key}",
                    task_ids,
                    retry_after_seconds(error),
                ) from None
            _heartbeat_sleep(
                _retry_delay(attempt, base_delay_seconds, max_delay_seconds, error),
                heartbeat,
            )
    raise AssertionError("unreachable")


def _inside_bbox(task: Task, bbox: Sequence[float]) -> bool:
    if len(bbox) != 4:
        raise ContractError("catalog item bbox must contain four coordinates")
    return bbox[0] <= task.longitude <= bbox[2] and bbox[1] <= task.latitude <= bbox[3]


def _epoch_fields(observed: date) -> tuple[int, int]:
    epoch_day = (observed - date(1970, 1, 1)).days
    return epoch_day, observed.timetuple().tm_yday


def _base_row(
    task: Task,
    item: Any,
    query_sha256: str,
) -> dict[str, object]:
    observed = item.acquired.date()
    epoch_day, day_of_year = _epoch_fields(observed)
    return {
        "sample_id": task.sample_id,
        "country": task.country,
        "year": task.year,
        "epoch_day": epoch_day,
        "day_of_year": day_of_year,
        "source_item_id": item.id,
        "source_item_sha256": item.raw_item_sha256,
        "catalog_query_sha256": query_sha256,
    }


def _catalog_items_for_tasks(
    tasks: Sequence[Task], snapshots_by_block: Mapping[str, Any]
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Task]],
    dict[str, dict[str, str]],
    dict[str, str],
]:
    items: dict[str, Any] = {}
    applicable: dict[str, dict[str, Task]] = defaultdict(dict)
    query_by_item_task: dict[str, dict[str, str]] = defaultdict(dict)
    query_by_task: dict[str, str] = {}
    for task in tasks:
        try:
            snapshot = snapshots_by_block[task.spatial_block]
        except KeyError as error:
            raise ContractError(
                f"no validated catalog for spatial block {task.spatial_block}"
            ) from error
        if snapshot.modality != task.modality or snapshot.tile.year != task.year:
            raise ContractError("catalog snapshot does not match the claimed task")
        query_by_task[task.task_id] = snapshot.sha256
        for item in snapshot.items:
            if not _inside_bbox(task, item.bbox):
                continue
            existing = items.setdefault(item.raw_item_sha256, item)
            if existing.id != item.id or existing.acquired != item.acquired:
                raise ContractError("raw STAC item digest has conflicting catalog summaries")
            applicable[item.raw_item_sha256][task.task_id] = task
            query_by_item_task[item.raw_item_sha256][task.task_id] = snapshot.sha256
    return items, applicable, query_by_item_task, query_by_task


def _item_assets(item: Any, required: Sequence[str]) -> Mapping[str, str]:
    assets = item.assets
    if not isinstance(assets, Mapping):
        raise ContractError(f"catalog item {item.id} assets must be a mapping")
    missing = set(required) - set(assets)
    if missing:
        raise ContractError(f"catalog item {item.id} is missing assets {sorted(missing)}")
    if any(not isinstance(assets[key], str) or not assets[key] for key in required):
        raise ContractError(f"catalog item {item.id} contains an invalid asset href")
    return assets


def _s2_records(
    tasks: Sequence[Task],
    snapshots_by_block: Mapping[str, Any],
    *,
    sampler: AssetSampler,
    signer: Callable[[str], str],
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    heartbeat: Callable[[], None],
) -> GroupRecords:
    import numpy as np

    items, applicable, query_for, query_by_task = _catalog_items_for_tasks(
        tasks, snapshots_by_block
    )
    target_coordinates = _compatibility_coordinates(tasks)
    target_crs = _COUNTRY_TARGET_CRS[tasks[0].country]
    by_day: dict[date, list[Any]] = defaultdict(list)
    for digest, item in items.items():
        if digest in applicable:
            by_day[item.acquired.date()].append(item)

    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    for observed, day_items in sorted(by_day.items()):
        ordered = sorted(day_items, key=lambda value: (value.acquired, value.id))
        invalid_fallback: dict[str, tuple[Any, int, list[int]]] = {}
        selected: dict[str, tuple[Any, int, list[int]]] = {}
        for item in ordered:
            digest = item.raw_item_sha256
            candidates = [
                task for task in applicable[digest].values() if task.task_id not in selected
            ]
            if not candidates:
                continue
            assets = _item_assets(item, (*CANONICAL_S2_BANDS, "SCL"))
            values = _sample_asset(
                item_id=item.id,
                asset_key="SCL",
                unsigned_href=assets["SCL"],
                coordinates=[target_coordinates[task.task_id] for task in candidates],
                resampling="nearest",
                kind="scl",
                sampler=sampler,
                signer=signer,
                max_attempts=max_attempts,
                base_delay_seconds=base_delay_seconds,
                max_delay_seconds=max_delay_seconds,
                heartbeat=heartbeat,
                task_ids=[task.task_id for task in candidates],
                target_crs=target_crs,
                target_resolution=_TARGET_RESOLUTION_METERS,
            )
            if len(values) != len(candidates):
                raise TerminalMaterializationError("SCL sampler returned the wrong value count")
            validity = valid_scl(values)
            valid_candidates: list[Task] = []
            scl_by_task: dict[str, int] = {}
            for task, value, is_valid in zip(candidates, values, validity, strict=True):
                if not np.isfinite(value) or not 0 <= value <= 11:
                    continue
                scl = int(value)
                if bool(is_valid):
                    valid_candidates.append(task)
                    scl_by_task[task.task_id] = scl
                else:
                    invalid_fallback.setdefault(task.task_id, (item, scl, [0] * 10))

            # Pinned TESSERA fixes tile selection from SCL alone, then applies
            # that same item choice to every spectral band.
            if not valid_candidates:
                continue
            coordinates = [target_coordinates[task.task_id] for task in valid_candidates]
            per_task: dict[str, list[float]] = {task.task_id: [] for task in valid_candidates}
            for band in CANONICAL_S2_BANDS:
                values = _sample_asset(
                    item_id=item.id,
                    asset_key=band,
                    unsigned_href=assets[band],
                    coordinates=coordinates,
                    resampling="bilinear",
                    kind="s2",
                    sampler=sampler,
                    signer=signer,
                    max_attempts=max_attempts,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    heartbeat=heartbeat,
                    task_ids=[task.task_id for task in valid_candidates],
                    target_crs=target_crs,
                    target_resolution=_TARGET_RESOLUTION_METERS,
                )
                if len(values) != len(valid_candidates):
                    raise TerminalMaterializationError(
                        f"{band} sampler returned the wrong value count"
                    )
                for task, value in zip(valid_candidates, values, strict=True):
                    per_task[task.task_id].append(float(value))
            for task in valid_candidates:
                raw_bands = per_task[task.task_id]
                selected[task.task_id] = (
                    item,
                    scl_by_task[task.task_id],
                    harmonize_mpc_s2(raw_bands, observed).tolist(),
                )

        choices = invalid_fallback | selected
        for task in tasks:
            choice = choices.get(task.task_id)
            if choice is None:
                continue
            item, scl, bands = choice
            digest = item.raw_item_sha256
            is_valid = bool(valid_scl([scl])[0])
            rows[task.task_id].append(
                _base_row(task, item, query_for[digest][task.task_id])
                | {"bands": bands, "scl": scl, "valid": is_valid}
            )
    return GroupRecords(
        rows_by_task={task.task_id: tuple(rows[task.task_id]) for task in tasks},
        query_sha256_by_task=query_by_task,
        provenance_documents={},
    )


def _s1_orbit(item: Any) -> int:
    orbit = str(item.properties.get("sat:orbit_state", "")).lower()
    if orbit == "ascending":
        return S1_ORBIT_ASCENDING
    if orbit == "descending":
        return S1_ORBIT_DESCENDING
    raise ContractError(f"catalog item {item.id} has unsupported orbit state {orbit!r}")


def _s1_records(
    tasks: Sequence[Task],
    snapshots_by_block: Mapping[str, Any],
    *,
    sampler: AssetSampler,
    signer: Callable[[str], str],
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    heartbeat: Callable[[], None],
) -> GroupRecords:
    import numpy as np

    items, applicable, _query_for, query_by_task = _catalog_items_for_tasks(
        tasks, snapshots_by_block
    )
    target_coordinates = _compatibility_coordinates(tasks)
    target_crs = _COUNTRY_TARGET_CRS[tasks[0].country]
    grouped: dict[tuple[date, int], list[Any]] = defaultdict(list)
    for digest, item in items.items():
        if digest in applicable:
            grouped[(item.acquired.date(), _s1_orbit(item))].append(item)

    rows: dict[str, list[dict[str, object]]] = defaultdict(list)
    provenance_documents: dict[str, bytes] = {}
    for (_observed, orbit), group_items in sorted(grouped.items()):
        ordered = sorted(group_items, key=lambda value: (value.acquired, value.id))
        fallback: dict[str, dict[str, tuple[Any, int]]] = {
            "vv": {},
            "vh": {},
        }
        selected: dict[str, dict[str, tuple[Any, int]]] = {
            "vv": {},
            "vh": {},
        }
        for item in ordered:
            digest = item.raw_item_sha256
            assets = _item_assets(item, ("vv", "vh"))
            for asset_key in ("vv", "vh"):
                candidates = [
                    task
                    for task in applicable[digest].values()
                    if task.task_id not in selected[asset_key]
                ]
                if not candidates:
                    continue
                values = _sample_asset(
                    item_id=item.id,
                    asset_key=asset_key,
                    unsigned_href=assets[asset_key],
                    coordinates=[target_coordinates[task.task_id] for task in candidates],
                    # Pinned TESSERA leaves stackstac's S1 resampling at its
                    # nearest-neighbor default.
                    resampling="nearest",
                    kind="s1",
                    sampler=sampler,
                    signer=signer,
                    max_attempts=max_attempts,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    heartbeat=heartbeat,
                    task_ids=[task.task_id for task in candidates],
                    target_crs=target_crs,
                    target_resolution=_TARGET_RESOLUTION_METERS,
                )
                if len(values) != len(candidates):
                    raise TerminalMaterializationError(
                        f"{asset_key} sampler returned the wrong value count"
                    )
                # stackstac promotes the raster before the logarithm. Pin
                # float64 here so the final integer truncation is deterministic.
                raw_values = np.asarray(values, dtype=np.float64)
                converted = amplitude_to_mpc_s1(raw_values)
                for task, source, band in zip(
                    candidates, raw_values, converted.tolist(), strict=True
                ):
                    if not np.isfinite(source):
                        continue
                    choice = (item, int(band))
                    fallback[asset_key].setdefault(task.task_id, choice)
                    if band != 0:
                        selected[asset_key][task.task_id] = choice

        choices = {
            asset_key: fallback[asset_key] | selected[asset_key] for asset_key in ("vv", "vh")
        }
        for task in tasks:
            vv = choices["vv"].get(task.task_id)
            vh = choices["vh"].get(task.task_id)
            if vv is None and vh is None:
                continue
            bands = [vv[1] if vv is not None else 0, vh[1] if vh is not None else 0]
            components = {
                "vv": vv[0] if vv is not None else None,
                "vh": vh[0] if vh is not None else None,
            }
            component_items = [item for item in components.values() if item is not None]
            if (
                len(component_items) == 2
                and component_items[0].raw_item_sha256 == component_items[1].raw_item_sha256
            ):
                source_item_id = component_items[0].id
                source_item_sha256 = component_items[0].raw_item_sha256
            else:
                identities = {
                    key: (
                        None
                        if item is None
                        else {"id": item.id, "raw_item_sha256": item.raw_item_sha256}
                    )
                    for key, item in components.items()
                }
                source_item_id = (
                    f"mosaic:vv={identities['vv']['id'] if identities['vv'] else 'none'};"
                    f"vh={identities['vh']['id'] if identities['vh'] else 'none'}"
                )
                document = json.dumps(
                    {
                        "schema": "spectrajam-s1-mosaic-provenance-v1",
                        "components": identities,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
                source_item_sha256 = hashlib.sha256(document).hexdigest()
                existing = provenance_documents.setdefault(source_item_sha256, document)
                if existing != document:
                    raise ContractError("S1 mosaic provenance digest collision")
            base_item = component_items[0]
            row = _base_row(task, base_item, query_by_task[task.task_id])
            row["source_item_id"] = source_item_id
            row["source_item_sha256"] = source_item_sha256
            rows[task.task_id].append(
                row
                | {
                    "bands": bands,
                    "orbit": orbit,
                    "valid": any(value != 0 for value in bands),
                }
            )
    return GroupRecords(
        rows_by_task={task.task_id: tuple(rows[task.task_id]) for task in tasks},
        query_sha256_by_task=query_by_task,
        provenance_documents=provenance_documents,
    )


def materialize_group_records(
    tasks: Sequence[Task],
    snapshots_by_block: Mapping[str, Any],
    *,
    sampler: AssetSampler = sample_cog_points,
    signer: Callable[[str], str] = sign_mpc_href,
    max_attempts: int = 3,
    base_delay_seconds: float = 0.5,
    max_delay_seconds: float = 30.0,
    heartbeat: Callable[[], None] = lambda: None,
) -> GroupRecords:
    """Read one country/year/modality group asset-major and return sparse rows."""
    if not tasks:
        raise ContractError("cannot materialize an empty task group")
    identity = {(task.country, task.year, task.modality) for task in tasks}
    if len(identity) != 1:
        raise ContractError("materialization tasks must share country, year, and modality")
    if max_attempts < 1:
        raise ContractError("max_attempts must be positive")
    modality = tasks[0].modality
    if modality == "s2":
        return _s2_records(
            tasks,
            snapshots_by_block,
            sampler=sampler,
            signer=signer,
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            heartbeat=heartbeat,
        )
    if modality == "s1":
        return _s1_records(
            tasks,
            snapshots_by_block,
            sampler=sampler,
            signer=signer,
            max_attempts=max_attempts,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            heartbeat=heartbeat,
        )
    raise ContractError(f"unsupported materialization modality: {modality}")


def point_shard_path(root: str | Path, task: Task) -> Path:
    return (
        Path(root)
        / task.country.lower()
        / str(task.year)
        / task.modality
        / f"{task.sample_id}.parquet"
    )


def _publish_provenance_document(root: str | Path, digest: str, content: bytes) -> Path:
    if hashlib.sha256(content).hexdigest() != digest:
        raise ContractError("provenance document digest does not match its content")
    destination = Path(root) / "provenance" / f"{digest}.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.part.{os.getpid()}.{uuid.uuid4().hex}")
    try:
        with temporary.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            if destination.read_bytes() != content:
                raise ContractError(
                    f"refusing to replace immutable provenance: {destination}"
                ) from error
        directory = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _artifact_metadata(
    task: Task,
    outcome: str,
    rows: Sequence[Mapping[str, object]],
    query_sha256: str,
    provenance_documents: Mapping[str, bytes],
    implementation_sha256: str,
) -> dict[str, object]:
    valid_count = sum(bool(row["valid"]) for row in rows)
    return {
        "schema": "spectrajam-materialization-v1",
        "outcome": outcome,
        "provider_profile": "mpc-v1.1",
        "materializer_contract_sha256": implementation_sha256,
        "sample_id": task.sample_id,
        "modality": task.modality,
        "source_observation_count": len(rows),
        "valid_observation_count": valid_count,
        "catalog_query_sha256": query_sha256,
        "source_item_sha256s": sorted({str(row["source_item_sha256"]) for row in rows}),
        "composite_provenance_sha256s": sorted(
            {
                str(row["source_item_sha256"])
                for row in rows
                if str(row["source_item_sha256"]) in provenance_documents
            }
        ),
    }


def _fail_tasks(
    ledger: AcquisitionLedger,
    tasks: Sequence[Task],
    error: BaseException,
    worker_id: str,
    base_delay_seconds: float,
    max_delay_seconds: float,
) -> tuple[int, int]:
    retryable = isinstance(error, TransientMaterializationError)
    requeued = failed = 0
    for task in tasks:
        status = ledger.fail(
            task,
            error,
            worker_id,
            retryable=retryable,
            base_delay_seconds=base_delay_seconds,
            max_delay_seconds=max_delay_seconds,
            retry_after_seconds=(
                error.retry_after_seconds if isinstance(error, MaterializationError) else None
            ),
        )
        if status == "retry":
            requeued += 1
        else:
            failed += 1
    return requeued, failed


def run_materialization(
    *,
    ledger: AcquisitionLedger,
    tile_by_identity: Mapping[tuple[str, str, int], Any],
    catalog_root: str | Path,
    output_root: str | Path,
    profile: Any,
    worker_id: str,
    batch_points: int,
    lease_seconds: int,
    max_attempts: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    max_groups: int | None = None,
    sampler: AssetSampler = sample_cog_points,
    signer: Callable[[str], str] = sign_mpc_href,
    progress: Callable[[dict[str, object]], None] | None = None,
    implementation_sha256: str | None = None,
) -> MaterializationRunResult:
    """Resolve all due ledger tasks, waiting for this process's bounded retries."""
    from .stac import read_catalog_snapshot

    if batch_points < 1:
        raise ContractError("batch_points must be positive")
    if lease_seconds < 180:
        raise ContractError(
            "materialization lease_seconds must be at least 180 for bounded COG reads"
        )
    implementation_sha256 = implementation_sha256 or materializer_contract_sha256()
    counters = _Counters()
    while max_groups is None or counters.groups < max_groups:
        tasks = ledger.claim_group(
            worker_id,
            limit=batch_points,
            lease_seconds=lease_seconds,
        )
        if not tasks:
            wait_seconds = ledger.seconds_until_due()
            if wait_seconds is None:
                break
            time.sleep(min(max(wait_seconds, 0.01), 5.0))
            continue

        counters.groups += 1
        counters.claimed += len(tasks)
        snapshots: dict[str, Any] = {}
        # These were validated before the first claim. Any change now is an
        # operator/infrastructure error, not a scientific terminal outcome.
        for block in sorted({task.spatial_block for task in tasks}):
            key = (tasks[0].country, block, tasks[0].year)
            if key not in tile_by_identity:
                raise ContractError(f"manifest work tile is missing: {key}")
            snapshots[block] = read_catalog_snapshot(
                catalog_root, tile_by_identity[key], tasks[0].modality, profile
            )

        pending = list(tasks)
        records: GroupRecords | None = None
        while pending:
            last_heartbeat = 0.0
            leased = tuple(pending)

            def heartbeat(leased_tasks: tuple[Task, ...] = leased) -> None:
                nonlocal last_heartbeat
                now = time.monotonic()
                if now - last_heartbeat >= max(1.0, lease_seconds / 3):
                    ledger.renew_leases(leased_tasks, worker_id, lease_seconds)
                    last_heartbeat = now

            try:
                records = materialize_group_records(
                    pending,
                    snapshots,
                    sampler=sampler,
                    signer=signer,
                    max_attempts=max_attempts,
                    base_delay_seconds=base_delay_seconds,
                    max_delay_seconds=max_delay_seconds,
                    heartbeat=heartbeat,
                )
                break
            except (TransientMaterializationError, TerminalMaterializationError) as error:
                affected_ids = error.task_ids or frozenset(task.task_id for task in pending)
                affected = [task for task in pending if task.task_id in affected_ids]
                if not affected:
                    raise RuntimeError(
                        "materialization error scope does not match leased tasks"
                    ) from error
                requeued, failed = _fail_tasks(
                    ledger,
                    affected,
                    error,
                    worker_id,
                    base_delay_seconds,
                    max_delay_seconds,
                )
                counters.requeued += requeued
                counters.failed += failed
                pending = [task for task in pending if task.task_id not in affected_ids]
            except ContractError:
                terminal = TerminalMaterializationError(
                    "materialized source data violates its contract",
                    [task.task_id for task in pending],
                )
                requeued, failed = _fail_tasks(
                    ledger,
                    pending,
                    terminal,
                    worker_id,
                    base_delay_seconds,
                    max_delay_seconds,
                )
                counters.requeued += requeued
                counters.failed += failed
                pending = []

        if records is None:
            if progress:
                progress(
                    {
                        "event": "materialization-group",
                        "country": tasks[0].country,
                        "year": tasks[0].year,
                        "modality": tasks[0].modality,
                        "claimed": len(tasks),
                        "summary": ledger.summary(),
                        "outcomes": ledger.outcomes(),
                    }
                )
            continue

        publication_heartbeat = 0.0
        for digest, content in sorted(records.provenance_documents.items()):
            now = time.monotonic()
            if now - publication_heartbeat >= max(1.0, lease_seconds / 3):
                ledger.renew_leases(pending, worker_id, lease_seconds)
                publication_heartbeat = now
            _publish_provenance_document(output_root, digest, content)
        for index, task in enumerate(pending):
            now = time.monotonic()
            if now - publication_heartbeat >= max(1.0, lease_seconds / 3):
                ledger.renew_leases(pending[index:], worker_id, lease_seconds)
                publication_heartbeat = now
            rows = records.rows_by_task[task.task_id]
            query_sha256 = records.query_sha256_by_task[task.task_id]
            if not rows:
                metadata = {
                    "schema": "spectrajam-materialization-v1",
                    "outcome": "no_source_observation",
                    "provider_profile": "mpc-v1.1",
                    "materializer_contract_sha256": implementation_sha256,
                    "sample_id": task.sample_id,
                    "modality": task.modality,
                    "source_observation_count": 0,
                    "valid_observation_count": 0,
                    "catalog_query_sha256": query_sha256,
                }
                ledger.resolve_without_artifact(
                    task,
                    outcome="no_source_observation",
                    metadata=metadata,
                    worker_id=worker_id,
                )
                counters.no_source_observation += 1
                continue

            valid_count = sum(bool(row["valid"]) for row in rows)
            outcome = "complete" if valid_count > 0 else "insufficient_valid_observations"
            written = write_point_store(point_shard_path(output_root, task), task.modality, rows)
            read_point_store(written.path, task.modality, written.sha256)
            ledger.succeed(
                task,
                Artifact(
                    uri=str(written.path),
                    sha256=written.sha256,
                    observation_count=written.row_count,
                    metadata=_artifact_metadata(
                        task,
                        outcome,
                        rows,
                        query_sha256,
                        records.provenance_documents,
                        implementation_sha256,
                    ),
                ),
                worker_id,
            )
            if outcome == "complete":
                counters.complete += 1
            else:
                counters.insufficient_valid_observations += 1

        if progress:
            progress(
                {
                    "event": "materialization-group",
                    "country": tasks[0].country,
                    "year": tasks[0].year,
                    "modality": tasks[0].modality,
                    "claimed": len(tasks),
                    "summary": ledger.summary(),
                    "outcomes": ledger.outcomes(),
                }
            )
    return counters.result()


def json_progress(payload: dict[str, object]) -> None:
    print(json.dumps(payload, sort_keys=True), flush=True)
