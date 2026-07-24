from __future__ import annotations

import logging
import random
import re
import time
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from threading import Lock

import numpy as np

from .catalog import (
    S1_ASSETS,
    S2_ASSETS,
    detached_item_dicts,
    signed_item_dicts,
    unsigned_items,
)
from .geometry import RasterWindow, projected_bounds_to_wgs84

S2_BANDS = S2_ASSETS[:-1]
INVALID_SCL = np.array([0, 1, 2, 3, 8, 9], dtype=np.int16)
EPOCH = date(1970, 1, 1)
_URL_QUERY = re.compile(r"(https?://[^?\s'\"<>]+)\?[^\s'\"<>]+")
_MPC_SIGNING_LOCK = Lock()
_GROUP_CACHE_SCHEMA_VERSION = 1
_CACHE_MISS = object()


def _sanitized_error(error: Exception, limit: int = 2000) -> str:
    message = _URL_QUERY.sub(r"\1?<redacted>", str(error))
    return message if len(message) <= limit else f"{message[:limit]}…"


class _SuppressUnsupportedSharingWarning(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "warp options does not support option SHARING" not in record.getMessage()


logging.getLogger("rasterio._env").addFilter(_SuppressUnsupportedSharingWarning())
LOGGER = logging.getLogger(__name__)


def epoch_day(value: date) -> int:
    return (value - EPOCH).days


def _positive_bbox_overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    if first[0] > first[2] or second[0] > second[2]:
        return True
    return max(first[0], second[0]) < min(first[2], second[2]) and max(first[1], second[1]) < min(
        first[3], second[3]
    )


def _proj_epsg(value: object) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.upper().startswith("EPSG:"):
        try:
            return int(value.split(":", 1)[1])
        except ValueError:
            return None
    return None


def _items_intersecting_grid(
    items: Sequence[object],
    asset_names: Sequence[str],
    grid: RasterWindow,
    bounds_wgs84: tuple[float, float, float, float],
) -> list[object]:
    result: list[object] = []
    for item in items:
        projected_overlaps: list[bool] = []
        all_assets_comparable = True
        for name in asset_names:
            asset = getattr(item, "assets", {}).get(name)
            if asset is None:
                all_assets_comparable = False
                break
            extra = getattr(asset, "extra_fields", {})
            projected_bbox = extra.get("proj:bbox")
            code = extra.get("proj:code", getattr(item, "properties", {}).get("proj:code"))
            if code is None:
                code = getattr(item, "properties", {}).get("proj:epsg")
            if (
                not isinstance(projected_bbox, (list, tuple))
                or len(projected_bbox) != 4
                or _proj_epsg(code) != grid.epsg
            ):
                all_assets_comparable = False
                break
            asset_bounds = tuple(float(value) for value in projected_bbox)
            projected_overlaps.append(_positive_bbox_overlap(asset_bounds, grid.bounds))
        if all_assets_comparable and projected_overlaps:
            if any(projected_overlaps):
                result.append(item)
            continue

        bbox = getattr(item, "bbox", None)
        if bbox is None:
            result.append(item)
            continue
        item_bounds = tuple(float(value) for value in bbox)
        if _positive_bbox_overlap(item_bounds, bounds_wgs84):
            result.append(item)
    return result


class NoSpatialCoverageError(RuntimeError):
    """The requested assets do not cover the tight task raster window."""


def _load_group_cache(
    path: Path,
    pixel_count: int,
    band_count: int,
    dtype: np.dtype,
) -> tuple[np.ndarray, np.ndarray] | None | object:
    if not path.is_file():
        return _CACHE_MISS
    try:
        with np.load(path, allow_pickle=False) as payload:
            schema_version = int(payload["schema_version"])
            present = bool(payload["present"])
            values = payload["values"]
            valid = payload["valid"]
    except Exception as error:
        raise RuntimeError(f"incremental group cache is unreadable: {path}") from error
    if schema_version != _GROUP_CACHE_SCHEMA_VERSION:
        raise RuntimeError(f"incremental group cache has the wrong schema: {path}")
    if not present:
        if values.size or valid.size:
            raise RuntimeError(f"empty incremental group cache contains data: {path}")
        return None
    expected_values = (pixel_count, band_count)
    if values.shape != expected_values or values.dtype != dtype:
        raise RuntimeError(
            f"incremental group cache values are invalid: {path} "
            f"({values.shape}, {values.dtype}) != ({expected_values}, {dtype})"
        )
    if valid.shape != (pixel_count,) or valid.dtype != np.bool_:
        raise RuntimeError(f"incremental group cache validity is invalid: {path}")
    return values, valid


def _write_group_cache(
    path: Path,
    result: tuple[np.ndarray, np.ndarray] | None,
    band_count: int,
    dtype: np.dtype,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    if result is None:
        values = np.empty((0, band_count), dtype=dtype)
        valid = np.empty(0, dtype=bool)
    else:
        values, valid = result
    with temporary.open("wb") as stream:
        np.savez(
            stream,
            schema_version=np.asarray(_GROUP_CACHE_SCHEMA_VERSION, dtype=np.uint8),
            present=np.asarray(result is not None, dtype=bool),
            values=values,
            valid=valid,
        )
    temporary.replace(path)


def harmonize_s2_mpc(values: np.ndarray) -> np.ndarray:
    """Apply the post-2022 MPC BOA offset and upstream uint16 storage cast."""
    result = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(result)
    result[finite & (result >= 1000.0)] -= 1000.0
    result[~finite] = 0.0
    return np.clip(result, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def select_s2_daily_mosaic(scl: np.ndarray, bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pick the first valid-SCL item per pixel and all bands from that item."""
    scl = np.asarray(scl)
    bands = np.asarray(bands)
    if scl.ndim != 2 or bands.ndim != 3:
        raise ValueError("S2 mosaic expects SCL [item,pixel] and bands [item,band,pixel]")
    if bands.shape[0] != scl.shape[0] or bands.shape[2] != scl.shape[1]:
        raise ValueError("S2 SCL and band shapes do not align")
    scl_classes = np.nan_to_num(scl, nan=0).astype(np.int16, copy=False)
    valid_by_item = np.isfinite(scl) & ~np.isin(scl_classes, INVALID_SCL)
    valid = valid_by_item.any(axis=0)
    selected_item = np.argmax(valid_by_item, axis=0)
    pixel_index = np.arange(scl.shape[1])
    selected = bands.transpose(0, 2, 1)[selected_item, pixel_index]
    values = harmonize_s2_mpc(selected)
    values[~valid] = 0
    return values, valid


def scale_s1_amplitude(values: np.ndarray) -> np.ndarray:
    amplitude = np.asarray(values, dtype=np.float32)
    result = np.zeros(amplitude.shape, dtype=np.int16)
    valid = np.isfinite(amplitude) & (amplitude > 0)
    if np.any(valid):
        scaled = (20.0 * np.log10(amplitude[valid]) + 50.0) * 200.0
        result[valid] = np.clip(scaled, 0, 32767).astype(np.int16)
    return result


def select_s1_daily_mosaic(bands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert amplitudes, then mosaic each polarization by first nonzero item."""
    bands = np.asarray(bands)
    if bands.ndim != 3 or bands.shape[1] != 2:
        raise ValueError("S1 mosaic expects [item,2,pixel]")
    scaled = scale_s1_amplitude(bands)
    valid_by_item_band = scaled != 0
    selected_item = np.argmax(valid_by_item_band, axis=0)
    band_index = np.arange(2)[:, None]
    pixel_index = np.arange(bands.shape[2])[None, :]
    selected = scaled[selected_item, band_index, pixel_index].T
    valid = np.any(selected != 0, axis=1)
    selected[~valid] = 0
    return selected, valid


@dataclass(frozen=True, slots=True)
class PixelTimelines:
    pixel_ids: tuple[str, ...]
    s2_values: np.ndarray
    s2_valid: np.ndarray
    s2_days: np.ndarray
    s1a_values: np.ndarray
    s1a_valid: np.ndarray
    s1a_days: np.ndarray
    s1d_values: np.ndarray
    s1d_valid: np.ndarray
    s1d_days: np.ndarray

    def validate(self) -> None:
        pixels = len(self.pixel_ids)
        if len(set(self.pixel_ids)) != pixels:
            raise ValueError("timeline pixel IDs must be unique")
        specs = (
            (self.s2_values, self.s2_valid, self.s2_days, 10, "s2"),
            (self.s1a_values, self.s1a_valid, self.s1a_days, 2, "s1a"),
            (self.s1d_values, self.s1d_valid, self.s1d_days, 2, "s1d"),
        )
        for values, valid, days, bands, name in specs:
            if values.ndim != 3 or values.shape[1:] != (pixels, bands):
                raise ValueError(f"{name} values must have shape [time,pixel,{bands}]")
            if valid.shape != values.shape[:2] or days.shape != (values.shape[0],):
                raise ValueError(f"{name} validity/day arrays do not align")
            if valid.dtype != np.bool_:
                raise ValueError(f"{name} validity must be boolean")
            if len(days) and np.any(days[:-1] > days[1:]):
                raise ValueError(f"{name} observations must be chronological")

    def save(self, path: Path) -> None:
        self.validate()
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                pixel_ids=np.asarray(self.pixel_ids, dtype=np.str_),
                s2_values=self.s2_values,
                s2_valid=self.s2_valid,
                s2_days=self.s2_days,
                s1a_values=self.s1a_values,
                s1a_valid=self.s1a_valid,
                s1a_days=self.s1a_days,
                s1d_values=self.s1d_values,
                s1d_valid=self.s1d_valid,
                s1d_days=self.s1d_days,
            )
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> PixelTimelines:
        with np.load(path, allow_pickle=False) as payload:
            result = cls(
                pixel_ids=tuple(str(value) for value in payload["pixel_ids"]),
                s2_values=payload["s2_values"],
                s2_valid=payload["s2_valid"],
                s2_days=payload["s2_days"],
                s1a_values=payload["s1a_values"],
                s1a_valid=payload["s1a_valid"],
                s1a_days=payload["s1a_days"],
                s1d_values=payload["s1d_values"],
                s1d_valid=payload["s1d_valid"],
                s1d_days=payload["s1d_days"],
            )
        result.validate()
        return result


def concatenate_timelines(
    seed: PixelTimelines,
    extension: PixelTimelines,
) -> PixelTimelines:
    if seed.pixel_ids != extension.pixel_ids:
        raise ValueError("timeline extension pixel IDs do not match the seed")

    def concatenate_modality(
        seed_values: np.ndarray,
        seed_valid: np.ndarray,
        seed_days: np.ndarray,
        extension_values: np.ndarray,
        extension_valid: np.ndarray,
        extension_days: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        days = np.concatenate((seed_days, extension_days))
        if len(np.unique(days)) != len(days):
            raise ValueError("timeline extension contains duplicate observation days")
        order = np.argsort(days, kind="stable")
        return (
            np.concatenate((seed_values, extension_values), axis=0)[order],
            np.concatenate((seed_valid, extension_valid), axis=0)[order],
            days[order],
        )

    s2_values, s2_valid, s2_days = concatenate_modality(
        seed.s2_values,
        seed.s2_valid,
        seed.s2_days,
        extension.s2_values,
        extension.s2_valid,
        extension.s2_days,
    )
    s1a_values, s1a_valid, s1a_days = concatenate_modality(
        seed.s1a_values,
        seed.s1a_valid,
        seed.s1a_days,
        extension.s1a_values,
        extension.s1a_valid,
        extension.s1a_days,
    )
    s1d_values, s1d_valid, s1d_days = concatenate_modality(
        seed.s1d_values,
        seed.s1d_valid,
        seed.s1d_days,
        extension.s1d_values,
        extension.s1d_valid,
        extension.s1d_days,
    )
    result = PixelTimelines(
        pixel_ids=seed.pixel_ids,
        s2_values=s2_values,
        s2_valid=s2_valid,
        s2_days=s2_days,
        s1a_values=s1a_values,
        s1a_valid=s1a_valid,
        s1a_days=s1a_days,
        s1d_values=s1d_values,
        s1d_valid=s1d_valid,
        s1d_days=s1d_days,
    )
    result.validate()
    return result


class MPCMaterializer:
    def __init__(
        self,
        stack_chunksize: int = 256,
        read_retries: int = 3,
        group_workers: int = 4,
    ):
        self.stack_chunksize = stack_chunksize
        self.read_retries = max(0, int(read_retries))
        self.group_workers = int(group_workers)
        if self.group_workers < 1:
            raise ValueError("group_workers must be positive")

    def _stack(
        self,
        items: Sequence[object],
        assets: Sequence[str],
        grid: RasterWindow,
        resampling: str,
        rescale: bool,
    ) -> np.ndarray:
        try:
            import stackstac
            from rasterio import Env
            from rasterio.enums import Resampling
        except ImportError as error:
            raise RuntimeError("install plain_tessera_incremental/requirements.txt") from error
        method = getattr(Resampling, resampling)
        output_dtype = "float64" if rescale else "float32"
        fill_value = np.float64(np.nan) if rescale else np.float32(np.nan)
        raw_items = detached_item_dicts(list(items))
        last_error: Exception | None = None
        for attempt in range(self.read_retries + 1):
            try:
                # planetary-computer's process-global SAS token cache is not
                # synchronized. Serialize only signing/cache access; raster
                # graph construction and reads remain parallel across groups.
                with _MPC_SIGNING_LOCK:
                    signed_items = signed_item_dicts(raw_items)
                data = stackstac.stack(
                    signed_items,
                    assets=list(assets),
                    epsg=grid.epsg,
                    resolution=(float(grid.resolution_m), float(grid.resolution_m)),
                    bounds=grid.bounds,
                    chunksize=self.stack_chunksize,
                    rescale=rescale,
                    resampling=method,
                    dtype=output_dtype,
                    fill_value=fill_value,
                )
                actual_assets = tuple(str(value) for value in data.coords["band"].values.tolist())
                if not actual_assets or data.sizes.get("time", 0) == 0:
                    raise NoSpatialCoverageError(
                        "stackstac found no requested assets over the tight raster window"
                    )
                if actual_assets != tuple(assets):
                    raise RuntimeError(
                        "stackstac changed checkpoint-critical band order: "
                        f"{actual_assets} != {tuple(assets)}"
                    )
                if data.sizes["y"] != grid.height or data.sizes["x"] != grid.width:
                    raise RuntimeError(
                        "stackstac output grid does not match the requested raster window: "
                        f"{data.sizes['y']}x{data.sizes['x']} != "
                        f"{grid.height}x{grid.width}"
                    )
                # Date/orbit groups are already parallelized by the bounded outer
                # pool. Keep each Dask graph synchronous to avoid an unbounded
                # nested thread pool multiplying remote COG requests.
                with Env(
                    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                    GDAL_HTTP_MAX_RETRY=min(3, self.read_retries),
                    GDAL_HTTP_RETRY_DELAY=1,
                ):
                    return (
                        data.transpose("time", "band", "y", "x")
                        .compute(scheduler="synchronous")
                        .values
                    )
            except NoSpatialCoverageError:
                raise
            except Exception as error:
                last_error = error
                if attempt >= self.read_retries:
                    break
                base_delay = min(2**attempt, 60)
                delay = base_delay + random.uniform(0.0, min(1.0, base_delay * 0.25))
                LOGGER.warning(
                    "raster read failed (%d/%d, %s); re-signing and retrying in %.1fs: %s",
                    attempt + 1,
                    self.read_retries + 1,
                    type(error).__name__,
                    delay,
                    _sanitized_error(error),
                )
                time.sleep(delay)
        error_type = type(last_error).__name__ if last_error is not None else "unknown"
        error_message = _sanitized_error(last_error) if last_error is not None else "unknown"
        raise RuntimeError(
            f"MPC raster read failed after retries; last error type: {error_type}: {error_message}"
        ) from None

    @staticmethod
    def _group_s2(items: Sequence[object]) -> dict[date, list[object]]:
        groups: dict[date, list[object]] = defaultdict(list)
        for item in items:
            if item.datetime is not None:
                groups[item.datetime.date()].append(item)
        return dict(sorted(groups.items()))

    @staticmethod
    def _group_s1(items: Sequence[object]) -> dict[tuple[date, str], list[object]]:
        groups: dict[tuple[date, str], list[object]] = defaultdict(list)
        for item in items:
            if item.datetime is None:
                continue
            orbit = str(item.properties.get("sat:orbit_state", "unknown")).lower()
            if orbit in {"ascending", "descending"}:
                groups[(item.datetime.date(), orbit)].append(item)
        return dict(sorted(groups.items()))

    @staticmethod
    def _ordered_results(
        futures: Sequence[Future[tuple[np.ndarray, np.ndarray] | None]],
        executor: ThreadPoolExecutor,
    ) -> list[tuple[np.ndarray, np.ndarray] | None]:
        done, _ = wait(futures, return_when=FIRST_EXCEPTION)
        failed = any(future.exception() is not None for future in done)
        if failed:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            error = next(
                future.exception()
                for future in futures
                if not future.cancelled() and future.exception() is not None
            )
            raise RuntimeError(
                "parallel raster materialization failed "
                f"({type(error).__name__}): {_sanitized_error(error)}"
            ) from None
        executor.shutdown(wait=True)
        return [future.result() for future in futures]

    def materialize(
        self,
        raw_s2_items: list[dict[str, object]],
        raw_s1_items: list[dict[str, object]],
        grid: RasterWindow,
        pixel_ids: tuple[str, ...],
        rows: np.ndarray,
        columns: np.ndarray,
        group_cache_dir: Path | None = None,
    ) -> PixelTimelines:
        if rows.shape != columns.shape or rows.shape != (len(pixel_ids),):
            raise ValueError("pixel IDs and raster indices do not align")
        s2_items = unsigned_items(raw_s2_items)
        s1_items = unsigned_items(raw_s1_items)
        grid_bounds_wgs84 = projected_bounds_to_wgs84(grid.bounds, grid.epsg)

        def process_s2_group(
            observed: date,
            items: Sequence[object],
        ) -> tuple[np.ndarray, np.ndarray] | None:
            cache_path = (
                None
                if group_cache_dir is None
                else group_cache_dir / f"s2-{observed.isoformat()}.npz"
            )
            if cache_path is not None:
                cached = _load_group_cache(
                    cache_path,
                    len(pixel_ids),
                    len(S2_BANDS),
                    np.dtype(np.uint16),
                )
                if cached is not _CACHE_MISS:
                    return cached
            items = _items_intersecting_grid(items, S2_ASSETS, grid, grid_bounds_wgs84)
            if not items:
                result = None
            else:
                try:
                    scl = self._stack(items, ["SCL"], grid, "nearest", rescale=False)
                    bands = self._stack(items, S2_BANDS, grid, "bilinear", rescale=False)
                except NoSpatialCoverageError:
                    result = None
                else:
                    result = select_s2_daily_mosaic(
                        scl[:, 0, rows, columns],
                        bands[:, :, rows, columns],
                    )
            if cache_path is not None:
                _write_group_cache(
                    cache_path,
                    result,
                    len(S2_BANDS),
                    np.dtype(np.uint16),
                )
            return result

        def process_s1_group(
            observed: date,
            orbit: str,
            items: Sequence[object],
        ) -> tuple[np.ndarray, np.ndarray] | None:
            cache_path = (
                None
                if group_cache_dir is None
                else group_cache_dir / f"s1-{observed.isoformat()}-{orbit}.npz"
            )
            if cache_path is not None:
                cached = _load_group_cache(
                    cache_path,
                    len(pixel_ids),
                    len(S1_ASSETS),
                    np.dtype(np.int16),
                )
                if cached is not _CACHE_MISS:
                    return cached
            items = _items_intersecting_grid(items, S1_ASSETS, grid, grid_bounds_wgs84)
            if not items:
                result = None
            else:
                try:
                    bands = self._stack(items, S1_ASSETS, grid, "nearest", rescale=True)
                except NoSpatialCoverageError:
                    result = None
                else:
                    result = select_s1_daily_mosaic(bands[:, :, rows, columns])
            if cache_path is not None:
                _write_group_cache(
                    cache_path,
                    result,
                    len(S1_ASSETS),
                    np.dtype(np.int16),
                )
            return result

        s2_groups = list(self._group_s2(s2_items).items())
        s1_groups = list(self._group_s1(s1_items).items())
        group_count = len(s2_groups) + len(s1_groups)
        ordered_groups: list[tuple[str, date, str | None]] = []
        results: list[tuple[np.ndarray, np.ndarray] | None] = []
        if group_count:
            worker_count = min(self.group_workers, group_count)
            LOGGER.info(
                "materializing %d S2 date groups and %d S1 date/orbit groups with %d workers",
                len(s2_groups),
                len(s1_groups),
                worker_count,
            )
            executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="tessera-raster",
            )
            futures: list[Future[tuple[np.ndarray, np.ndarray] | None]] = []
            try:
                for index in range(max(len(s2_groups), len(s1_groups))):
                    if index < len(s2_groups):
                        observed, items = s2_groups[index]
                        ordered_groups.append(("s2", observed, None))
                        futures.append(executor.submit(process_s2_group, observed, items))
                    if index < len(s1_groups):
                        (observed, orbit), items = s1_groups[index]
                        ordered_groups.append(("s1", observed, orbit))
                        futures.append(executor.submit(process_s1_group, observed, orbit, items))
                results = self._ordered_results(futures, executor)
            except BaseException:
                executor.shutdown(wait=True, cancel_futures=True)
                raise

        s2_values: list[np.ndarray] = []
        s2_valid: list[np.ndarray] = []
        s2_days: list[int] = []
        s1_by_orbit: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[int]]] = {
            "ascending": ([], [], []),
            "descending": ([], [], []),
        }
        for (modality, observed, orbit), result in zip(ordered_groups, results, strict=True):
            if result is None:
                continue
            values, valid = result
            if modality == "s2":
                s2_values.append(values)
                s2_valid.append(valid)
                s2_days.append(epoch_day(observed))
            else:
                value_list, valid_list, day_list = s1_by_orbit[str(orbit)]
                value_list.append(values)
                valid_list.append(valid)
                day_list.append(epoch_day(observed))

        pixels = len(pixel_ids)

        def stack_values(values: list[np.ndarray], bands: int, dtype: np.dtype) -> np.ndarray:
            return (
                np.stack(values).astype(dtype, copy=False)
                if values
                else np.empty((0, pixels, bands), dtype=dtype)
            )

        def stack_valid(values: list[np.ndarray]) -> np.ndarray:
            return (
                np.stack(values).astype(bool, copy=False) if values else np.empty((0, pixels), bool)
            )

        ascending = s1_by_orbit["ascending"]
        descending = s1_by_orbit["descending"]
        result = PixelTimelines(
            pixel_ids=pixel_ids,
            s2_values=stack_values(s2_values, 10, np.dtype(np.uint16)),
            s2_valid=stack_valid(s2_valid),
            s2_days=np.asarray(s2_days, dtype=np.int32),
            s1a_values=stack_values(ascending[0], 2, np.dtype(np.int16)),
            s1a_valid=stack_valid(ascending[1]),
            s1a_days=np.asarray(ascending[2], dtype=np.int32),
            s1d_values=stack_values(descending[0], 2, np.dtype(np.int16)),
            s1d_valid=stack_valid(descending[1]),
            s1d_days=np.asarray(descending[2], dtype=np.int32),
        )
        result.validate()
        return result
