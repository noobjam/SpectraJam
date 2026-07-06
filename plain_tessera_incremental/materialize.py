from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
import logging
from pathlib import Path
import time
from typing import Sequence

import numpy as np

from .catalog import (
    S1_ASSETS,
    S2_ASSETS,
    detached_item_dicts,
    signed_item_dicts,
    unsigned_items,
)
from .geometry import RasterWindow


S2_BANDS = S2_ASSETS[:-1]
INVALID_SCL = np.array([0, 1, 2, 3, 8, 9], dtype=np.int16)
EPOCH = date(1970, 1, 1)
LOGGER = logging.getLogger(__name__)


def epoch_day(value: date) -> int:
    return (value - EPOCH).days


def harmonize_s2_mpc(values: np.ndarray) -> np.ndarray:
    """Apply the post-2022 MPC BOA offset and upstream uint16 storage cast."""
    result = np.asarray(values, dtype=np.float32).copy()
    finite = np.isfinite(result)
    result[finite & (result >= 1000.0)] -= 1000.0
    result[~finite] = 0.0
    return np.clip(result, 0, np.iinfo(np.uint16).max).astype(np.uint16)


def select_s2_daily_mosaic(
    scl: np.ndarray, bands: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


class MPCMaterializer:
    def __init__(self, stack_chunksize: int = 256, read_retries: int = 3):
        self.stack_chunksize = stack_chunksize
        self.read_retries = max(0, int(read_retries))

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
                data = stackstac.stack(
                    signed_item_dicts(raw_items),
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
                actual_assets = tuple(
                    str(value) for value in data.coords["band"].values.tolist()
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
                return data.transpose("time", "band", "y", "x").compute().values
            except Exception as error:
                last_error = error
                if attempt >= self.read_retries:
                    break
                delay = min(2**attempt, 8)
                LOGGER.warning(
                    "raster read failed (%d/%d, %s); re-signing and retrying in %ds",
                    attempt + 1,
                    self.read_retries + 1,
                    type(error).__name__,
                    delay,
                )
                time.sleep(delay)
        error_type = type(last_error).__name__ if last_error is not None else "unknown"
        raise RuntimeError(
            f"MPC raster read failed after retries; last error type: {error_type}"
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

    def materialize(
        self,
        raw_s2_items: list[dict[str, object]],
        raw_s1_items: list[dict[str, object]],
        grid: RasterWindow,
        pixel_ids: tuple[str, ...],
        rows: np.ndarray,
        columns: np.ndarray,
    ) -> PixelTimelines:
        if rows.shape != columns.shape or rows.shape != (len(pixel_ids),):
            raise ValueError("pixel IDs and raster indices do not align")
        s2_items = unsigned_items(raw_s2_items)
        s1_items = unsigned_items(raw_s1_items)
        s2_values: list[np.ndarray] = []
        s2_valid: list[np.ndarray] = []
        s2_days: list[int] = []
        for observed, items in self._group_s2(s2_items).items():
            scl = self._stack(items, ["SCL"], grid, "nearest", rescale=False)
            bands = self._stack(items, S2_BANDS, grid, "bilinear", rescale=False)
            values, valid = select_s2_daily_mosaic(
                scl[:, 0, rows, columns],
                bands[:, :, rows, columns],
            )
            s2_values.append(values)
            s2_valid.append(valid)
            s2_days.append(epoch_day(observed))

        s1_by_orbit: dict[str, tuple[list[np.ndarray], list[np.ndarray], list[int]]] = {
            "ascending": ([], [], []),
            "descending": ([], [], []),
        }
        for (observed, orbit), items in self._group_s1(s1_items).items():
            bands = self._stack(items, S1_ASSETS, grid, "nearest", rescale=True)
            values, valid = select_s1_daily_mosaic(bands[:, :, rows, columns])
            value_list, valid_list, day_list = s1_by_orbit[orbit]
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
            return np.stack(values).astype(bool, copy=False) if values else np.empty((0, pixels), bool)

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
