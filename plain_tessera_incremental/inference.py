from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from .materialize import EPOCH, PixelTimelines
from .windows import PrefixWindow


OBSERVATION_BINS = np.arange(8, 257, 8, dtype=np.int32)


def build_resample_indices(valid_length: int, target_size: int) -> np.ndarray:
    """Pinned TESSERA v1.1 deterministic bucket resampling."""
    valid_length = int(valid_length)
    target_size = int(target_size)
    if valid_length <= 0:
        return np.array([], dtype=np.int64)
    if target_size == valid_length:
        return np.arange(valid_length, dtype=np.int64)
    if target_size < valid_length:
        chunks = np.array_split(np.arange(valid_length), target_size)
        return np.asarray([chunk[len(chunk) // 2] for chunk in chunks], dtype=np.int64)
    extra = target_size - valid_length
    anchors = np.linspace(0, valid_length - 1, num=extra + 2, dtype=np.float64)[1:-1]
    extras = np.rint(anchors).astype(np.int64)
    return np.concatenate(
        [np.arange(valid_length, dtype=np.int64), np.clip(extras, 0, valid_length - 1)]
    )


def bucket_size(valid_count: int) -> int:
    position = int(np.searchsorted(OBSERVATION_BINS, int(valid_count), side="left"))
    return int(OBSERVATION_BINS[min(position, len(OBSERVATION_BINS) - 1)])


def day_of_year(days: np.ndarray) -> np.ndarray:
    dates = np.datetime64("1970-01-01") + np.asarray(days).astype("timedelta64[D]")
    return (dates - dates.astype("datetime64[Y]")).astype(np.int32) + 1


@dataclass(frozen=True, slots=True)
class PreparedInputs:
    s2: np.ndarray
    s1: np.ndarray
    s2_valid_count: int
    s1_valid_count: int


def prepare_pixel_inputs(
    timelines: PixelTimelines,
    pixel_index: int,
    start_day: int,
    end_day: int,
    s2_target: int,
    s1_target: int,
    stats: object,
) -> PreparedInputs:
    s2_time = (timelines.s2_days >= start_day) & (timelines.s2_days < end_day)
    s2_indices = np.flatnonzero(s2_time & timelines.s2_valid[:, pixel_index])
    if len(s2_indices):
        local = build_resample_indices(len(s2_indices), s2_target)
        selected = s2_indices[local]
        bands = timelines.s2_values[selected, pixel_index].astype(np.float32)
        normalized = (bands - np.asarray(stats.s2_mean, np.float32)) / (
            np.asarray(stats.s2_std, np.float32) + 1e-9
        )
        s2 = np.column_stack((normalized, day_of_year(timelines.s2_days[selected]))).astype(
            np.float32,
            copy=False,
        )
    else:
        s2 = np.zeros((s2_target, 11), dtype=np.float32)

    s1_blocks: list[np.ndarray] = []
    s1_day_blocks: list[np.ndarray] = []
    orbit_specs = (
        (
            timelines.s1a_values,
            timelines.s1a_valid,
            timelines.s1a_days,
            stats.s1_ascending_mean,
            stats.s1_ascending_std,
        ),
        (
            timelines.s1d_values,
            timelines.s1d_valid,
            timelines.s1d_days,
            stats.s1_descending_mean,
            stats.s1_descending_std,
        ),
    )
    for values, valid, days, mean, std in orbit_specs:
        time_mask = (days >= start_day) & (days < end_day)
        indices = np.flatnonzero(time_mask & valid[:, pixel_index])
        if len(indices):
            normalized = (values[indices, pixel_index].astype(np.float32) - np.asarray(mean)) / (
                np.asarray(std) + 1e-9
            )
            s1_blocks.append(normalized.astype(np.float32, copy=False))
            s1_day_blocks.append(day_of_year(days[indices]))
    if s1_blocks:
        all_bands = np.concatenate(s1_blocks, axis=0)
        all_days = np.concatenate(s1_day_blocks, axis=0)
        local = build_resample_indices(len(all_bands), s1_target)
        s1 = np.column_stack((all_bands[local], all_days[local])).astype(np.float32, copy=False)
        s1_count = len(all_bands)
    else:
        s1 = np.zeros((s1_target, 3), dtype=np.float32)
        s1_count = 0
    return PreparedInputs(s2, s1, len(s2_indices), s1_count)


@dataclass(frozen=True, slots=True)
class WindowEmbeddings:
    embeddings: np.ndarray
    outcome: np.ndarray
    s2_valid_count: np.ndarray
    s1_valid_count: np.ndarray
    s2_input_count: np.ndarray
    s1_input_count: np.ndarray
    s2_source_count: int
    s1_source_count: int


class PlainTesseraRunner:
    def __init__(
        self,
        checkpoint_path: str,
        checkpoint_sha256: str,
        device: str = "auto",
        batch_size: int = 256,
    ):
        try:
            import torch

            from spectrajam.models.tessera_v11 import load_tessera_v11
            from spectrajam.normalization import get_stats
        except ImportError as error:
            raise RuntimeError("install the repository and plain pipeline requirements") from error
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.stats = get_stats("mpc")
        self.model = load_tessera_v11(checkpoint_path, checkpoint_sha256, self.device)
        self.model.requires_grad_(False).eval()

    def embed_window(
        self, timelines: PixelTimelines, window: PrefixWindow
    ) -> WindowEmbeddings:
        timelines.validate()
        start_day = (window.start - EPOCH).days
        end_day = (window.end_exclusive - EPOCH).days
        pixels = len(timelines.pixel_ids)

        def counts(valid: np.ndarray, days: np.ndarray) -> np.ndarray:
            selected = (days >= start_day) & (days < end_day)
            return valid[selected].sum(axis=0, dtype=np.int32)

        s2_counts = counts(timelines.s2_valid, timelines.s2_days)
        s1_counts = counts(timelines.s1a_valid, timelines.s1a_days) + counts(
            timelines.s1d_valid, timelines.s1d_days
        )
        embeddings = np.full((pixels, 128), np.nan, dtype=np.float32)
        outcome = np.full(pixels, "empty_window", dtype="U16")
        s2_input = np.zeros(pixels, dtype=np.int32)
        s1_input = np.zeros(pixels, dtype=np.int32)
        groups: dict[tuple[int, int], list[int]] = {}
        for pixel in np.flatnonzero((s2_counts + s1_counts) > 0):
            key = (bucket_size(int(s2_counts[pixel])), bucket_size(int(s1_counts[pixel])))
            groups.setdefault(key, []).append(int(pixel))

        for (s2_target, s1_target), group in sorted(groups.items()):
            for offset in range(0, len(group), self.batch_size):
                batch_pixels = group[offset : offset + self.batch_size]
                s2_batch = np.empty((len(batch_pixels), s2_target, 11), dtype=np.float32)
                s1_batch = np.empty((len(batch_pixels), s1_target, 3), dtype=np.float32)
                for row, pixel in enumerate(batch_pixels):
                    prepared = prepare_pixel_inputs(
                        timelines,
                        pixel,
                        start_day,
                        end_day,
                        s2_target,
                        s1_target,
                        self.stats,
                    )
                    s2_batch[row] = prepared.s2
                    s1_batch[row] = prepared.s1
                with self.torch.inference_mode():
                    encoded = self.model(
                        self.torch.from_numpy(s2_batch).to(self.device),
                        self.torch.from_numpy(s1_batch).to(self.device),
                    )
                values = encoded[:, :128].detach().float().cpu().numpy()
                if values.shape != (len(batch_pixels), 128) or not np.isfinite(values).all():
                    raise RuntimeError("plain TESSERA returned an invalid embedding batch")
                embeddings[batch_pixels] = values
                outcome[batch_pixels] = "complete"
                s2_input[batch_pixels] = s2_target
                s1_input[batch_pixels] = s1_target

        s2_source_count = int(
            np.count_nonzero((timelines.s2_days >= start_day) & (timelines.s2_days < end_day))
        )
        s1_source_count = int(
            np.count_nonzero((timelines.s1a_days >= start_day) & (timelines.s1a_days < end_day))
            + np.count_nonzero(
                (timelines.s1d_days >= start_day) & (timelines.s1d_days < end_day)
            )
        )
        return WindowEmbeddings(
            embeddings=embeddings,
            outcome=outcome,
            s2_valid_count=s2_counts,
            s1_valid_count=s1_counts,
            s2_input_count=s2_input,
            s1_input_count=s1_input,
            s2_source_count=s2_source_count,
            s1_source_count=s1_source_count,
        )
