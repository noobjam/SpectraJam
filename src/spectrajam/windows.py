from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Literal

import torch

WindowMode = Literal["arbitrary", "rolling", "prefix"]
_EPOCH = date(1970, 1, 1)


def epoch_day(value: str | date | datetime) -> int:
    """Convert an ISO date or timezone-aware datetime to a UTC calendar-day index."""
    if isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        value = (
            date.fromisoformat(normalized)
            if len(normalized) == 10
            else datetime.fromisoformat(normalized)
        )
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetime windows must be timezone-aware")
        value = value.astimezone(UTC).date()
    if not isinstance(value, date):
        raise TypeError("window boundaries must be ISO strings, dates, or datetimes")
    return (value - _EPOCH).days


def date_from_epoch_day(value: int) -> date:
    return _EPOCH + timedelta(days=int(value))


def day_of_year(value: int) -> int:
    return date_from_epoch_day(value).timetuple().tm_yday


@dataclass(frozen=True, slots=True)
class TemporalWindow:
    """A causal, half-open UTC-day interval: ``start_day <= t < end_day``."""

    start_day: int
    end_day: int
    mode: WindowMode = "arbitrary"

    def __post_init__(self) -> None:
        if self.end_day <= self.start_day:
            raise ValueError("a temporal window must have positive duration")
        if self.mode not in {"arbitrary", "rolling", "prefix"}:
            raise ValueError(f"unsupported temporal window mode: {self.mode}")

    @property
    def duration_days(self) -> int:
        return self.end_day - self.start_day

    @classmethod
    def from_dates(
        cls,
        start: str | date | datetime,
        end: str | date | datetime,
        mode: WindowMode = "arbitrary",
    ) -> TemporalWindow:
        return cls(epoch_day(start), epoch_day(end), mode)


def rolling_windows(
    start_day: int,
    stop_day: int,
    width_days: int,
    step_days: int = 1,
) -> tuple[TemporalWindow, ...]:
    if width_days < 1 or step_days < 1:
        raise ValueError("rolling width and step must be positive")
    if stop_day <= start_day:
        raise ValueError("rolling stop must be after start")
    return tuple(
        TemporalWindow(cursor, cursor + width_days, "rolling")
        for cursor in range(start_day, stop_day - width_days + 1, step_days)
    )


def cumulative_prefix_windows(
    start_day: int, end_days: tuple[int, ...] | list[int]
) -> tuple[TemporalWindow, ...]:
    windows = tuple(TemporalWindow(start_day, int(end), "prefix") for end in end_days)
    if any(left.end_day >= right.end_day for left, right in pairwise(windows)):
        raise ValueError("prefix end days must be strictly increasing")
    return windows


@dataclass(frozen=True, slots=True)
class TemporalSeries:
    """One normalized sensor timeline before a temporal window is selected."""

    bands: torch.Tensor
    observation_day: torch.Tensor
    day_of_year: torch.Tensor
    valid: torch.Tensor
    observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.bands.ndim != 2:
            raise ValueError("series bands must have shape [time, channels]")
        if not self.bands.is_floating_point():
            raise ValueError("series bands must use a floating-point dtype")
        length = self.bands.shape[0]
        if self.observation_day.shape != (length,):
            raise ValueError("observation_day must have shape [time]")
        if self.day_of_year.shape != (length,) or self.valid.shape != (length,):
            raise ValueError("day_of_year and valid must have shape [time]")
        if len(self.observation_ids) != length:
            raise ValueError("observation_ids must contain one stable ID per timestep")
        if len(set(self.observation_ids)) != length:
            raise ValueError("observation IDs must be unique within a sensor timeline")
        if self.observation_day.is_floating_point():
            raise ValueError("observation_day must use an integer dtype")
        if self.day_of_year.is_floating_point():
            raise ValueError("day_of_year must use an integer dtype")
        if self.valid.dtype != torch.bool:
            raise ValueError("valid must use torch.bool")
        devices = {
            self.bands.device,
            self.observation_day.device,
            self.day_of_year.device,
            self.valid.device,
        }
        if len(devices) != 1:
            raise ValueError("all TemporalSeries tensors must use the same device")
        if length and not torch.all((self.day_of_year >= 1) & (self.day_of_year <= 366)):
            raise ValueError("day_of_year must be in [1, 366]")
        expected_doy = torch.tensor(
            [day_of_year(value) for value in self.observation_day.detach().cpu().tolist()],
            device=self.day_of_year.device,
            dtype=self.day_of_year.dtype,
        )
        if not torch.equal(self.day_of_year, expected_doy):
            raise ValueError("day_of_year does not match the absolute observation day")
        if length and not torch.isfinite(self.bands[self.valid]).all():
            raise ValueError("valid observations must contain finite band values")


def slice_series(series: TemporalSeries, window: TemporalWindow) -> TemporalSeries:
    """Select valid observations in a window and return them in deterministic order."""
    selected = torch.nonzero(
        series.valid.bool()
        & (series.observation_day >= window.start_day)
        & (series.observation_day < window.end_day),
        as_tuple=False,
    ).flatten()
    indices = sorted(
        (int(index) for index in selected.cpu()),
        key=lambda index: (int(series.observation_day[index]), series.observation_ids[index]),
    )
    tensor_indices = torch.tensor(indices, device=series.bands.device, dtype=torch.long)
    ids = tuple(series.observation_ids[index] for index in indices)
    return TemporalSeries(
        bands=series.bands.index_select(0, tensor_indices),
        observation_day=series.observation_day.index_select(0, tensor_indices),
        day_of_year=series.day_of_year.index_select(0, tensor_indices),
        valid=torch.ones(len(indices), device=series.valid.device, dtype=torch.bool),
        observation_ids=ids,
    )


@dataclass(frozen=True, slots=True)
class WindowSamplingPolicy:
    """Training distribution over window lengths and start positions."""

    minimum_days: int = 7
    maximum_days: int = 365
    anchor_days: tuple[int, ...] = (7, 14, 30, 60, 90, 180, 365)
    anchor_probability: float = 0.7
    prefix_probability: float = 0.2
    view_dropout_probability: float = 0.2

    def validate(self) -> None:
        if self.minimum_days < 1 or self.maximum_days < self.minimum_days:
            raise ValueError("invalid training window duration range")
        if not self.anchor_days or any(
            value < self.minimum_days or value > self.maximum_days
            for value in self.anchor_days
        ):
            raise ValueError("window anchors must lie inside the training duration range")
        if len(set(self.anchor_days)) != len(self.anchor_days):
            raise ValueError("window anchors must be unique")
        if not 0 <= self.anchor_probability <= 1:
            raise ValueError("anchor_probability must be in [0, 1]")
        if not 0 <= self.prefix_probability <= 1:
            raise ValueError("prefix_probability must be in [0, 1]")
        if not 0 <= self.view_dropout_probability <= 1:
            raise ValueError("view_dropout_probability must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class BatchWindow:
    start_day: torch.Tensor
    end_day: torch.Tensor

    def __post_init__(self) -> None:
        if self.start_day.ndim != 1 or self.start_day.shape != self.end_day.shape:
            raise ValueError("batch window bounds must have matching [batch] shapes")
        if torch.any(self.end_day <= self.start_day):
            raise ValueError("every batch window must have positive duration")

    @property
    def duration_days(self) -> torch.Tensor:
        return self.end_day - self.start_day


def _sample_duration(
    policy: WindowSamplingPolicy,
    maximum_available: int,
    generator: torch.Generator | None,
    device: torch.device,
) -> int:
    maximum = min(policy.maximum_days, maximum_available)
    if maximum < policy.minimum_days:
        raise ValueError("source coverage is shorter than the minimum training window")
    anchors = tuple(value for value in policy.anchor_days if value <= maximum)
    choose_anchor = bool(
        anchors
        and torch.rand((), generator=generator, device=device).item()
        < policy.anchor_probability
    )
    if choose_anchor:
        index = int(
            torch.randint(len(anchors), (), generator=generator, device=device).item()
        )
        return anchors[index]
    draw = float(torch.rand((), generator=generator, device=device).item())
    low = math.log(policy.minimum_days)
    high = math.log(maximum)
    return max(policy.minimum_days, min(maximum, int(round(math.exp(low + draw * (high - low))))))


def sample_batch_window(
    coverage_start_day: torch.Tensor,
    coverage_end_day: torch.Tensor,
    policy: WindowSamplingPolicy,
    generator: torch.Generator | None = None,
) -> BatchWindow:
    """Draw one duration per batch and causal starts independently per sample."""
    policy.validate()
    if coverage_start_day.ndim != 1 or coverage_start_day.shape != coverage_end_day.shape:
        raise ValueError("coverage bounds must have matching [batch] shapes")
    coverage_days = coverage_end_day - coverage_start_day
    if generator is not None and generator.device != coverage_start_day.device:
        raise ValueError("generator and coverage tensors must use the same device")
    if torch.any(coverage_days <= 0):
        raise ValueError("coverage intervals must have positive duration")
    duration = _sample_duration(
        policy,
        int(coverage_days.min().item()),
        generator,
        coverage_start_day.device,
    )
    starts: list[int] = []
    for row in range(coverage_start_day.shape[0]):
        available_offset = int(coverage_days[row].item()) - duration
        prefix = (
            torch.rand((), generator=generator, device=coverage_start_day.device).item()
            < policy.prefix_probability
        )
        offset = 0
        if not prefix and available_offset:
            offset = int(
                torch.randint(
                    available_offset + 1,
                    (),
                    generator=generator,
                    device=coverage_start_day.device,
                ).item()
            )
        starts.append(int(coverage_start_day[row].item()) + offset)
    start = torch.tensor(starts, device=coverage_start_day.device, dtype=torch.long)
    return BatchWindow(start, start + duration)


def window_valid_mask(
    observation_day: torch.Tensor,
    valid: torch.Tensor,
    window: BatchWindow,
) -> torch.Tensor:
    if observation_day.shape != valid.shape or observation_day.ndim != 2:
        raise ValueError("observation days and validity must have shape [batch, time]")
    if observation_day.shape[0] != window.start_day.shape[0]:
        raise ValueError("window and observation batch sizes differ")
    return (
        valid.bool()
        & (observation_day >= window.start_day[:, None])
        & (observation_day < window.end_day[:, None])
    )


def relative_days(
    observation_day: torch.Tensor,
    valid: torch.Tensor,
    window: BatchWindow,
) -> torch.Tensor:
    mask = window_valid_mask(observation_day, valid, window)
    relative = observation_day - window.start_day[:, None]
    return torch.where(mask, relative, torch.zeros_like(relative))
