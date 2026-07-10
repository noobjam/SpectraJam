from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True, slots=True)
class PrefixWindow:
    window_id: str
    ordinal: int
    start: date
    end_exclusive: date

    def __post_init__(self) -> None:
        if not self.window_id:
            raise ValueError("window_id is required")
        if self.ordinal < 1:
            raise ValueError("window ordinal must be positive")
        if self.end_exclusive <= self.start:
            raise ValueError("window end must be after its start")

    @property
    def duration_days(self) -> int:
        return (self.end_exclusive - self.start).days


def build_prefix_windows(start: str, cutoffs: list[str] | tuple[str, ...]) -> tuple[PrefixWindow, ...]:
    start_date = date.fromisoformat(start)
    end_dates = tuple(date.fromisoformat(value) for value in cutoffs)
    if len(end_dates) != 4:
        raise ValueError("exactly four incremental cutoffs are required")
    if any(left >= right for left, right in zip(end_dates, end_dates[1:], strict=False)):
        raise ValueError("incremental cutoffs must be strictly increasing")
    if end_dates[0] <= start_date:
        raise ValueError("every cutoff must be after the common start")
    return tuple(
        PrefixWindow(f"w{ordinal}", ordinal, start_date, cutoff)
        for ordinal, cutoff in enumerate(end_dates, start=1)
    )
