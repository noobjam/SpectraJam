from datetime import datetime

import pytest

torch = pytest.importorskip("torch")

from spectrajam.views import paired_temporal_views
from spectrajam.windows import (
    BatchWindow,
    TemporalSeries,
    TemporalWindow,
    WindowSamplingPolicy,
    cumulative_prefix_windows,
    day_of_year,
    epoch_day,
    rolling_windows,
    sample_batch_window,
    slice_series,
    window_valid_mask,
)


def _series(days: list[int], channels: int = 10) -> TemporalSeries:
    return TemporalSeries(
        bands=torch.arange(len(days) * channels, dtype=torch.float32).reshape(
            len(days), channels
        ),
        observation_day=torch.tensor(days),
        day_of_year=torch.tensor([day_of_year(value) for value in days]),
        valid=torch.ones(len(days), dtype=torch.bool),
        observation_ids=tuple(f"obs-{index}" for index in range(len(days))),
    )


def test_half_open_cross_year_window_is_sorted_and_has_no_future_leakage() -> None:
    dec_31 = epoch_day("2023-12-31")
    jan_1 = epoch_day("2024-01-01")
    jan_2 = epoch_day("2024-01-02")
    series = _series([jan_2, dec_31, jan_1])
    selected = slice_series(
        series, TemporalWindow(dec_31, jan_2, mode="arbitrary")
    )
    assert selected.observation_day.tolist() == [dec_31, jan_1]
    assert selected.day_of_year.tolist() == [365, 1]
    assert selected.observation_ids == ("obs-1", "obs-2")


def test_rolling_and_prefix_window_boundaries_are_exact() -> None:
    assert rolling_windows(10, 20, width_days=7, step_days=2) == (
        TemporalWindow(10, 17, "rolling"),
        TemporalWindow(12, 19, "rolling"),
    )
    assert cumulative_prefix_windows(10, [17, 24]) == (
        TemporalWindow(10, 17, "prefix"),
        TemporalWindow(10, 24, "prefix"),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        cumulative_prefix_windows(10, [20, 19])


def test_training_sampler_can_force_a_seven_day_prefix() -> None:
    policy = WindowSamplingPolicy(
        minimum_days=7,
        maximum_days=7,
        anchor_days=(7,),
        anchor_probability=1.0,
        prefix_probability=1.0,
    )
    bounds = sample_batch_window(
        torch.tensor([100, 200]),
        torch.tensor([120, 230]),
        policy,
        torch.Generator().manual_seed(4),
    )
    assert bounds.start_day.tolist() == [100, 200]
    assert bounds.end_day.tolist() == [107, 207]


def test_window_mask_excludes_the_end_boundary() -> None:
    observation_day = torch.tensor([[99, 100, 106, 107]])
    valid = torch.ones_like(observation_day, dtype=torch.bool)
    mask = window_valid_mask(
        observation_day,
        valid,
        BatchWindow(torch.tensor([100]), torch.tensor([107])),
    )
    assert mask.tolist() == [[False, True, True, False]]


def test_short_views_pad_instead_of_duplicating_observations() -> None:
    bands = torch.randn(2, 6, 10)
    days = torch.tensor([[30, 10, 50, 20, 40, 60], [1, 2, 3, 4, 5, 6]])
    valid = torch.tensor(
        [[True, True, True, True, True, True], [True, True, False, False, False, False]]
    )
    first, _ = paired_temporal_views(
        bands, days, valid, target_size=4, generator=torch.Generator().manual_seed(4)
    )
    assert first.valid.sum(dim=1).tolist() == [4, 2]
    assert torch.equal(first.day_of_year[1], torch.tensor([1, 2, 0, 0]))
    assert torch.equal(first.bands[1, 2:], torch.zeros_like(first.bands[1, 2:]))


def test_sparse_ssl_views_apply_independent_observation_dropout() -> None:
    bands = torch.randn(2, 5, 10)
    days = torch.arange(1, 6).repeat(2, 1)
    valid = torch.ones(2, 5, dtype=torch.bool)
    first, second = paired_temporal_views(
        bands,
        days,
        valid,
        target_size=5,
        generator=torch.Generator().manual_seed(21),
        dropout_probability=1.0,
    )
    assert first.valid.sum(dim=1).tolist() == [1, 1]
    assert second.valid.sum(dim=1).tolist() == [1, 1]


def test_naive_datetime_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        epoch_day(datetime(2024, 1, 1))


def test_space_separated_datetime_is_normalized_to_utc() -> None:
    with_space = epoch_day("2024-01-01 00:30:00+05:00")
    with_t = epoch_day("2024-01-01T00:30:00+05:00")
    assert with_space == with_t == epoch_day("2023-12-31")
    with pytest.raises(ValueError, match="timezone-aware"):
        epoch_day("2024-01-01 00:30:00")


def test_series_rejects_day_of_year_that_disagrees_with_absolute_day() -> None:
    with pytest.raises(ValueError, match="does not match"):
        TemporalSeries(
            bands=torch.zeros(1, 10),
            observation_day=torch.tensor([epoch_day("2024-01-01")]),
            day_of_year=torch.tensor([365]),
            valid=torch.ones(1, dtype=torch.bool),
            observation_ids=("bad-doy",),
        )
