from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True)
class TemporalView:
    bands: torch.Tensor
    day_of_year: torch.Tensor
    valid: torch.Tensor
    observation_day: torch.Tensor | None = None
    relative_day: torch.Tensor | None = None


def sample_temporal_view(
    bands: torch.Tensor,
    day_of_year: torch.Tensor,
    valid: torch.Tensor,
    target_size: int,
    generator: torch.Generator | None = None,
    observation_day: torch.Tensor | None = None,
    window_start_day: torch.Tensor | None = None,
    dropout_probability: float = 0.0,
) -> TemporalView:
    """Sample a chronological padded view without inventing extra observations."""
    if bands.ndim != 3 or day_of_year.shape != valid.shape or bands.shape[:2] != valid.shape:
        raise ValueError("expected bands [B,T,C], day_of_year/valid [B,T]")
    if observation_day is not None and observation_day.shape != valid.shape:
        raise ValueError("observation_day must match the [batch, time] validity shape")
    if window_start_day is not None and window_start_day.shape != (bands.shape[0],):
        raise ValueError("window_start_day must have shape [batch]")
    if (observation_day is None) != (window_start_day is None):
        raise ValueError("observation_day and window_start_day must be supplied together")
    if target_size < 1:
        raise ValueError("target_size must be positive")
    if not 0 <= dropout_probability <= 1:
        raise ValueError("dropout_probability must be in [0, 1]")
    batch, _, channels = bands.shape
    output_bands = bands.new_zeros((batch, target_size, channels))
    output_doy = day_of_year.new_zeros((batch, target_size))
    output_valid = torch.zeros((batch, target_size), device=valid.device, dtype=torch.bool)
    output_day = (
        observation_day.new_zeros((batch, target_size))
        if observation_day is not None
        else None
    )
    output_relative = (
        observation_day.new_zeros((batch, target_size))
        if observation_day is not None
        else None
    )

    for row in range(batch):
        indices = torch.nonzero(valid[row].bool(), as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        if dropout_probability and indices.numel() > 1:
            keep = (
                torch.rand(
                    indices.numel(),
                    generator=generator,
                    device=indices.device,
                )
                >= dropout_probability
            )
            if not keep.any():
                chosen = torch.randint(
                    indices.numel(), (), generator=generator, device=indices.device
                )
                keep[chosen] = True
            indices = indices[keep]
        selected = indices
        if indices.numel() >= target_size:
            order = torch.randperm(indices.numel(), generator=generator, device=indices.device)
            selected = indices[order[:target_size]]
        sort_value = (
            observation_day[row, selected]
            if observation_day is not None
            else day_of_year[row, selected]
        )
        selected = selected[torch.argsort(sort_value, stable=True)]
        count = selected.numel()
        output_bands[row, :count] = bands[row, selected]
        output_doy[row, :count] = day_of_year[row, selected]
        output_valid[row, :count] = True
        if output_day is not None and output_relative is not None:
            output_day[row, :count] = observation_day[row, selected]
            output_relative[row, :count] = (
                observation_day[row, selected] - window_start_day[row]
            )
    return TemporalView(
        output_bands,
        output_doy,
        output_valid,
        output_day,
        output_relative,
    )


def paired_temporal_views(
    bands: torch.Tensor,
    day_of_year: torch.Tensor,
    valid: torch.Tensor,
    target_size: int,
    generator: torch.Generator | None = None,
    observation_day: torch.Tensor | None = None,
    window_start_day: torch.Tensor | None = None,
    dropout_probability: float = 0.0,
) -> tuple[TemporalView, TemporalView]:
    return (
        sample_temporal_view(
            bands,
            day_of_year,
            valid,
            target_size,
            generator,
            observation_day,
            window_start_day,
            dropout_probability,
        ),
        sample_temporal_view(
            bands,
            day_of_year,
            valid,
            target_size,
            generator,
            observation_day,
            window_start_day,
            dropout_probability,
        ),
    )
