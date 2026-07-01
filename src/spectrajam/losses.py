from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return F.normalize(value.float(), dim=-1, eps=1e-6)


def _weighted_mean(value: torch.Tensor, sample_weights: torch.Tensor | None) -> torch.Tensor:
    if sample_weights is None:
        return value.mean()
    if sample_weights.shape != value.shape:
        raise ValueError("sample weights must have one value per sample")
    weights = sample_weights.to(device=value.device, dtype=value.dtype).clamp_min(0)
    return (value * weights).mean()


def cosine_alignment(
    student: torch.Tensor,
    teacher: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    distance = 1.0 - (_normalized(student) * _normalized(teacher.detach())).sum(dim=-1)
    return _weighted_mean(distance, sample_weights)


def relational_geometry(
    student: torch.Tensor,
    teacher: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    student_gram = _normalized(student) @ _normalized(student).T
    teacher_gram = _normalized(teacher.detach()) @ _normalized(teacher.detach()).T
    squared_error = (student_gram - teacher_gram).pow(2)
    if sample_weights is None:
        return squared_error.mean()
    if sample_weights.shape != (student.shape[0],):
        raise ValueError("sample weights must have one value per sample")
    weights = sample_weights.to(squared_error).clamp_min(0)
    pair_weights = torch.sqrt(weights[:, None] * weights[None, :])
    return (squared_error * pair_weights).mean()


def barlow_twins(
    first: torch.Tensor,
    second: torch.Tensor,
    redundancy: float = 5e-3,
) -> torch.Tensor:
    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("Barlow Twins inputs must have matching [batch, features] shapes")
    if first.shape[0] < 2:
        raise ValueError("Barlow Twins requires at least two globally shuffled samples")
    first = first.float()
    second = second.float()
    first = (first - first.mean(0)) / first.std(0, unbiased=False).clamp_min(1e-4)
    second = (second - second.mean(0)) / second.std(0, unbiased=False).clamp_min(1e-4)
    correlation = first.T @ second / first.shape[0]
    diagonal = torch.diagonal(correlation)
    invariance = (diagonal - 1).pow(2).sum()
    off_diagonal = correlation.pow(2).sum() - diagonal.pow(2).sum()
    return invariance + redundancy * off_diagonal


def variance_floor(value: torch.Tensor, target_std: float = 1.0) -> torch.Tensor:
    standard_deviation = torch.sqrt(value.float().var(dim=0, unbiased=False) + 1e-4)
    return F.relu(target_std - standard_deviation).mean()


def covariance_penalty(value: torch.Tensor) -> torch.Tensor:
    value = value.float() - value.float().mean(dim=0)
    covariance = value.T @ value / max(value.shape[0] - 1, 1)
    diagonal = torch.diagonal(covariance)
    return (covariance.pow(2).sum() - diagonal.pow(2).sum()) / value.shape[1]


def _cross_correlation(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    first = (first.float() - first.float().mean(0)) / first.float().std(
        0, unbiased=False
    ).clamp_min(1e-4)
    second = (second.float() - second.float().mean(0)) / second.float().std(
        0, unbiased=False
    ).clamp_min(1e-4)
    return first.T @ second / first.shape[0]


def mixup_consistency(
    view_a: torch.Tensor,
    view_b: torch.Tensor,
    mixed: torch.Tensor,
    permutation: torch.Tensor,
    alpha: float | torch.Tensor,
    redundancy: float = 5e-3,
) -> torch.Tensor:
    """TESSERA-style correlation target for a mixed temporal view."""
    if view_a.shape != view_b.shape or view_a.shape != mixed.shape:
        raise ValueError("mixup embeddings must have matching shapes")
    shuffled_b = view_b[permutation]
    alpha_value = torch.as_tensor(alpha, device=view_a.device, dtype=torch.float32)
    actual_a = _cross_correlation(mixed, view_a)
    actual_b = _cross_correlation(mixed, view_b)
    target_a = (
        alpha_value * _cross_correlation(view_a, view_a)
        + (1 - alpha_value) * _cross_correlation(shuffled_b, view_a)
    )
    target_b = (
        alpha_value * _cross_correlation(view_a, view_b)
        + (1 - alpha_value) * _cross_correlation(shuffled_b, view_b)
    )
    return redundancy * (
        F.mse_loss(actual_a, target_a.detach(), reduction="sum")
        + F.mse_loss(actual_b, target_b.detach(), reduction="sum")
    )


@dataclass(frozen=True, slots=True)
class StudentLossWeights:
    teacher_alignment: float = 1.0
    relational_geometry: float = 1.0
    temporal_barlow: float = 0.1
    variance: float = 0.1
    covariance: float = 0.01


def student_objective(
    student_view_a: torch.Tensor,
    student_view_b: torch.Tensor,
    teacher: torch.Tensor,
    weights: StudentLossWeights | None = None,
    *,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or StudentLossWeights()
    mean_student = (student_view_a + student_view_b) / 2
    terms = {
        "teacher_alignment": (
            cosine_alignment(student_view_a, teacher, sample_weights)
            + cosine_alignment(student_view_b, teacher, sample_weights)
        )
        / 2,
        "relational_geometry": relational_geometry(
            mean_student, teacher, sample_weights
        ),
        "temporal_barlow": barlow_twins(student_view_a, student_view_b),
        "variance": (
            variance_floor(student_view_a) + variance_floor(student_view_b)
        )
        / 2,
        "covariance": (
            covariance_penalty(student_view_a) + covariance_penalty(student_view_b)
        )
        / 2,
    }
    total = (
        weights.teacher_alignment * terms["teacher_alignment"]
        + weights.relational_geometry * terms["relational_geometry"]
        + weights.temporal_barlow * terms["temporal_barlow"]
        + weights.variance * terms["variance"]
        + weights.covariance * terms["covariance"]
    )
    return total, terms


@dataclass(frozen=True, slots=True)
class LoRALossWeights:
    base_anchor: float = 0.5
    temporal_barlow: float = 1.0
    relational_geometry: float = 0.25


def lora_objective(
    adapted_view_a: torch.Tensor,
    adapted_view_b: torch.Tensor,
    frozen_base: torch.Tensor,
    weights: LoRALossWeights | None = None,
    *,
    sample_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    weights = weights or LoRALossWeights()
    adapted_mean = (adapted_view_a + adapted_view_b) / 2
    terms = {
        "base_anchor": cosine_alignment(adapted_mean, frozen_base, sample_weights),
        "temporal_barlow": barlow_twins(adapted_view_a, adapted_view_b),
        "relational_geometry": relational_geometry(
            adapted_mean, frozen_base, sample_weights
        ),
    }
    total = (
        weights.base_anchor * terms["base_anchor"]
        + weights.temporal_barlow * terms["temporal_barlow"]
        + weights.relational_geometry * terms["relational_geometry"]
    )
    return total, terms
