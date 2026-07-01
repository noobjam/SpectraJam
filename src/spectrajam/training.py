from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from .ledger import AcquisitionLedger
from .losses import (
    LoRALossWeights,
    StudentLossWeights,
    lora_objective,
    mixup_consistency,
    student_objective,
)
from .models.windowed import WindowedTesseraEncoder
from .parity import parity_receipt_matches
from .views import TemporalView, paired_temporal_views
from .windows import (
    BatchWindow,
    WindowSamplingPolicy,
    relative_days,
    sample_batch_window,
    window_valid_mask,
)


@dataclass(slots=True)
class ObservationBatch:
    """A storage-shard batch; model semantics may be any sub-window of the shard."""

    s2_bands: torch.Tensor
    s2_day_of_year: torch.Tensor
    s2_valid: torch.Tensor
    s1_bands: torch.Tensor
    s1_day_of_year: torch.Tensor
    s1_valid: torch.Tensor
    teacher_target: torch.Tensor | None
    sample_ids: tuple[str, ...]
    spatial_blocks: tuple[str, ...]
    countries: tuple[str, ...]
    temporal_splits: tuple[str, ...]
    s2_observation_day: torch.Tensor | None = None
    s1_observation_day: torch.Tensor | None = None
    coverage_start_day: torch.Tensor | None = None
    coverage_end_day: torch.Tensor | None = None

    def validate(self, require_window_metadata: bool = False) -> None:
        batch = self.s2_bands.shape[0]
        if self.s2_bands.ndim != 3 or self.s2_bands.shape[-1] != 10:
            raise ValueError("s2_bands must have shape [batch, time, 10]")
        if self.s1_bands.ndim != 3 or self.s1_bands.shape[-1] != 2:
            raise ValueError("s1_bands must have shape [batch, time, 2]")
        for bands, day, valid in (
            (self.s2_bands, self.s2_day_of_year, self.s2_valid),
            (self.s1_bands, self.s1_day_of_year, self.s1_valid),
        ):
            if bands.shape[:2] != day.shape or day.shape != valid.shape:
                raise ValueError("band, day-of-year, and validity shapes do not align")
            if bands.shape[0] != batch:
                raise ValueError("modalities have different batch sizes")
        if self.teacher_target is not None and (
            self.teacher_target.ndim != 2 or self.teacher_target.shape[0] != batch
        ):
            raise ValueError("teacher_target must have shape [batch, embedding]")
        if batch < 2:
            raise ValueError("regional SSL requires globally shuffled batches of at least two")
        if not (
            len(self.sample_ids)
            == len(self.spatial_blocks)
            == len(self.countries)
            == len(self.temporal_splits)
            == batch
        ):
            raise ValueError("batch provenance must contain one entry per sample")
        if len(set(self.sample_ids)) != batch:
            raise ValueError("a globally shuffled batch cannot contain duplicate sample IDs")
        if len(set(self.spatial_blocks)) < 2:
            raise ValueError("regional SSL batches must mix at least two spatial blocks")
        if not set(self.temporal_splits) <= {"train", "validation", "test"}:
            raise ValueError("batch temporal_splits contain an unsupported value")

        window_values = (
            self.s2_observation_day,
            self.s1_observation_day,
            self.coverage_start_day,
            self.coverage_end_day,
        )
        if require_window_metadata and any(value is None for value in window_values):
            raise ValueError(
                "window training requires absolute observation days and coverage bounds"
            )
        if require_window_metadata and self.teacher_target is not None:
            raise ValueError(
                "window training rejects precomputed targets; encode the same window"
            )
        if self.s2_observation_day is not None and (
            self.s2_observation_day.shape != self.s2_valid.shape
        ):
            raise ValueError("s2_observation_day must match s2_valid")
        if self.s1_observation_day is not None and (
            self.s1_observation_day.shape != self.s1_valid.shape
        ):
            raise ValueError("s1_observation_day must match s1_valid")
        if self.coverage_start_day is not None:
            if self.coverage_start_day.shape != (batch,):
                raise ValueError("coverage_start_day must have shape [batch]")
            if self.coverage_end_day is None or self.coverage_end_day.shape != (batch,):
                raise ValueError("coverage_end_day must have shape [batch]")
            if torch.any(self.coverage_end_day <= self.coverage_start_day):
                raise ValueError("coverage intervals must have positive duration")


@dataclass(frozen=True, slots=True)
class TrainingGate:
    ledger_path: str
    parity_receipt_path: str
    checkpoint_sha256: str
    manifest_sha256: str
    config_sha256: str

    def verify(self) -> None:
        ledger = AcquisitionLedger(self.ledger_path)
        ledger.require_binding(self.manifest_sha256, self.config_sha256)
        ledger.assert_complete()
        if not parity_receipt_matches(self.parity_receipt_path, self.checkpoint_sha256):
            raise RuntimeError(
                "upstream parity receipt is missing or belongs to another checkpoint"
            )


def _require_gate(gate: TrainingGate | None, allow_unverified: bool) -> None:
    if allow_unverified:
        return
    if gate is None:
        raise RuntimeError("training requires acquisition and upstream-parity gates")
    gate.verify()


def _require_training_split(batch: ObservationBatch) -> None:
    if set(batch.temporal_splits) != {"train"}:
        raise ValueError("trainers accept only observations declared in the train split")


def _choose_view_size(
    sizes: Sequence[int], generator: torch.Generator | None, device: torch.device
) -> int:
    if not sizes or any(size < 1 for size in sizes):
        raise ValueError("view sizes must be positive")
    index = int(torch.randint(len(sizes), (1,), generator=generator, device=device).item())
    return int(sizes[index])


def _window_for_batch(
    batch: ObservationBatch,
    policy: WindowSamplingPolicy,
    generator: torch.Generator | None,
) -> BatchWindow:
    assert batch.coverage_start_day is not None
    assert batch.coverage_end_day is not None
    return sample_batch_window(
        batch.coverage_start_day,
        batch.coverage_end_day,
        policy,
        generator,
    )


def _views_for_both_modalities(
    batch: ObservationBatch,
    target_size: int,
    generator: torch.Generator | None,
    window: BatchWindow | None = None,
    dropout_probability: float = 0.0,
) -> tuple[TemporalView, TemporalView, TemporalView, TemporalView]:
    if window is None:
        s2_valid = batch.s2_valid
        s1_valid = batch.s1_valid
        s2_day = None
        s1_day = None
        start = None
    else:
        assert batch.s2_observation_day is not None
        assert batch.s1_observation_day is not None
        s2_day = batch.s2_observation_day
        s1_day = batch.s1_observation_day
        start = window.start_day
        s2_valid = window_valid_mask(s2_day, batch.s2_valid, window)
        s1_valid = window_valid_mask(s1_day, batch.s1_valid, window)
    s2_a, s2_b = paired_temporal_views(
        batch.s2_bands,
        batch.s2_day_of_year,
        s2_valid,
        target_size,
        generator,
        s2_day,
        start,
        dropout_probability,
    )
    s1_a, s1_b = paired_temporal_views(
        batch.s1_bands,
        batch.s1_day_of_year,
        s1_valid,
        target_size,
        generator,
        s1_day,
        start,
        dropout_probability,
    )
    return s2_a, s2_b, s1_a, s1_b


def _encode_views(
    model: nn.Module,
    s2: TemporalView,
    s1: TemporalView,
    duration_days: torch.Tensor | None,
    s2_observation_count: torch.Tensor | None = None,
    s1_observation_count: torch.Tensor | None = None,
) -> torch.Tensor:
    return model(
        s2.bands,
        s2.day_of_year,
        s2.valid,
        s1.bands,
        s1.day_of_year,
        s1.valid,
        s2_relative_day=s2.relative_day,
        s1_relative_day=s1.relative_day,
        window_duration_days=duration_days,
        s2_observation_count=s2_observation_count,
        s1_observation_count=s1_observation_count,
    )


def _select_view(view: TemporalView, selected: torch.Tensor) -> TemporalView:
    return TemporalView(
        bands=view.bands[selected],
        day_of_year=view.day_of_year[selected],
        valid=view.valid[selected],
        observation_day=(
            None if view.observation_day is None else view.observation_day[selected]
        ),
        relative_day=None if view.relative_day is None else view.relative_day[selected],
    )


def _available_samples(
    s2: TemporalView,
    s1: TemporalView,
    spatial_blocks: tuple[str, ...],
) -> torch.Tensor:
    available = s2.valid.any(dim=1) | s1.valid.any(dim=1)
    if int(available.sum()) < 2:
        raise ValueError(
            "window has fewer than two samples with any observation; draw another batch"
        )
    retained_blocks = {
        block
        for block, keep in zip(spatial_blocks, available.tolist(), strict=True)
        if keep
    }
    if len(retained_blocks) < 2:
        raise ValueError(
            "window filtering left fewer than two spatial blocks; draw another batch"
        )
    return available


def _encode_complete_window(
    model: nn.Module,
    batch: ObservationBatch,
    window: BatchWindow,
) -> torch.Tensor:
    s2_valid, s1_valid = _complete_window_masks(batch, window)
    assert batch.s2_observation_day is not None
    assert batch.s1_observation_day is not None
    return model(
        batch.s2_bands,
        batch.s2_day_of_year,
        s2_valid,
        batch.s1_bands,
        batch.s1_day_of_year,
        s1_valid,
        s2_relative_day=relative_days(
            batch.s2_observation_day, batch.s2_valid, window
        ),
        s1_relative_day=relative_days(
            batch.s1_observation_day, batch.s1_valid, window
        ),
        window_duration_days=window.duration_days,
        s2_observation_count=s2_valid.sum(dim=1),
        s1_observation_count=s1_valid.sum(dim=1),
    )


def _complete_window_masks(
    batch: ObservationBatch, window: BatchWindow
) -> tuple[torch.Tensor, torch.Tensor]:
    assert batch.s2_observation_day is not None
    assert batch.s1_observation_day is not None
    s2_valid = window_valid_mask(batch.s2_observation_day, batch.s2_valid, window)
    s1_valid = window_valid_mask(batch.s1_observation_day, batch.s1_valid, window)
    return s2_valid, s1_valid


def _teacher_anchor_weights(
    window: BatchWindow,
    full_weight_days: int,
    full_weight_observations: int,
    s2_valid: torch.Tensor,
    s1_valid: torch.Tensor,
) -> torch.Tensor:
    if full_weight_days < 1:
        raise ValueError("teacher_anchor_full_weight_days must be positive")
    if full_weight_observations < 1:
        raise ValueError("teacher_anchor_full_observations must be positive")
    duration_weight = (window.duration_days.float() / full_weight_days).clamp(max=1.0)
    evidence_count = torch.maximum(s2_valid.sum(dim=1), s1_valid.sum(dim=1)).float()
    evidence_weight = (evidence_count / full_weight_observations).clamp(max=1.0)
    return duration_weight * evidence_weight


class StudentTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        view_sizes: Sequence[int] = (20, 30, 40),
        loss_weights: StudentLossWeights | None = None,
        gradient_clip_norm: float = 2.0,
        gate: TrainingGate | None = None,
        allow_unverified: bool = False,
        window_policy: WindowSamplingPolicy | None = None,
        teacher_model: nn.Module | None = None,
        teacher_anchor_full_weight_days: int = 90,
        teacher_anchor_full_observations: int = 10,
    ):
        _require_gate(gate, allow_unverified)
        if window_policy is not None and teacher_model is None:
            raise ValueError(
                "windowed distillation requires a frozen teacher on the same window"
            )
        self.model = model
        self.optimizer = optimizer
        self.view_sizes = tuple(view_sizes)
        self.loss_weights = loss_weights or StudentLossWeights()
        self.gradient_clip_norm = gradient_clip_norm
        self.window_policy = window_policy
        self.teacher_model = teacher_model
        self.teacher_anchor_full_weight_days = teacher_anchor_full_weight_days
        self.teacher_anchor_full_observations = teacher_anchor_full_observations
        if self.teacher_model is not None:
            self.teacher_model.requires_grad_(False).eval()

    def step(
        self, batch: ObservationBatch, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        windowed = self.window_policy is not None
        batch.validate(require_window_metadata=windowed)
        _require_training_split(batch)
        self.model.train()
        target_size = _choose_view_size(self.view_sizes, generator, batch.s2_bands.device)
        window = (
            _window_for_batch(batch, self.window_policy, generator)
            if self.window_policy is not None
            else None
        )
        s2_a, s2_b, s1_a, s1_b = _views_for_both_modalities(
            batch,
            target_size,
            generator,
            window,
            (
                self.window_policy.view_dropout_probability
                if self.window_policy is not None
                else 0.0
            ),
        )
        duration = window.duration_days if window is not None else None
        empty_fraction = 0.0
        available = None
        complete_s2_valid = batch.s2_valid
        complete_s1_valid = batch.s1_valid
        if window is not None:
            complete_s2_valid, complete_s1_valid = _complete_window_masks(batch, window)
            available = _available_samples(s2_a, s1_a, batch.spatial_blocks)
            empty_fraction = float((~available).float().mean())
            s2_a, s2_b = _select_view(s2_a, available), _select_view(s2_b, available)
            s1_a, s1_b = _select_view(s1_a, available), _select_view(s1_b, available)
            duration = duration[available]
            complete_s2_valid = complete_s2_valid[available]
            complete_s1_valid = complete_s1_valid[available]
        s2_count = complete_s2_valid.sum(dim=1)
        s1_count = complete_s1_valid.sum(dim=1)
        first = _encode_views(self.model, s2_a, s1_a, duration, s2_count, s1_count)
        second = _encode_views(self.model, s2_b, s1_b, duration, s2_count, s1_count)

        if window is None:
            if batch.teacher_target is None:
                raise ValueError("annual training requires teacher_target")
            teacher = batch.teacher_target.detach()
            sample_weights = None
        else:
            assert self.teacher_model is not None
            assert available is not None
            with torch.inference_mode():
                teacher = _encode_complete_window(self.teacher_model, batch, window)[
                    available
                ]
            sample_weights = _teacher_anchor_weights(
                window,
                self.teacher_anchor_full_weight_days,
                self.teacher_anchor_full_observations,
                *(_complete_window_masks(batch, window)),
            )[available]
        loss, terms = student_objective(
            first,
            second,
            teacher,
            self.loss_weights,
            sample_weights=sample_weights,
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        metrics = {"loss": float(loss.detach()), "view_size": float(target_size)}
        if window is not None:
            metrics["window_days"] = float(window.duration_days[0])
            metrics["teacher_anchor_weight"] = float(sample_weights[0])
            metrics["empty_window_fraction"] = empty_fraction
        return metrics | {key: float(value.detach()) for key, value in terms.items()}


class LoRATrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        view_size: int = 40,
        loss_weights: LoRALossWeights | None = None,
        mixup_weight: float = 1.0,
        gradient_clip_norm: float = 2.0,
        gate: TrainingGate | None = None,
        allow_unverified: bool = False,
        window_policy: WindowSamplingPolicy | None = None,
        reference_model: nn.Module | None = None,
        teacher_anchor_full_weight_days: int = 90,
        teacher_anchor_full_observations: int = 10,
    ):
        _require_gate(gate, allow_unverified)
        if not isinstance(model, WindowedTesseraEncoder):
            raise ValueError("LoRA training requires the mask-aware TESSERA wrapper")
        if window_policy is not None and reference_model is None:
            raise ValueError("windowed LoRA requires a frozen same-window base encoder")
        if window_policy is not None and model.window_conditioner is None:
            raise ValueError("windowed LoRA requires WindowedTesseraEncoder conditioning")
        self.model = model
        self.optimizer = optimizer
        self.view_size = view_size
        self.loss_weights = loss_weights or LoRALossWeights()
        self.mixup_weight = mixup_weight
        self.gradient_clip_norm = gradient_clip_norm
        self.window_policy = window_policy
        self.reference_model = reference_model
        self.teacher_anchor_full_weight_days = teacher_anchor_full_weight_days
        self.teacher_anchor_full_observations = teacher_anchor_full_observations
        if self.reference_model is not None:
            self.reference_model.requires_grad_(False).eval()

    def _annual_step(
        self,
        batch: ObservationBatch,
        s2_a: TemporalView,
        s2_b: TemporalView,
        s1_a: TemporalView,
        s1_b: TemporalView,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if batch.teacher_target is None:
            raise ValueError("annual LoRA training requires teacher_target")
        duration = torch.full(
            (s2_a.bands.shape[0],),
            365,
            device=s2_a.bands.device,
            dtype=torch.long,
        )
        s2_count = batch.s2_valid.sum(dim=1)
        s1_count = batch.s1_valid.sum(dim=1)
        first = _encode_views(
            self.model, s2_a, s1_a, duration, s2_count, s1_count
        )
        second = _encode_views(
            self.model, s2_b, s1_b, duration, s2_count, s1_count
        )
        loss, terms = lora_objective(
            first, second, batch.teacher_target.detach(), self.loss_weights
        )
        mixup = first.sum() * 0
        if all(bool(view.valid.all()) for view in (s2_a, s2_b, s1_a, s1_b)):
            permutation = torch.randperm(
                first.shape[0], generator=generator, device=first.device
            )
            alpha = torch.rand((), generator=generator, device=first.device)
            mixed_s2 = TemporalView(
                alpha * s2_a.bands + (1 - alpha) * s2_b.bands[permutation],
                alpha * s2_a.day_of_year + (1 - alpha) * s2_b.day_of_year[permutation],
                s2_a.valid,
            )
            mixed_s1 = TemporalView(
                alpha * s1_a.bands + (1 - alpha) * s1_b.bands[permutation],
                alpha * s1_a.day_of_year + (1 - alpha) * s1_b.day_of_year[permutation],
                s1_a.valid,
            )
            mixed_s2_count = alpha * s2_count + (1 - alpha) * s2_count[permutation]
            mixed_s1_count = alpha * s1_count + (1 - alpha) * s1_count[permutation]
            mixed = _encode_views(
                self.model,
                mixed_s2,
                mixed_s1,
                duration,
                mixed_s2_count,
                mixed_s1_count,
            )
            mixup = mixup_consistency(first, second, mixed, permutation, alpha)
            loss = loss + self.mixup_weight * mixup
        return loss, mixup, terms

    def _window_step(
        self,
        batch: ObservationBatch,
        window: BatchWindow,
        s2_a: TemporalView,
        s2_b: TemporalView,
        s1_a: TemporalView,
        s1_b: TemporalView,
        generator: torch.Generator | None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        assert self.reference_model is not None
        available = _available_samples(s2_a, s1_a, batch.spatial_blocks)
        s2_a, s2_b = _select_view(s2_a, available), _select_view(s2_b, available)
        s1_a, s1_b = _select_view(s1_a, available), _select_view(s1_b, available)
        duration = window.duration_days[available]
        complete_s2_valid, complete_s1_valid = _complete_window_masks(batch, window)
        s2_count = complete_s2_valid.sum(dim=1)[available]
        s1_count = complete_s1_valid.sum(dim=1)[available]
        first = _encode_views(
            self.model, s2_a, s1_a, duration, s2_count, s1_count
        )
        second = _encode_views(
            self.model, s2_b, s1_b, duration, s2_count, s1_count
        )
        with torch.inference_mode():
            reference = _encode_complete_window(self.reference_model, batch, window)[
                available
            ]
        sample_weights = _teacher_anchor_weights(
            window,
            self.teacher_anchor_full_weight_days,
            self.teacher_anchor_full_observations,
            complete_s2_valid,
            complete_s1_valid,
        )[available]
        loss, terms = lora_objective(
            first,
            second,
            reference,
            self.loss_weights,
            sample_weights=sample_weights,
        )

        # TESSERA input mixup is only defined when both views contain a full,
        # position-aligned draw. Sparse short windows keep the SSL/base terms
        # but do not fabricate dates merely to make mixup possible.
        all_dense = all(
            bool(view.valid.all()) for view in (s2_a, s2_b, s1_a, s1_b)
        )
        mixup = first.sum() * 0
        if all_dense:
            permutation = torch.randperm(
                first.shape[0], generator=generator, device=first.device
            )
            alpha = torch.rand((), generator=generator, device=first.device)
            mixed_s2 = TemporalView(
                alpha * s2_a.bands + (1 - alpha) * s2_b.bands[permutation],
                alpha * s2_a.day_of_year + (1 - alpha) * s2_b.day_of_year[permutation],
                s2_a.valid,
                relative_day=(
                    alpha * s2_a.relative_day
                    + (1 - alpha) * s2_b.relative_day[permutation]
                ),
            )
            mixed_s1 = TemporalView(
                alpha * s1_a.bands + (1 - alpha) * s1_b.bands[permutation],
                alpha * s1_a.day_of_year + (1 - alpha) * s1_b.day_of_year[permutation],
                s1_a.valid,
                relative_day=(
                    alpha * s1_a.relative_day
                    + (1 - alpha) * s1_b.relative_day[permutation]
                ),
            )
            mixed_s2_count = alpha * s2_count + (1 - alpha) * s2_count[permutation]
            mixed_s1_count = alpha * s1_count + (1 - alpha) * s1_count[permutation]
            mixed = _encode_views(
                self.model,
                mixed_s2,
                mixed_s1,
                duration,
                mixed_s2_count,
                mixed_s1_count,
            )
            mixup = mixup_consistency(first, second, mixed, permutation, alpha)
            loss = loss + self.mixup_weight * mixup
        return loss, mixup, terms

    def step(
        self, batch: ObservationBatch, generator: torch.Generator | None = None
    ) -> dict[str, float]:
        windowed = self.window_policy is not None
        batch.validate(require_window_metadata=windowed)
        _require_training_split(batch)
        self.model.train()
        window = (
            _window_for_batch(batch, self.window_policy, generator)
            if self.window_policy is not None
            else None
        )
        s2_a, s2_b, s1_a, s1_b = _views_for_both_modalities(
            batch,
            self.view_size,
            generator,
            window,
            (
                self.window_policy.view_dropout_probability
                if self.window_policy is not None
                else 0.0
            ),
        )
        if window is None:
            loss, mixup, terms = self._annual_step(
                batch, s2_a, s2_b, s1_a, s1_b, generator
            )
        else:
            loss, mixup, terms = self._window_step(
                batch, window, s2_a, s2_b, s1_a, s1_b, generator
            )

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)
        self.optimizer.step()
        metrics = {"loss": float(loss.detach()), "mixup": float(mixup.detach())}
        if window is not None:
            metrics["window_days"] = float(window.duration_days[0])
            available = s2_a.valid.any(dim=1) | s1_a.valid.any(dim=1)
            metrics["empty_window_fraction"] = float((~available).float().mean())
        return metrics | {key: float(value.detach()) for key, value in terms.items()}


def save_checkpoint_atomic(payload: dict[str, object], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(destination)
