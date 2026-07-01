"""Deterministic, non-scientific execution smoke for both adaptation tracks.

This module verifies that a checksum-pinned TESSERA v1.1 checkpoint can execute
one arbitrary-window Teacher-Student update and one arbitrary-window LoRA
update.  Its synthetic, normalized observations are deliberately unsuitable
for model-quality conclusions.  Acquisition and parity gates are bypassed only
because no acquired imagery participates in this smoke.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn

from .models.lora import LoRAConfig, inject_lora, trainable_parameter_count
from .models.student import RegionalStudent
from .models.tessera_v11 import TesseraV11, load_tessera_v11
from .models.windowed import WindowedTesseraEncoder
from .training import LoRATrainer, ObservationBatch, StudentTrainer
from .windows import WindowSamplingPolicy, day_of_year, epoch_day

SmokeScalar = str | int | float | bool
SmokeReport = dict[str, SmokeScalar | dict[str, SmokeScalar]]


def _canonical_device(requested: str | torch.device) -> torch.device:
    device = torch.device(requested)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("the model smoke supports only cpu and cuda devices")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("a CUDA device was requested but CUDA is unavailable")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index is out of range: {index}")
        return torch.device("cuda", index)
    return device


def _normalized_noise(
    shape: tuple[int, int, int], generator: torch.Generator
) -> torch.Tensor:
    values = torch.randn(shape, generator=generator, dtype=torch.float32)
    mean = values.mean(dim=(0, 1), keepdim=True)
    standard_deviation = values.std(dim=(0, 1), unbiased=False, keepdim=True)
    return (values - mean) / standard_deviation.clamp_min(1e-6)


def _synthetic_window_batch(
    duration_days: int,
    device: str | torch.device,
    *,
    batch_size: int = 4,
    seed: int = 0,
) -> ObservationBatch:
    """Create exact-duration, normalized timelines on deterministic dates."""
    if duration_days not in {7, 14}:
        raise ValueError("the execution smoke supports only 7-day and 14-day windows")
    if batch_size < 2:
        raise ValueError("the execution smoke needs at least two samples")
    target_device = _canonical_device(device)
    generator = torch.Generator(device="cpu").manual_seed(seed + duration_days)
    s2_bands = _normalized_noise(
        (batch_size, duration_days, 10), generator
    ).to(target_device)
    s1_bands = _normalized_noise(
        (batch_size, duration_days, 2), generator
    ).to(target_device)

    # These deliberately non-round dates make the smoke exercise arbitrary
    # intervals instead of accidentally encoding calendar-year assumptions.
    first_start = epoch_day("2021-02-03") + duration_days
    starts_cpu = torch.tensor(
        [first_start + row * 23 for row in range(batch_size)], dtype=torch.long
    )
    observation_days_cpu = starts_cpu[:, None] + torch.arange(duration_days)[None, :]
    day_of_year_cpu = torch.tensor(
        [day_of_year(value) for value in observation_days_cpu.flatten().tolist()],
        dtype=torch.long,
    ).reshape(batch_size, duration_days)
    observation_days = observation_days_cpu.to(target_device)
    days_of_year = day_of_year_cpu.to(target_device)
    s2_valid = torch.ones(
        (batch_size, duration_days), device=target_device, dtype=torch.bool
    )
    s1_valid = s2_valid.clone()
    s2_valid[:, -1] = False
    s1_valid[:, -1] = False
    if duration_days == 14:
        s2_valid[0, 2] = False
        s1_valid[-1, 3] = False
    starts = starts_cpu.to(target_device)

    batch = ObservationBatch(
        s2_bands=s2_bands,
        s2_day_of_year=days_of_year.clone(),
        s2_valid=s2_valid,
        s1_bands=s1_bands,
        s1_day_of_year=days_of_year.clone(),
        s1_valid=s1_valid,
        teacher_target=None,
        sample_ids=tuple(
            f"synthetic-{duration_days}d-{row}" for row in range(batch_size)
        ),
        spatial_blocks=tuple(
            f"synthetic-block-{duration_days}d-{row}" for row in range(batch_size)
        ),
        countries=tuple("RWA" if row % 2 == 0 else "ISR" for row in range(batch_size)),
        temporal_splits=("train",) * batch_size,
        s2_observation_day=observation_days.clone(),
        s1_observation_day=observation_days.clone(),
        coverage_start_day=starts,
        coverage_end_day=starts + duration_days,
    )
    batch.validate(require_window_metadata=True)
    return batch


def _fixed_window_policy(duration_days: int) -> WindowSamplingPolicy:
    return WindowSamplingPolicy(
        minimum_days=duration_days,
        maximum_days=duration_days,
        anchor_days=(duration_days,),
        anchor_probability=1.0,
        prefix_probability=0.0,
        view_dropout_probability=0.0,
    )


def _trainable_snapshots(model: nn.Module) -> tuple[tuple[str, torch.Tensor], ...]:
    return tuple(
        (name, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )


def _changed_tensor_count(
    model: nn.Module, before: tuple[tuple[str, torch.Tensor], ...]
) -> int:
    after = dict(model.named_parameters())
    return sum(
        not torch.equal(value, after[name].detach()) for name, value in before
    )


def _compact_metrics(
    metrics: dict[str, float], keys: tuple[str, ...]
) -> dict[str, float]:
    selected = {key: float(metrics[key]) for key in keys}
    if not all(math.isfinite(value) for value in selected.values()):
        raise RuntimeError(f"the model smoke produced non-finite metrics: {selected}")
    return selected


def _run_model_smoke_with_models(
    base_encoder: TesseraV11,
    student: nn.Module,
    *,
    device: str | torch.device = "cpu",
    seed: int = 20260701,
    batch_size: int = 4,
    lora_config: LoRAConfig | None = None,
    checkpoint_verified: bool = False,
) -> SmokeReport:
    """Internal dependency-injected path used by small CPU unit tests."""
    target_device = _canonical_device(device)
    if base_encoder.output_dim < 2:
        raise ValueError("the smoke requires an embedding width of at least two")
    if batch_size < 2:
        raise ValueError("the smoke requires at least two globally shuffled samples")
    lora_config = lora_config or LoRAConfig(rank=8, alpha=8.0)

    # Preserve the caller's RNG state while making view sampling and dropout
    # repeatable. The synthetic input itself is generated on CPU for identical
    # values across CPU and CUDA runs.
    cuda_devices = [target_device.index] if target_device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        base_encoder = base_encoder.to(target_device).eval()
        adapted_base = copy.deepcopy(base_encoder)
        reference = WindowedTesseraEncoder(base_encoder).eval()
        student = student.to(target_device)

        student_batch = _synthetic_window_batch(
            7, target_device, batch_size=batch_size, seed=seed
        )
        student_optimizer = torch.optim.AdamW(student.parameters(), lr=1e-4)
        student_before = _trainable_snapshots(student)
        student_metrics = StudentTrainer(
            student,
            student_optimizer,
            view_sizes=(4,),
            allow_unverified=True,
            window_policy=_fixed_window_policy(7),
            teacher_model=reference,
        ).step(
            student_batch,
            torch.Generator(device=target_device).manual_seed(seed + 1),
        )
        student_changed = _changed_tensor_count(student, student_before)
        if student_changed == 0:
            raise RuntimeError("the Teacher-Student optimizer step changed no parameters")

        adapted = WindowedTesseraEncoder(adapted_base, condition_windows=True)
        installed_targets = inject_lora(adapted, lora_config)
        adapted = adapted.to(target_device)
        lora_parameters = [
            parameter for parameter in adapted.parameters() if parameter.requires_grad
        ]
        if not lora_parameters:
            raise RuntimeError("LoRA injection produced no trainable parameters")
        lora_optimizer = torch.optim.AdamW(lora_parameters, lr=1e-4)
        lora_before = _trainable_snapshots(adapted)
        lora_metrics = LoRATrainer(
            adapted,
            lora_optimizer,
            view_size=4,
            allow_unverified=True,
            window_policy=_fixed_window_policy(14),
            reference_model=reference,
        ).step(
            _synthetic_window_batch(
                14, target_device, batch_size=batch_size, seed=seed
            ),
            torch.Generator(device=target_device).manual_seed(seed + 2),
        )
        lora_changed = _changed_tensor_count(adapted, lora_before)
        if lora_changed == 0:
            raise RuntimeError("the LoRA optimizer step changed no parameters")

    student_selected = _compact_metrics(
        student_metrics, ("loss", "window_days", "teacher_alignment")
    )
    lora_selected = _compact_metrics(
        lora_metrics, ("loss", "window_days", "base_anchor", "mixup")
    )
    return {
        "kind": "non-scientific-synthetic-execution-smoke",
        "checkpoint_verified": checkpoint_verified,
        "device": str(target_device),
        "seed": seed,
        "student": {
            **student_selected,
            "trainable_parameters": trainable_parameter_count(student),
            "updated_parameter_tensors": student_changed,
        },
        "lora": {
            **lora_selected,
            "trainable_parameters": trainable_parameter_count(adapted),
            "updated_parameter_tensors": lora_changed,
            "installed_targets": len(installed_targets),
        },
    }


def run_model_smoke(
    checkpoint_path: str | Path,
    expected_sha256: str,
    *,
    device: str | torch.device = "cpu",
    seed: int = 20260701,
    batch_size: int = 4,
) -> SmokeReport:
    """Run both synthetic optimizer smokes with a verified official encoder.

    ``load_tessera_v11`` verifies the SHA-256 and exact checkpoint graph before
    returning, so ``checkpoint_verified`` can only be true after both checks.
    """
    target_device = _canonical_device(device)
    cuda_devices = [target_device.index] if target_device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        base = load_tessera_v11(
            checkpoint_path, expected_sha256, device=target_device
        )
        student = RegionalStudent(output_dim=base.output_dim)
        return _run_model_smoke_with_models(
            base,
            student,
            device=target_device,
            seed=seed,
            batch_size=batch_size,
            checkpoint_verified=True,
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the non-scientific SpectraJam model execution smoke."
    )
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", default=20260701, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _build_parser().parse_args(argv)
    report = run_model_smoke(
        arguments.checkpoint,
        arguments.sha256,
        device=arguments.device,
        seed=arguments.seed,
        batch_size=arguments.batch_size,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised as a local module
    raise SystemExit(main())
