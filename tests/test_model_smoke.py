import json

import pytest

torch = pytest.importorskip("torch")

from spectrajam import model_smoke
from spectrajam.models.lora import LoRAConfig
from spectrajam.models.student import RegionalStudent
from spectrajam.models.tessera_v11 import TesseraV11


def _tiny_models() -> tuple[TesseraV11, RegionalStudent]:
    torch.manual_seed(41)
    base = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    student = RegionalStudent(
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        output_dim=8,
        dropout=0.0,
    )
    return base, student


@pytest.mark.parametrize("duration", [7, 14])
def test_synthetic_window_batch_is_deterministic_normalized_and_exact(duration) -> None:
    first = model_smoke._synthetic_window_batch(
        duration, "cpu", batch_size=4, seed=19
    )
    second = model_smoke._synthetic_window_batch(
        duration, "cpu", batch_size=4, seed=19
    )

    assert torch.equal(first.s2_bands, second.s2_bands)
    assert torch.equal(first.s1_bands, second.s1_bands)
    assert torch.equal(
        first.coverage_end_day - first.coverage_start_day,
        torch.full((4,), duration),
    )
    assert first.s2_bands.shape == (4, duration, 10)
    assert first.s1_bands.shape == (4, duration, 2)
    assert not bool(first.s2_valid.all())
    assert not bool(first.s1_valid.all())
    assert torch.allclose(first.s2_bands.mean((0, 1)), torch.zeros(10), atol=1e-6)
    assert torch.allclose(
        first.s2_bands.std((0, 1), unbiased=False), torch.ones(10), atol=1e-6
    )
    assert torch.allclose(first.s1_bands.mean((0, 1)), torch.zeros(2), atol=1e-6)
    assert torch.allclose(
        first.s1_bands.std((0, 1), unbiased=False), torch.ones(2), atol=1e-6
    )
    assert set(first.countries) == {"RWA", "ISR"}
    first.validate(require_window_metadata=True)


def test_injected_tiny_models_complete_both_optimizer_smokes() -> None:
    base, student = _tiny_models()
    report = model_smoke._run_model_smoke_with_models(
        base,
        student,
        device="cpu",
        seed=23,
        batch_size=4,
        lora_config=LoRAConfig(rank=2, alpha=2.0),
    )

    assert report["kind"] == "non-scientific-synthetic-execution-smoke"
    assert report["checkpoint_verified"] is False
    assert report["device"] == "cpu"
    student_report = report["student"]
    lora_report = report["lora"]
    assert isinstance(student_report, dict)
    assert isinstance(lora_report, dict)
    assert student_report["window_days"] == 7
    assert lora_report["window_days"] == 14
    assert student_report["loss"] > 0
    assert lora_report["loss"] > 0
    assert student_report["trainable_parameters"] > 0
    assert lora_report["trainable_parameters"] > 0
    assert student_report["updated_parameter_tensors"] > 0
    assert lora_report["updated_parameter_tensors"] > 0
    assert lora_report["installed_targets"] > 0


def test_module_main_prints_json_report(monkeypatch, capsys, tmp_path) -> None:
    expected = {
        "kind": "non-scientific-synthetic-execution-smoke",
        "checkpoint_verified": True,
    }
    captured = {}

    def fake_run(checkpoint, sha256, *, device, seed, batch_size):
        captured.update(
            checkpoint=checkpoint,
            sha256=sha256,
            device=device,
            seed=seed,
            batch_size=batch_size,
        )
        return expected

    monkeypatch.setattr(model_smoke, "run_model_smoke", fake_run)
    checkpoint = tmp_path / "tessera.pt"
    result = model_smoke.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--sha256",
            "a" * 64,
            "--device",
            "cpu",
            "--seed",
            "11",
            "--batch-size",
            "3",
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert captured == {
        "checkpoint": checkpoint,
        "sha256": "a" * 64,
        "device": "cpu",
        "seed": 11,
        "batch_size": 3,
    }
