import pytest

torch = pytest.importorskip("torch")

from spectrajam.losses import (
    cosine_alignment,
    lora_objective,
    mixup_consistency,
    relational_geometry,
    student_objective,
)
from spectrajam.models.lora import (
    LoRAConfig,
    adapter_state_dict,
    inject_lora,
    load_adapter,
    merge_lora,
    save_adapter,
    trainable_parameter_count,
)
from spectrajam.models.student import RegionalStudent
from spectrajam.models.tessera_v11 import TesseraV11
from spectrajam.models.windowed import WindowedTesseraEncoder
from spectrajam.training import LoRATrainer, ObservationBatch, StudentTrainer
from spectrajam.views import paired_temporal_views
from spectrajam.windows import WindowSamplingPolicy


def test_zero_lora_is_exactly_base_equivalent() -> None:
    torch.manual_seed(1)
    model = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).eval()
    s2 = torch.randn(2, 4, 11)
    s1 = torch.randn(2, 4, 3)
    before = model(s2, s1).detach()
    installed = inject_lora(model, LoRAConfig(rank=2, alpha=2, target="attention_ffn"))
    after = model(s2, s1).detach()
    assert installed
    assert torch.equal(before, after)
    assert trainable_parameter_count(model) > 0
    assert adapter_state_dict(model)
    merge_lora(model)
    assert torch.allclose(before, model(s2, s1).detach(), atol=1e-6, rtol=0)


def test_lora_parameters_inherit_base_device_and_dtype() -> None:
    model = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).to(dtype=torch.float64)
    inject_lora(model, LoRAConfig(rank=2, alpha=2, target="attention_ffn"))
    adapter_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if "parametrizations" in name and not name.endswith(".original")
    ]
    assert adapter_parameters
    assert all(parameter.dtype == torch.float64 for parameter in adapter_parameters)
    base_device = next(model.parameters()).device
    assert all(parameter.device == base_device for parameter in adapter_parameters)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_lora_can_be_injected_into_a_cuda_resident_base() -> None:
    model = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).to("cuda")
    inject_lora(model, LoRAConfig(rank=2, alpha=2))
    output = model(
        torch.randn(2, 4, 11, device="cuda"),
        torch.randn(2, 4, 3, device="cuda"),
    )
    assert output.device.type == "cuda"


def test_student_handles_missing_modality_and_has_expected_shape() -> None:
    model = RegionalStudent(model_dim=32, layers=1, heads=4, feedforward_dim=64)
    batch, steps = 3, 5
    output = model(
        torch.randn(batch, steps, 10),
        torch.randint(1, 366, (batch, steps)),
        torch.ones(batch, steps, dtype=torch.bool),
        torch.randn(batch, steps, 2),
        torch.randint(1, 366, (batch, steps)),
        torch.zeros(batch, steps, dtype=torch.bool),
    )
    assert output.shape == (batch, 128)
    assert torch.isfinite(output).all()


def test_student_returns_zero_when_both_modalities_are_empty() -> None:
    model = RegionalStudent(
        model_dim=16, layers=1, heads=4, feedforward_dim=32, output_dim=8
    ).eval()
    output = model(
        torch.randn(2, 3, 10),
        torch.ones(2, 3, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.bool),
        torch.randn(2, 3, 2),
        torch.ones(2, 3, dtype=torch.long),
        torch.zeros(2, 3, dtype=torch.bool),
        window_duration_days=torch.tensor([7, 14]),
    )
    assert torch.equal(output, torch.zeros_like(output))


def test_student_rejects_interspersed_padding() -> None:
    model = RegionalStudent(
        model_dim=16, layers=1, heads=4, feedforward_dim=32, output_dim=8
    )
    with pytest.raises(ValueError, match="left-packed"):
        model(
            torch.randn(2, 3, 10),
            torch.ones(2, 3, dtype=torch.long),
            torch.tensor([[True, False, True], [True, True, True]]),
            torch.randn(2, 3, 2),
            torch.ones(2, 3, dtype=torch.long),
            torch.ones(2, 3, dtype=torch.bool),
        )


def test_representation_objectives_are_finite_and_differentiable() -> None:
    teacher = torch.randn(8, 16)
    first = torch.randn(8, 16, requires_grad=True)
    second = torch.randn(8, 16, requires_grad=True)
    student_loss, student_terms = student_objective(first, second, teacher)
    adapter_loss, adapter_terms = lora_objective(first, second, teacher)
    total = student_loss + adapter_loss
    total.backward()
    assert torch.isfinite(total)
    assert first.grad is not None
    assert set(student_terms) == {
        "teacher_alignment",
        "relational_geometry",
        "temporal_barlow",
        "variance",
        "covariance",
    }
    assert set(adapter_terms) == {"base_anchor", "temporal_barlow", "relational_geometry"}


def test_teacher_confidence_is_an_absolute_loss_gate() -> None:
    student = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    teacher = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    confidence = torch.full((2,), 0.1)
    assert cosine_alignment(student, teacher, confidence) == pytest.approx(
        0.1 * cosine_alignment(student, teacher)
    )
    assert relational_geometry(student, teacher, confidence) == pytest.approx(
        0.1 * relational_geometry(student, teacher)
    )


def test_temporal_views_preserve_size_and_chronology() -> None:
    bands = torch.randn(2, 6, 10)
    day = torch.tensor([[30, 10, 50, 20, 40, 60], [1, 2, 3, 4, 5, 6]])
    valid = torch.tensor(
        [[True, True, True, True, True, True], [True, True, False, False, False, False]]
    )
    first, second = paired_temporal_views(
        bands, day, valid, target_size=4, generator=torch.Generator().manual_seed(4)
    )
    assert first.bands.shape == second.bands.shape == (2, 4, 10)
    for row in range(first.valid.shape[0]):
        selected = first.day_of_year[row, first.valid[row]]
        assert torch.all(selected[1:] >= selected[:-1])
    assert first.valid.sum(dim=1).tolist() == [4, 2]


def test_mask_aware_tessera_is_dense_base_equivalent() -> None:
    torch.manual_seed(3)
    base = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).eval()
    wrapper = WindowedTesseraEncoder(base).eval()
    s2_bands = torch.randn(3, 4, 10)
    s1_bands = torch.randn(3, 4, 2)
    s2_doy = torch.randint(1, 366, (3, 4))
    s1_doy = torch.randint(1, 366, (3, 4))
    expected = base(
        torch.cat([s2_bands, s2_doy.unsqueeze(-1).to(s2_bands)], dim=-1),
        torch.cat([s1_bands, s1_doy.unsqueeze(-1).to(s1_bands)], dim=-1),
    )
    actual = wrapper(
        s2_bands,
        s2_doy,
        torch.ones_like(s2_doy, dtype=torch.bool),
        s1_bands,
        s1_doy,
        torch.ones_like(s1_doy, dtype=torch.bool),
    )
    assert torch.equal(expected, actual)


def test_mask_aware_tessera_handles_empty_modalities() -> None:
    base = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).eval()
    wrapper = WindowedTesseraEncoder(base).eval()
    output = wrapper(
        torch.randn(2, 3, 10),
        torch.ones(2, 3, dtype=torch.long),
        torch.tensor([[True, False, False], [False, False, False]]),
        torch.randn(2, 3, 2),
        torch.ones(2, 3, dtype=torch.long),
        torch.tensor([[False, False, False], [False, False, False]]),
    )
    assert torch.isfinite(output).all()
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_mask_aware_tessera_matches_manual_compaction_and_ignores_padding() -> None:
    base = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    ).eval()
    wrapper = WindowedTesseraEncoder(base).eval()
    s2 = torch.randn(1, 4, 10)
    s1 = torch.randn(1, 4, 2)
    doy = torch.tensor([[10, 20, 30, 40]])
    valid = torch.tensor([[True, True, False, False]])
    padded = wrapper(s2, doy, valid, s1, doy, valid)
    s2[:, 2:] = 1e6
    s1[:, 2:] = -1e6
    ignored_padding = wrapper(s2, doy, valid, s1, doy, valid)
    compact = wrapper(
        s2[:, :2],
        doy[:, :2],
        torch.ones(1, 2, dtype=torch.bool),
        s1[:, :2],
        doy[:, :2],
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert torch.equal(padded, ignored_padding)
    assert torch.equal(padded, compact)


def test_mixup_consistency_is_finite() -> None:
    first = torch.randn(8, 16)
    second = torch.randn(8, 16)
    permutation = torch.randperm(8)
    mixed = 0.4 * first + 0.6 * second[permutation]
    loss = mixup_consistency(first, second, mixed, permutation, alpha=0.4)
    assert torch.isfinite(loss)


def _annual_batch(embedding_dim: int) -> ObservationBatch:
    batch, steps = 4, 6
    return ObservationBatch(
        s2_bands=torch.randn(batch, steps, 10),
        s2_day_of_year=torch.arange(1, steps + 1).repeat(batch, 1),
        s2_valid=torch.ones(batch, steps, dtype=torch.bool),
        s1_bands=torch.randn(batch, steps, 2),
        s1_day_of_year=torch.arange(1, steps + 1).repeat(batch, 1),
        s1_valid=torch.ones(batch, steps, dtype=torch.bool),
        teacher_target=torch.randn(batch, embedding_dim),
        sample_ids=tuple(f"sample-{index}" for index in range(batch)),
        spatial_blocks=tuple(f"block-{index}" for index in range(batch)),
        countries=("RWA", "ISR", "RWA", "ISR"),
        temporal_splits=("train",) * batch,
    )


def _window_batch() -> ObservationBatch:
    batch = _annual_batch(8)
    observation_days = torch.arange(100, 106).repeat(4, 1)
    batch.teacher_target = None
    batch.s2_observation_day = observation_days
    batch.s1_observation_day = observation_days.clone()
    batch.coverage_start_day = torch.full((4,), 100)
    batch.coverage_end_day = torch.full((4,), 107)
    return batch


class _RecordingTeacher(torch.nn.Module):
    def __init__(self, output_dim: int = 8):
        super().__init__()
        self.output_dim = output_dim
        self.last_s2_valid = None
        self.last_s1_valid = None

    def forward(
        self,
        s2_bands,
        s2_day_of_year,
        s2_valid,
        s1_bands,
        s1_day_of_year,
        s1_valid,
        **kwargs,
    ):
        del s2_day_of_year, s1_day_of_year, kwargs
        self.last_s2_valid = s2_valid.detach().clone()
        self.last_s1_valid = s1_valid.detach().clone()
        total = (s2_bands * s2_valid[..., None]).sum(dim=(1, 2))
        total = total + (s1_bands * s1_valid[..., None]).sum(dim=(1, 2))
        return total[:, None].repeat(1, self.output_dim)


class _CountRecordingStudent(RegionalStudent):
    def forward(self, *args, **kwargs):
        self.last_s2_count = kwargs.get("s2_observation_count")
        self.last_s1_count = kwargs.get("s1_observation_count")
        return super().forward(*args, **kwargs)


def test_student_training_step_updates_a_real_model() -> None:
    model = RegionalStudent(
        model_dim=16, layers=1, heads=4, feedforward_dim=32, output_dim=8
    )
    trainer = StudentTrainer(
        model,
        torch.optim.AdamW(model.parameters(), lr=1e-3),
        view_sizes=(4,),
        allow_unverified=True,
    )
    metrics = trainer.step(_annual_batch(8), torch.Generator().manual_seed(1))
    assert metrics["view_size"] == 4
    assert metrics["loss"] > 0


def test_lora_training_step_updates_only_adapters() -> None:
    base = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    batch = _annual_batch(8)
    dense_s2 = torch.cat(
        [batch.s2_bands, batch.s2_day_of_year.unsqueeze(-1).to(batch.s2_bands)], dim=-1
    )
    dense_s1 = torch.cat(
        [batch.s1_bands, batch.s1_day_of_year.unsqueeze(-1).to(batch.s1_bands)], dim=-1
    )
    base.eval()
    batch.teacher_target = base(dense_s2, dense_s1).detach()
    inject_lora(base, LoRAConfig(rank=2, alpha=2))
    model = WindowedTesseraEncoder(base)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )
    trainer = LoRATrainer(model, optimizer, view_size=4, allow_unverified=True)
    metrics = trainer.step(batch, torch.Generator().manual_seed(2))
    assert metrics["loss"] > 0
    assert metrics["mixup"] >= 0


def test_annual_lora_masks_sparse_padding_instead_of_encoding_it() -> None:
    arguments = dict(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    adapted_base = TesseraV11(**arguments)
    reference_base = TesseraV11(**arguments)
    reference_base.load_state_dict(adapted_base.state_dict())
    batch = _annual_batch(8)
    batch.s2_valid[:, 2:] = False
    batch.s1_valid[:, 2:] = False
    reference = WindowedTesseraEncoder(reference_base).eval()
    with torch.inference_mode():
        batch.teacher_target = reference(
            batch.s2_bands,
            batch.s2_day_of_year,
            batch.s2_valid,
            batch.s1_bands,
            batch.s1_day_of_year,
            batch.s1_valid,
            window_duration_days=torch.full((4,), 365),
            s2_observation_count=batch.s2_valid.sum(dim=1),
            s1_observation_count=batch.s1_valid.sum(dim=1),
        )
    inject_lora(adapted_base, LoRAConfig(rank=2, alpha=2))
    adapted = WindowedTesseraEncoder(adapted_base)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    metrics = LoRATrainer(
        adapted, optimizer, view_size=4, allow_unverified=True
    ).step(batch, torch.Generator().manual_seed(5))
    assert metrics["loss"] > 0
    assert metrics["mixup"] == 0


def test_student_trains_on_same_seven_day_teacher_window() -> None:
    student = _CountRecordingStudent(
        model_dim=16, layers=1, heads=4, feedforward_dim=32, output_dim=8
    )
    teacher = WindowedTesseraEncoder(
        TesseraV11(
            latent_dim=4,
            representation_dim=8,
            output_dim=8,
            num_heads=2,
            num_layers=1,
            dim_feedforward=16,
        )
    )
    policy = WindowSamplingPolicy(
        minimum_days=7,
        maximum_days=7,
        anchor_days=(7,),
        anchor_probability=1.0,
        prefix_probability=1.0,
    )
    trainer = StudentTrainer(
        student,
        torch.optim.AdamW(student.parameters(), lr=1e-3),
        view_sizes=(4,),
        allow_unverified=True,
        window_policy=policy,
        teacher_model=teacher,
    )
    metrics = trainer.step(_window_batch(), torch.Generator().manual_seed(12))
    assert metrics["window_days"] == 7
    assert metrics["teacher_anchor_weight"] == pytest.approx((7 / 90) * (6 / 10))
    assert metrics["loss"] > 0
    assert student.last_s2_count.tolist() == [6, 6, 6, 6]
    assert student.last_s1_count.tolist() == [6, 6, 6, 6]


def test_window_trainer_excludes_the_cutoff_observation_end_to_end() -> None:
    batch = _window_batch()
    batch.s2_observation_day[:, -1] = 107
    batch.s1_observation_day[:, -1] = 107
    batch.s2_bands[:, -1] = 1e6
    batch.s1_bands[:, -1] = -1e6
    teacher = _RecordingTeacher()
    student = RegionalStudent(
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        output_dim=8,
        dropout=0,
    )
    policy = WindowSamplingPolicy(
        minimum_days=7,
        maximum_days=7,
        anchor_days=(7,),
        anchor_probability=1,
        prefix_probability=1,
        view_dropout_probability=0,
    )
    trainer = StudentTrainer(
        student,
        torch.optim.AdamW(student.parameters(), lr=1e-3),
        view_sizes=(6,),
        allow_unverified=True,
        window_policy=policy,
        teacher_model=teacher,
    )
    trainer.step(batch, torch.Generator().manual_seed(2))
    assert not teacher.last_s2_valid[:, -1].any()
    assert not teacher.last_s1_valid[:, -1].any()


def test_window_filtering_rechecks_spatial_block_diversity() -> None:
    batch = _window_batch()
    batch.spatial_blocks = ("same", "same", "other-a", "other-b")
    batch.s2_valid[2:] = False
    batch.s1_valid[2:] = False
    teacher = _RecordingTeacher()
    student = RegionalStudent(
        model_dim=16, layers=1, heads=4, feedforward_dim=32, output_dim=8
    )
    policy = WindowSamplingPolicy(
        minimum_days=7,
        maximum_days=7,
        anchor_days=(7,),
        anchor_probability=1,
        prefix_probability=1,
        view_dropout_probability=0,
    )
    trainer = StudentTrainer(
        student,
        torch.optim.AdamW(student.parameters(), lr=1e-3),
        view_sizes=(4,),
        allow_unverified=True,
        window_policy=policy,
        teacher_model=teacher,
    )
    with pytest.raises(ValueError, match="spatial blocks"):
        trainer.step(batch, torch.Generator().manual_seed(3))


def test_window_training_rejects_precomputed_annual_target() -> None:
    batch = _window_batch()
    batch.teacher_target = torch.randn(4, 8)
    with pytest.raises(ValueError, match="rejects precomputed"):
        batch.validate(require_window_metadata=True)


def test_lora_trains_on_same_seven_day_reference_window() -> None:
    arguments = dict(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    torch.manual_seed(7)
    adapted_base = TesseraV11(**arguments)
    reference_base = TesseraV11(**arguments)
    reference_base.load_state_dict(adapted_base.state_dict())
    inject_lora(adapted_base, LoRAConfig(rank=2, alpha=2))
    adapted = WindowedTesseraEncoder(adapted_base, condition_windows=True)
    reference = WindowedTesseraEncoder(reference_base)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in adapted.parameters() if parameter.requires_grad],
        lr=1e-3,
    )
    policy = WindowSamplingPolicy(
        minimum_days=7,
        maximum_days=7,
        anchor_days=(7,),
        anchor_probability=1.0,
        prefix_probability=1.0,
    )
    trainer = LoRATrainer(
        adapted,
        optimizer,
        view_size=4,
        allow_unverified=True,
        window_policy=policy,
        reference_model=reference,
    )
    metrics = trainer.step(_window_batch(), torch.Generator().manual_seed(13))
    assert metrics["window_days"] == 7
    assert metrics["loss"] > 0


def test_trainers_require_verification_gates_by_default() -> None:
    model = RegionalStudent(
        model_dim=16, layers=1, heads=4, feedforward_dim=32, output_dim=8
    )
    with pytest.raises(RuntimeError, match="requires acquisition"):
        StudentTrainer(model, torch.optim.AdamW(model.parameters()))


def test_adapter_round_trip_preserves_metadata_and_outputs(tmp_path) -> None:
    base_args = dict(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    torch.manual_seed(9)
    source = TesseraV11(**base_args).eval()
    base_state = {key: value.clone() for key, value in source.state_dict().items()}
    config = LoRAConfig(rank=2, alpha=3, target="attention")
    inject_lora(source, config)
    for name, parameter in source.named_parameters():
        if "q_B" in name or "v_B" in name or "lora_B" in name:
            parameter.data.normal_(std=0.01)
    path = tmp_path / "adapter.pt"
    save_adapter(path, source, config, "a" * 64, "mpc-v1.1", "RWA")

    target = TesseraV11(**base_args).eval()
    target.load_state_dict(base_state)
    loaded = load_adapter(path, target, "a" * 64, "mpc-v1.1", "RWA")
    s2 = torch.randn(2, 4, 11)
    s1 = torch.randn(2, 4, 3)
    assert loaded == config
    assert torch.allclose(source(s2, s1), target(s2, s1))
    with pytest.raises(Exception, match="pristine base model"):
        load_adapter(path, target, "a" * 64, "mpc-v1.1", "RWA")


def test_adapter_save_rejects_metadata_that_changes_installed_scale(tmp_path) -> None:
    model = TesseraV11(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    inject_lora(model, LoRAConfig(rank=2, alpha=2))
    with pytest.raises(Exception, match="does not match installed"):
        save_adapter(
            tmp_path / "wrong.pt",
            model,
            LoRAConfig(rank=2, alpha=8),
            "a" * 64,
            "mpc-v1.1",
            "RWA",
        )


def test_failed_adapter_load_leaves_model_pristine_for_retry(tmp_path) -> None:
    arguments = dict(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    source = TesseraV11(**arguments)
    config = LoRAConfig(rank=2, alpha=2)
    inject_lora(source, config)
    valid_path = tmp_path / "valid.pt"
    save_adapter(valid_path, source, config, "a" * 64, "mpc-v1.1", "RWA")
    payload = torch.load(valid_path, map_location="cpu", weights_only=True)
    payload["adapter_state"].pop(next(iter(payload["adapter_state"])))
    corrupt_path = tmp_path / "corrupt.pt"
    torch.save(payload, corrupt_path)

    target = TesseraV11(**arguments)
    with pytest.raises(Exception, match="parameter mismatch"):
        load_adapter(corrupt_path, target, "a" * 64, "mpc-v1.1", "RWA")
    loaded = load_adapter(valid_path, target, "a" * 64, "mpc-v1.1", "RWA")
    assert loaded == config


def test_conditioned_window_adapter_round_trip(tmp_path) -> None:
    arguments = dict(
        latent_dim=4,
        representation_dim=8,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        dim_feedforward=16,
    )
    torch.manual_seed(19)
    source_base = TesseraV11(**arguments)
    base_state = {key: value.clone() for key, value in source_base.state_dict().items()}
    source = WindowedTesseraEncoder(source_base, condition_windows=True).eval()
    config = LoRAConfig(rank=2, alpha=2)
    inject_lora(source, config)
    for name, parameter in source.named_parameters():
        if "q_B" in name or "v_B" in name or "window_conditioner.2" in name:
            parameter.data.normal_(std=0.01)
    path = tmp_path / "conditioned-adapter.pt"
    save_adapter(path, source, config, "d" * 64, "mpc-v1.1", "ISR")

    target_base = TesseraV11(**arguments)
    target_base.load_state_dict(base_state)
    target = WindowedTesseraEncoder(target_base, condition_windows=True).eval()
    load_adapter(path, target, "d" * 64, "mpc-v1.1", "ISR")
    s2_bands = torch.randn(2, 3, 10)
    s1_bands = torch.randn(2, 3, 2)
    doy = torch.tensor([[1, 2, 3], [10, 11, 12]])
    valid = torch.tensor([[True, True, False], [True, False, False]])
    arguments_forward = (
        s2_bands,
        doy,
        valid,
        s1_bands,
        doy,
        valid,
    )
    duration = torch.tensor([7, 14])
    assert torch.allclose(
        source(*arguments_forward, window_duration_days=duration),
        target(*arguments_forward, window_duration_days=duration),
    )
