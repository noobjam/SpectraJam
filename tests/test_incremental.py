import pytest

torch = pytest.importorskip("torch")

from spectrajam.incremental import (
    EncoderIdentity,
    ExactIncrementalEmbeddingBuilder,
    MemoryEmbeddingCache,
    TorchWindowEncoder,
)
from spectrajam.models.student import RegionalStudent
from spectrajam.windows import TemporalSeries, TemporalWindow, day_of_year


def _series(days: list[int], channels: int, prefix: str) -> TemporalSeries:
    return TemporalSeries(
        bands=torch.stack(
            [torch.full((channels,), float(index + 1)) for index in range(len(days))]
        )
        if days
        else torch.empty((0, channels)),
        observation_day=torch.tensor(days, dtype=torch.long),
        day_of_year=torch.tensor([day_of_year(value) for value in days], dtype=torch.long),
        valid=torch.ones(len(days), dtype=torch.bool),
        observation_ids=tuple(f"{prefix}-{index}" for index in range(len(days))),
    )


class CountingEncoder:
    def __init__(self):
        self.calls = 0

    def encode(self, prepared):
        self.calls += 1
        return torch.tensor(
            [
                prepared.window.duration_days,
                prepared.s2.bands.sum(),
                prepared.s1.bands.sum(),
            ],
            dtype=torch.float32,
        )


def _identity(model: str = "a") -> EncoderIdentity:
    return EncoderIdentity(
        model_sha256=model * 64,
        preprocessing_sha256="b" * 64,
        provider_profile="mpc-v1.1",
    )


def test_identical_window_is_an_exact_cache_hit() -> None:
    encoder = CountingEncoder()
    cache = MemoryEmbeddingCache()
    builder = ExactIncrementalEmbeddingBuilder(
        encoder,
        _identity(),
        _series([100, 105], 10, "s2"),
        _series([101, 104], 2, "s1"),
        cache,
    )
    window = TemporalWindow(100, 107, "rolling")
    first = builder.embed(window)
    second = builder.embed(window)
    assert not first.cache_hit
    assert second.cache_hit
    assert torch.equal(first.embedding, second.embedding)
    assert first.cache_key == second.cache_key
    assert first.s2_observation_count == 2
    assert first.s1_observation_count == 2
    assert encoder.calls == 1
    assert len(cache) == 1


def test_late_observation_invalidates_only_windows_it_enters() -> None:
    encoder = CountingEncoder()
    builder = ExactIncrementalEmbeddingBuilder(
        encoder,
        _identity(),
        _series([100], 10, "s2"),
        _series([100], 2, "s1"),
    )
    early = TemporalWindow(100, 107, "rolling")
    late = TemporalWindow(107, 114, "rolling")
    builder.embed(early)
    assert builder.embed(late).empty_window
    builder.upsert("s2", _series([110], 10, "late-s2"))
    assert builder.embed(early).cache_hit
    assert not builder.embed(late).cache_hit
    assert encoder.calls == 2


def test_model_identity_is_part_of_the_cache_key() -> None:
    cache = MemoryEmbeddingCache()
    series_s2 = _series([100], 10, "s2")
    series_s1 = _series([100], 2, "s1")
    window = TemporalWindow(100, 107)
    first_encoder = CountingEncoder()
    second_encoder = CountingEncoder()
    first = ExactIncrementalEmbeddingBuilder(
        first_encoder, _identity("a"), series_s2, series_s1, cache
    ).embed(window)
    second = ExactIncrementalEmbeddingBuilder(
        second_encoder, _identity("c"), series_s2, series_s1, cache
    ).embed(window)
    assert first.cache_key != second.cache_key
    assert not second.cache_hit
    assert second_encoder.calls == 1


def test_empty_window_returns_no_data_instead_of_a_fake_embedding() -> None:
    encoder = CountingEncoder()
    builder = ExactIncrementalEmbeddingBuilder(
        encoder,
        _identity(),
        _series([100], 10, "s2"),
        _series([100], 2, "s1"),
    )
    result = builder.embed(TemporalWindow(200, 207))
    assert result.empty_window
    assert result.embedding is None
    assert result.s2_observation_count == 0
    assert result.s1_observation_count == 0
    assert not result.cache_hit
    assert encoder.calls == 0


def test_torch_builder_matches_fresh_full_window_recomputation() -> None:
    model = RegionalStudent(
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        output_dim=8,
        dropout=0,
    ).eval()
    encoder = TorchWindowEncoder(model)
    builder = ExactIncrementalEmbeddingBuilder(
        encoder,
        _identity(),
        _series([100, 103, 106], 10, "s2"),
        _series([101, 105], 2, "s1"),
    )
    window = TemporalWindow(100, 107, "rolling")
    result = builder.embed(window)
    fresh = encoder.encode(builder.prepare(window))
    assert result.embedding is not None
    assert torch.equal(result.embedding, fresh)

    builder.upsert("s1", _series([104], 2, "new-s1"))
    updated = builder.embed(window)
    updated_fresh = encoder.encode(builder.prepare(window))
    assert not updated.cache_hit
    assert updated.embedding is not None
    assert torch.equal(updated.embedding, updated_fresh)


def test_torch_encoder_rejects_weight_mutation_after_cache_identity_is_bound() -> None:
    model = RegionalStudent(
        model_dim=16,
        layers=1,
        heads=4,
        feedforward_dim=32,
        output_dim=8,
        dropout=0,
    )
    encoder = TorchWindowEncoder(model)
    builder = ExactIncrementalEmbeddingBuilder(
        encoder,
        _identity(),
        _series([100], 10, "s2"),
        _series([100], 2, "s1"),
    )
    builder.embed(TemporalWindow(100, 107))
    with torch.no_grad():
        next(model.parameters()).add_(1)
    with pytest.raises(RuntimeError, match="encoder weights changed"):
        builder.embed(TemporalWindow(100, 107))


def test_bfloat16_prepared_inputs_have_a_stable_fingerprint() -> None:
    s2 = _series([100], 10, "s2")
    s2 = TemporalSeries(
        bands=s2.bands.to(torch.bfloat16),
        observation_day=s2.observation_day,
        day_of_year=s2.day_of_year,
        valid=s2.valid,
        observation_ids=s2.observation_ids,
    )
    builder = ExactIncrementalEmbeddingBuilder(
        CountingEncoder(), _identity(), s2, _series([100], 2, "s1")
    )
    assert len(builder.prepare(TemporalWindow(100, 107)).fingerprint()) == 64
