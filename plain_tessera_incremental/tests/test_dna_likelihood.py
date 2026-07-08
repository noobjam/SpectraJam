from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace

from plain_tessera_incremental.notebooks._dna_likelihood import (
    block_bootstrap_mosaic_share,
    calibrate_temperature,
    crop_log_scores,
    crop_posteriors,
    empirical_upper_tail_probability,
    field_balanced_weights,
    fit_crop_density,
    fit_mosaic_share,
    fit_parent_axis,
    fit_validated_crop_density,
    l2_normalize,
    with_temperature,
)
from plain_tessera_incremental.notebooks._dna_workflow import build_field_crosswalk


def test_field_balanced_weights_equalize_crops_fields_and_pixels() -> None:
    fields = np.array(["a", "a", "b", "c", "c", "c"])
    labels = np.array(["Bean", "Bean", "Bean", "Maize", "Maize", "Maize"])

    weights = field_balanced_weights(fields, labels, ("Bean", "Maize"))

    np.testing.assert_allclose(weights[labels == "Bean"].sum(), 0.5)
    np.testing.assert_allclose(weights[labels == "Maize"].sum(), 0.5)
    np.testing.assert_allclose(weights[fields == "a"].sum(), 0.25)
    np.testing.assert_allclose(weights[fields == "b"].sum(), 0.25)
    np.testing.assert_allclose(weights[fields == "c"].sum(), 0.5)


def _correlated_reference_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    dimensions = 128
    common = rng.normal(size=(dimensions, 8))
    bean_center = np.zeros(dimensions)
    maize_center = np.zeros(dimensions)
    bean_center[:8] = 0.35
    maize_center[:8] = -0.35
    rows = []
    labels = []
    fields = []
    for crop, center in (("Bean", bean_center), ("Maize", maize_center)):
        for field_index in range(8):
            field_shift = rng.normal(scale=0.025, size=dimensions)
            for _ in range(5):
                correlated_noise = common @ rng.normal(scale=0.012, size=8)
                rows.append(center + field_shift + correlated_noise)
                labels.append(crop)
                fields.append(f"{crop}-{field_index}")
    return l2_normalize(rows), np.asarray(labels), np.asarray(fields)


def test_shared_covariance_model_separates_correlated_references() -> None:
    features, labels, fields = _correlated_reference_fixture()
    weights = field_balanced_weights(fields, labels, ("Bean", "Maize"))

    model = fit_crop_density(
        features,
        labels,
        crop_names=("Bean", "Maize"),
        sample_weights=weights,
        shrinkage=0.25,
    )
    scores = crop_log_scores(model, features)
    probabilities = crop_posteriors(scores)
    predicted = np.asarray(model.crop_names)[np.argmax(probabilities, axis=1)]

    assert np.mean(predicted == labels) > 0.95
    assert model.component_count == 128
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_validated_model_keeps_fields_together_and_returns_oof_scores() -> None:
    features, labels, fields = _correlated_reference_fixture()

    validated = fit_validated_crop_density(
        features,
        labels,
        fields,
        crop_names=("Bean", "Maize"),
        shrinkage_candidates=(0.25, 0.5),
        max_folds=4,
        seed=9,
    )

    for field_id in set(fields):
        assert len(set(validated.sample_folds[fields == field_id])) == 1
    assert validated.oof_log_scores.shape == (len(features), 2)
    assert validated.fold_count == 4
    assert validated.field_balanced_accuracy > 0.9
    assert validated.chosen_shrinkage in {0.25, 0.5}


def test_parent_axis_uses_whitened_geometry_and_retains_residual() -> None:
    features, labels, fields = _correlated_reference_fixture()
    model = fit_crop_density(
        features,
        labels,
        crop_names=("Bean", "Maize"),
        sample_weights=field_balanced_weights(fields, labels, ("Bean", "Maize")),
    )
    midpoint = l2_normalize(
        [features[labels == "Bean"].mean(axis=0) + features[labels == "Maize"].mean(axis=0)]
    )

    fit = fit_parent_axis(model, midpoint, "Bean", "Maize")

    assert 0.35 < fit.parent_a_share[0] < 0.65
    assert fit.in_segment[0]
    assert fit.tube_squared_distance[0] >= 0
    assert fit.parent_squared_separation > 0


def test_mosaic_fit_recovers_pure_and_mixed_pixel_sources() -> None:
    pure_a = fit_mosaic_share([0.0, 0.0, 0.0], [-9.0, -8.0, -10.0])
    pure_b = fit_mosaic_share([-8.0, -10.0], [0.0, 0.0])
    mixed = fit_mosaic_share(
        [0.0, 0.0, 0.0, -9.0],
        [-9.0, -8.0, -10.0, 0.0],
        grid_size=401,
    )

    assert pure_a.parent_a_share == 1.0
    assert pure_b.parent_a_share == 0.0
    assert 0.65 < mixed.parent_a_share < 0.85
    assert mixed.log_evidence_over_pure > 0
    assert np.all((mixed.pixel_parent_a_probability >= 0) & (mixed.pixel_parent_a_probability <= 1))


def test_block_bootstrap_resamples_whole_blocks_deterministically() -> None:
    first = np.array([0.0, 0.0, -8.0, -8.0])
    second = np.array([-8.0, -8.0, 0.0, 0.0])
    blocks = np.array(["left", "left", "right", "right"])

    first_run = block_bootstrap_mosaic_share(
        first,
        second,
        blocks,
        replicates=20,
        seed=17,
    )
    second_run = block_bootstrap_mosaic_share(
        first,
        second,
        blocks,
        replicates=20,
        seed=17,
    )

    np.testing.assert_array_equal(first_run, second_run)
    assert set(first_run) <= {0.0, 0.5, 1.0}


def test_temperature_and_empirical_tail_are_explicit_evidence_calibrations() -> None:
    raw_scores = np.array([[8.0, 0.0], [0.0, 8.0], [4.0, 0.0], [0.0, 4.0]])
    labels = np.array(["Bean", "Maize", "Bean", "Maize"])
    temperature = calibrate_temperature(
        raw_scores,
        labels,
        crop_names=("Bean", "Maize"),
    )
    features, reference_labels, fields = _correlated_reference_fixture()
    model = fit_crop_density(
        features,
        reference_labels,
        crop_names=("Bean", "Maize"),
        sample_weights=field_balanced_weights(
            fields,
            reference_labels,
            ("Bean", "Maize"),
        ),
    )

    assert 0 < temperature <= 8
    assert with_temperature(model, temperature).temperature == temperature
    assert empirical_upper_tail_probability(3.0, [1.0, 2.0, 3.0, 4.0]) == pytest.approx(0.6)


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (l2_normalize, ([[0.0, 0.0]],), "zero-norm"),
        (
            field_balanced_weights,
            (["field"], ["Bean"], ("Bean", "Maize")),
            "missing fields",
        ),
        (fit_mosaic_share, ([0.0], [0.0, 1.0]), "equally sized"),
        (empirical_upper_tail_probability, (1.0, []), "nonempty"),
    ],
)
def test_invalid_inputs_fail_loudly(function, args, message) -> None:
    with pytest.raises(ValueError, match=message):
        function(*args)


def test_field_crosswalk_keeps_every_record_but_collapses_only_exact_label_replicas() -> None:
    fields = pd.DataFrame(
        {
            "field_uid": ["a", "b", "c", "d", "e"],
            "id": [1, 1, 2, 3, 4],
            "landcover": ["Bean", "Bean", "Maize", "Bean", "Other"],
            "wkt": ["A", "A", "A", "D", "E"],
            "utm_epsg": [32735] * 5,
            "pixel_count": [2] * 5,
            "geometry_sha256": ["same", "same", "same", "unique", "other"],
        }
    )

    crosswalk, units = build_field_crosswalk(
        SimpleNamespace(fields=fields),
        ("Bean", "Maize"),
    )

    assert len(crosswalk) == len(fields)
    assert crosswalk.set_index("field_uid").loc["b", "analysis_unit_uid"] == "a"
    assert crosswalk.set_index("field_uid").loc["b", "record_role"] == (
        "conflicting_label_replica"
    )
    assert crosswalk.set_index("field_uid").loc["e", "record_role"] == (
        "unsupported_label"
    )
    assert set(units["analysis_unit_uid"]) == {"d"}
