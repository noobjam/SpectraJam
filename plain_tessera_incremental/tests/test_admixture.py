from __future__ import annotations

import numpy as np
import pytest

from plain_tessera_incremental.notebooks._admixture import (
    dual_parent_evidence,
    solve_parent_segment,
    solve_simplex_admixture,
)


def test_dual_parent_evidence_requires_both_parents_and_named_mass() -> None:
    weights = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.25, 0.25, 0.25, 0.25],
            [0.0, 0.0, 0.5, 0.5],
        ]
    )

    score = dual_parent_evidence(weights, 0, 1)

    np.testing.assert_allclose(score, [0.0, 0.0, 1.0, 0.5, 0.0])


@pytest.mark.parametrize(
    ("weights", "parent_a", "parent_b", "message"),
    [
        ([[0.5, -0.5]], 0, 1, "nonnegative"),
        ([[0.5, 0.5]], 0, 0, "distinct"),
        ([[0.5, 0.5]], 0, 2, "outside"),
        ([[0.5, 0.5]], 0.0, 1, "integers"),
    ],
)
def test_dual_parent_evidence_rejects_invalid_inputs(
    weights,
    parent_a,
    parent_b,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        dual_parent_evidence(weights, parent_a, parent_b)


def test_four_parent_simplex_recovers_exact_128d_mixture() -> None:
    rng = np.random.default_rng(7)
    prototypes = rng.normal(size=(4, 128))
    expected = np.array(
        [
            [0.10, 0.20, 0.30, 0.40],
            [0.00, 0.75, 0.25, 0.00],
        ]
    )
    targets = expected @ prototypes

    fit = solve_simplex_admixture(
        targets,
        prototypes,
        prototype_names=("Bean", "Maize", "Potato", "Rice"),
    )

    np.testing.assert_allclose(fit.weights, expected, atol=1e-12)
    np.testing.assert_allclose(fit.reconstruction, targets, atol=1e-12)
    np.testing.assert_allclose(fit.residual_norm, 0.0, atol=1e-12)
    np.testing.assert_array_equal(fit.active_count, [4, 2])
    assert fit.prototype_names == ("Bean", "Maize", "Potato", "Rice")


def test_simplex_projection_uses_boundary_for_target_outside_hull() -> None:
    prototypes = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    fit = solve_simplex_admixture([2.0, 0.25], prototypes)

    np.testing.assert_allclose(fit.weights, [[0.0, 1.0, 0.0]])
    assert np.all(fit.weights >= 0)
    np.testing.assert_array_equal(fit.weights.sum(axis=1), [1.0])
    np.testing.assert_allclose(fit.reconstruction, [[1.0, 0.0]])
    np.testing.assert_allclose(fit.squared_error, [1.0625])


def test_simplex_handles_duplicate_prototypes_without_optimizer_failure() -> None:
    prototypes = np.array([[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    fit = solve_simplex_admixture([[0.25, 0.5]], prototypes)

    np.testing.assert_allclose(fit.reconstruction, [[0.25, 0.5]])
    np.testing.assert_allclose(fit.weights.sum(axis=1), [1.0])
    assert np.all(fit.weights >= 0)


def test_two_parent_simplex_matches_closed_segment_projection() -> None:
    targets = np.array([[2.5, 1.0], [12.0, 1.0]])
    prototypes = np.array([[0.0, 0.0], [10.0, 0.0]])

    simplex = solve_simplex_admixture(targets, prototypes)
    segment = solve_parent_segment(targets, prototypes[0], prototypes[1])

    np.testing.assert_allclose(simplex.weights, segment.weights)
    np.testing.assert_allclose(simplex.reconstruction, segment.reconstruction)
    np.testing.assert_allclose(simplex.residual_norm, segment.residual_norm)


def test_exact_weights_survive_same_invertible_affine_diagonal_transform() -> None:
    rng = np.random.default_rng(23)
    prototypes = rng.normal(size=(4, 128))
    simplex_weights = np.array([[0.15, 0.35, 0.10, 0.40]])
    simplex_target = simplex_weights @ prototypes
    segment_weights = np.array([[0.3, 0.7]])
    segment_target = segment_weights @ prototypes[:2]

    scales = np.linspace(0.2, 3.0, 128)
    scales[::2] *= -1.0
    offset = rng.normal(size=128)
    transformed_prototypes = prototypes * scales + offset
    transformed_simplex_target = simplex_target * scales + offset
    transformed_segment_target = segment_target * scales + offset

    original_simplex = solve_simplex_admixture(simplex_target, prototypes)
    transformed_simplex = solve_simplex_admixture(
        transformed_simplex_target,
        transformed_prototypes,
    )
    original_segment = solve_parent_segment(segment_target, prototypes[0], prototypes[1])
    transformed_segment = solve_parent_segment(
        transformed_segment_target,
        transformed_prototypes[0],
        transformed_prototypes[1],
    )

    np.testing.assert_allclose(original_simplex.weights, simplex_weights, atol=1e-12)
    np.testing.assert_allclose(transformed_simplex.weights, simplex_weights, atol=1e-12)
    np.testing.assert_allclose(original_segment.weights, segment_weights, atol=1e-12)
    np.testing.assert_allclose(transformed_segment.weights, segment_weights, atol=1e-12)


def test_parent_segment_exposes_raw_and_clipped_named_weights() -> None:
    targets = np.array([[2.5, 1.0], [12.0, 1.0], [-1.0, 1.0]])

    fit = solve_parent_segment(
        targets,
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        parent_names=("Bean", "Maize"),
    )

    assert fit.parent_names == ("Bean", "Maize")
    np.testing.assert_allclose(fit.raw_coordinate, [0.25, 1.2, -0.1])
    np.testing.assert_allclose(
        fit.raw_weights,
        [[0.75, 0.25], [-0.2, 1.2], [1.1, -0.1]],
    )
    np.testing.assert_allclose(fit.weights, [[0.75, 0.25], [0.0, 1.0], [1.0, 0.0]])
    np.testing.assert_array_equal(fit.in_segment, [True, False, False])
    np.testing.assert_allclose(fit.residual_norm, [1.0, np.sqrt(5.0), np.sqrt(2.0)])


@pytest.mark.parametrize(
    ("targets", "prototypes", "message"),
    [
        ([[np.nan, 0.0]], [[0.0, 0.0], [1.0, 1.0]], "finite"),
        ([[0.0, 0.0]], [[0.0, 0.0], [1.0, np.inf]], "finite"),
        ([[0.0, 0.0]], [[0.0], [1.0]], "feature count"),
        ([[0.0, 0.0]], [[0.0, 0.0]], "between 2 and 4"),
        ([[0.0, 0.0]], np.zeros((5, 2)), "between 2 and 4"),
    ],
)
def test_simplex_rejects_invalid_shapes_and_values(targets, prototypes, message) -> None:
    with pytest.raises(ValueError, match=message):
        solve_simplex_admixture(targets, prototypes)


def test_simplex_rejects_invalid_names() -> None:
    prototypes = [[0.0, 0.0], [1.0, 1.0]]

    with pytest.raises(ValueError, match="expected 2 names"):
        solve_simplex_admixture([[0.5, 0.5]], prototypes, prototype_names=("Bean",))
    with pytest.raises(ValueError, match="unique"):
        solve_simplex_admixture(
            [[0.5, 0.5]],
            prototypes,
            prototype_names=("Bean", "Bean"),
        )
    with pytest.raises(ValueError, match="not one string"):
        solve_simplex_admixture([[0.5, 0.5]], prototypes, prototype_names="BM")


def test_parent_segment_rejects_nonfinite_mismatched_and_duplicate_parents() -> None:
    with pytest.raises(ValueError, match="finite"):
        solve_parent_segment([[0.0, np.nan]], [0.0, 0.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="same feature count"):
        solve_parent_segment([[0.0, 0.0]], [0.0, 0.0], [1.0])
    with pytest.raises(ValueError, match="distinct"):
        solve_parent_segment([[0.0, 0.0]], [1.0, 1.0], [1.0, 1.0])
