"""Small NumPy-only admixture solvers for embedding-space diagnostics.

The simplex solver minimizes Euclidean reconstruction error under nonnegative
weights that sum to one.  It enumerates every nonempty active set, which is
exact for the intended two-to-four prototype use case and needs no optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


_FEASIBILITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class SimplexAdmixtureFit:
    """Result of projecting embedding rows onto a prototype convex hull."""

    prototype_names: tuple[str, ...]
    weights: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    squared_error: np.ndarray
    residual_norm: np.ndarray
    residual_rmse: np.ndarray
    relative_residual: np.ndarray
    active_count: np.ndarray


@dataclass(frozen=True)
class ParentSegmentFit:
    """Closed-form projection onto the line segment between two parents.

    ``raw_coordinate`` is the unconstrained coordinate of ``parent_names[1]``;
    the other raw weight is therefore ``1 - raw_coordinate``.  ``weights`` are
    the clipped segment weights used for reconstruction.
    """

    parent_names: tuple[str, str]
    raw_coordinate: np.ndarray
    raw_weights: np.ndarray
    weights: np.ndarray
    in_segment: np.ndarray
    reconstruction: np.ndarray
    residual: np.ndarray
    squared_error: np.ndarray
    residual_norm: np.ndarray
    residual_rmse: np.ndarray
    relative_residual: np.ndarray


def _finite_matrix(values: object, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric array") from error
    if array.ndim == 1:
        array = array[np.newaxis, :]
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must have shape (samples, features) with no empty axis")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _finite_vector(values: object, name: str) -> np.ndarray:
    if np.iscomplexobj(values):
        raise ValueError(f"{name} must be real-valued")
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a numeric vector") from error
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must have shape (features,)")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validated_names(
    names: Sequence[str] | None,
    count: int,
    prefix: str,
) -> tuple[str, ...]:
    if isinstance(names, str):
        raise ValueError("names must be a sequence of nonempty strings, not one string")
    result = tuple(f"{prefix}_{index}" for index in range(count)) if names is None else tuple(names)
    if len(result) != count:
        raise ValueError(f"expected {count} names, received {len(result)}")
    if any(not isinstance(name, str) or not name.strip() for name in result):
        raise ValueError("names must be nonempty strings")
    if len(set(result)) != count:
        raise ValueError("names must be unique")
    return result


def _diagnostics(
    targets: np.ndarray,
    reconstruction: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    residual = targets - reconstruction
    squared_error = np.einsum("ij,ij->i", residual, residual)
    residual_norm = np.sqrt(squared_error)
    residual_rmse = np.sqrt(np.mean(np.square(residual), axis=1))
    target_norm = np.linalg.norm(targets, axis=1)
    relative_residual = residual_norm / np.maximum(target_norm, np.finfo(np.float64).eps)
    return residual, squared_error, residual_norm, residual_rmse, relative_residual


def _remove_roundoff_from_weights(weights: np.ndarray) -> np.ndarray:
    weights = np.maximum(weights, 0.0)
    weights /= weights.sum(axis=1, keepdims=True)
    pivot = np.argmax(weights, axis=1)
    weights[np.arange(len(weights)), pivot] += 1.0 - weights.sum(axis=1)
    return weights


def dual_parent_evidence(
    weights: object,
    parent_a_index: int,
    parent_b_index: int,
) -> np.ndarray:
    """Score joint named-parent evidence while retaining off-target mass.

    The score is ``4 * w_a * w_b / (w_a + w_b)``. It is zero at either
    pure-parent endpoint, one for a full-mass 50/50 signature, and is reduced
    when the two named parents explain only part of the fitted signature.
    """

    matrix = _finite_matrix(weights, "weights")
    column_count = matrix.shape[1]
    indices = (parent_a_index, parent_b_index)
    if any(not isinstance(index, (int, np.integer)) for index in indices):
        raise ValueError("parent indices must be integers")
    if parent_a_index == parent_b_index:
        raise ValueError("parent indices must be distinct")
    if any(index < 0 or index >= column_count for index in indices):
        raise ValueError("parent index is outside the weight columns")
    if np.any(matrix < -_FEASIBILITY_TOLERANCE):
        raise ValueError("weights must be nonnegative")

    parent_a = np.maximum(matrix[:, parent_a_index], 0.0)
    parent_b = np.maximum(matrix[:, parent_b_index], 0.0)
    named_mass = parent_a + parent_b
    return np.divide(
        4.0 * parent_a * parent_b,
        named_mass,
        out=np.zeros_like(named_mass),
        where=named_mass > np.finfo(np.float64).eps,
    )


def solve_simplex_admixture(
    targets: object,
    prototypes: object,
    *,
    prototype_names: Sequence[str] | None = None,
) -> SimplexAdmixtureFit:
    """Project one or more embeddings onto two-to-four prototype signatures.

    Inputs may contain any feature count, including the intended 128 embedding
    dimensions.  A one-dimensional target is treated as one sample; result
    arrays always retain a leading sample axis.
    """

    target_matrix = _finite_matrix(targets, "targets")
    prototype_matrix = _finite_matrix(prototypes, "prototypes")
    prototype_count, feature_count = prototype_matrix.shape
    if not 2 <= prototype_count <= 4:
        raise ValueError("prototypes must contain between 2 and 4 rows")
    if target_matrix.shape[1] != feature_count:
        raise ValueError(
            "targets and prototypes must have the same feature count "
            f"({target_matrix.shape[1]} != {feature_count})"
        )
    names = _validated_names(prototype_names, prototype_count, "prototype")

    sample_count = len(target_matrix)
    best_weights = np.zeros((sample_count, prototype_count), dtype=np.float64)
    best_error = np.full(sample_count, np.inf, dtype=np.float64)

    indices = tuple(range(prototype_count))
    for active_count in range(1, prototype_count + 1):
        for active in combinations(indices, active_count):
            candidate = np.zeros_like(best_weights)
            if active_count == 1:
                candidate[:, active[0]] = 1.0
                feasible = np.ones(sample_count, dtype=bool)
            else:
                base = prototype_matrix[active[0]]
                directions = (prototype_matrix[list(active[1:])] - base).T
                coordinates = np.linalg.lstsq(
                    directions,
                    (target_matrix - base).T,
                    rcond=None,
                )[0].T
                candidate[:, list(active[1:])] = coordinates
                candidate[:, active[0]] = 1.0 - coordinates.sum(axis=1)
                feasible = np.all(
                    candidate[:, list(active)] >= -_FEASIBILITY_TOLERANCE,
                    axis=1,
                )

            if not feasible.any():
                continue
            feasible_rows = np.flatnonzero(feasible)
            feasible_weights = _remove_roundoff_from_weights(candidate[feasible_rows])
            reconstruction = feasible_weights @ prototype_matrix
            difference = target_matrix[feasible_rows] - reconstruction
            error = np.einsum("ij,ij->i", difference, difference)
            improved = error < best_error[feasible_rows]
            improved_rows = feasible_rows[improved]
            best_weights[improved_rows] = feasible_weights[improved]
            best_error[improved_rows] = error[improved]

    if not np.isfinite(best_error).all():  # Single-prototype active sets make this unreachable.
        raise RuntimeError("simplex projection failed to find a feasible solution")

    reconstruction = best_weights @ prototype_matrix
    residual, squared_error, residual_norm, residual_rmse, relative_residual = _diagnostics(
        target_matrix,
        reconstruction,
    )
    return SimplexAdmixtureFit(
        prototype_names=names,
        weights=best_weights,
        reconstruction=reconstruction,
        residual=residual,
        squared_error=squared_error,
        residual_norm=residual_norm,
        residual_rmse=residual_rmse,
        relative_residual=relative_residual,
        active_count=np.count_nonzero(best_weights > _FEASIBILITY_TOLERANCE, axis=1),
    )


def solve_parent_segment(
    targets: object,
    parent_a: object,
    parent_b: object,
    *,
    parent_names: Sequence[str] = ("parent_a", "parent_b"),
) -> ParentSegmentFit:
    """Fit closed-form proportions on the segment between two named parents."""

    target_matrix = _finite_matrix(targets, "targets")
    first = _finite_vector(parent_a, "parent_a")
    second = _finite_vector(parent_b, "parent_b")
    if target_matrix.shape[1] != first.size or first.shape != second.shape:
        raise ValueError("targets and both parents must have the same feature count")
    names = _validated_names(parent_names, 2, "parent")

    direction = second - first
    squared_length = float(direction @ direction)
    if squared_length == 0.0:
        raise ValueError("parent prototypes must be distinct")

    raw_coordinate = ((target_matrix - first) @ direction) / squared_length
    raw_weights = np.column_stack((1.0 - raw_coordinate, raw_coordinate))
    clipped_coordinate = np.clip(raw_coordinate, 0.0, 1.0)
    weights = np.column_stack((1.0 - clipped_coordinate, clipped_coordinate))
    in_segment = (raw_coordinate >= -_FEASIBILITY_TOLERANCE) & (
        raw_coordinate <= 1.0 + _FEASIBILITY_TOLERANCE
    )
    reconstruction = weights @ np.vstack((first, second))
    residual, squared_error, residual_norm, residual_rmse, relative_residual = _diagnostics(
        target_matrix,
        reconstruction,
    )
    return ParentSegmentFit(
        parent_names=(names[0], names[1]),
        raw_coordinate=raw_coordinate,
        raw_weights=raw_weights,
        weights=weights,
        in_segment=in_segment,
        reconstruction=reconstruction,
        residual=residual,
        squared_error=squared_error,
        residual_norm=residual_norm,
        residual_rmse=residual_rmse,
        relative_residual=relative_residual,
    )


__all__ = [
    "ParentSegmentFit",
    "SimplexAdmixtureFit",
    "dual_parent_evidence",
    "solve_parent_segment",
    "solve_simplex_admixture",
]
