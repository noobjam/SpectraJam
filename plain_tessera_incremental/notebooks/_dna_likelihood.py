"""NumPy-only likelihood helpers for the intercropping DNA notebook.

The model is deliberately small and explicit:

* crop references are field-balanced;
* all embedding dimensions are scored jointly with one shrunken covariance;
* named-parent evidence is separated from the fitted parent balance; and
* fields are treated as bags of spatially correlated pixels.

The resulting shares describe TESSERA embedding signatures.  They are not
plant counts, biomass, yield, or planted-area fractions.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import numpy as np


_EPSILON = np.finfo(np.float64).eps


def _matrix(values: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric") from error
    if result.ndim == 1:
        result = result[np.newaxis, :]
    if result.ndim != 2 or min(result.shape) < 1:
        raise ValueError(f"{name} must have shape (samples, features)")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _vector(values: object, name: str, length: int | None = None) -> np.ndarray:
    result = np.asarray(values)
    if result.ndim != 1 or result.size == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if length is not None and result.size != length:
        raise ValueError(f"{name} must contain {length} values")
    return result


def _weights(values: object | None, sample_count: int) -> np.ndarray:
    if values is None:
        result = np.ones(sample_count, dtype=np.float64)
    else:
        result = np.asarray(values, dtype=np.float64)
    if result.shape != (sample_count,):
        raise ValueError("sample_weights must have one value per sample")
    if not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("sample_weights must be finite and strictly positive")
    return result / result.sum()


def _names(values: Sequence[str], name: str) -> tuple[str, ...]:
    if isinstance(values, str):
        raise ValueError(f"{name} must be a sequence, not one string")
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain nonempty names")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique names")
    return result


def l2_normalize(features: object) -> np.ndarray:
    """Return row-wise unit vectors without changing the input array."""

    matrix = _matrix(features, "features")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= _EPSILON):
        raise ValueError("features contains a zero-norm row")
    return matrix / norms


def field_balanced_weights(
    field_ids: object,
    labels: object,
    crop_names: Sequence[str],
) -> np.ndarray:
    """Give every crop equal mass, every field equal mass inside its crop,
    and every pixel equal mass inside its field.
    """

    fields = _vector(field_ids, "field_ids")
    crop_labels = _vector(labels, "labels", len(fields)).astype(str)
    names = _names(crop_names, "crop_names")
    unknown = set(crop_labels).difference(names)
    if unknown:
        raise ValueError(f"labels contains unknown crops: {sorted(unknown)}")

    field_label: dict[str, str] = {}
    for field_id, label in zip(fields.astype(str), crop_labels, strict=True):
        previous = field_label.setdefault(field_id, label)
        if previous != label:
            raise ValueError(f"field {field_id!r} has multiple crop labels")

    fields_per_crop = {
        crop: sum(label == crop for label in field_label.values()) for crop in names
    }
    missing = [crop for crop, count in fields_per_crop.items() if count == 0]
    if missing:
        raise ValueError(f"crop references are missing fields: {missing}")

    pixel_counts: dict[str, int] = {}
    for field_id in fields.astype(str):
        pixel_counts[field_id] = pixel_counts.get(field_id, 0) + 1

    crop_count = len(names)
    result = np.empty(len(fields), dtype=np.float64)
    for index, (field_id, crop) in enumerate(
        zip(fields.astype(str), crop_labels, strict=True)
    ):
        result[index] = 1.0 / (
            crop_count * fields_per_crop[crop] * pixel_counts[field_id]
        )
    result /= result.sum()
    return result


@dataclass(frozen=True)
class CropDensityModel:
    """Shared-covariance Gaussian reference model in a whitened basis."""

    crop_names: tuple[str, ...]
    feature_origin: np.ndarray
    whitener: np.ndarray
    crop_centers: np.ndarray
    covariance_eigenvalues: np.ndarray
    shrinkage: float
    temperature: float = 1.0

    @property
    def feature_count(self) -> int:
        return int(self.feature_origin.size)

    @property
    def component_count(self) -> int:
        return int(self.whitener.shape[1])


@dataclass(frozen=True)
class ValidatedCropDensity:
    """Final reference model plus field-held-out calibration evidence."""

    model: CropDensityModel
    sample_folds: np.ndarray
    oof_log_scores: np.ndarray
    chosen_shrinkage: float
    temperature: float
    fold_count: int
    field_balanced_accuracy: float
    field_log_loss: float


def fit_crop_density(
    features: object,
    labels: object,
    *,
    crop_names: Sequence[str],
    sample_weights: object | None = None,
    shrinkage: float = 0.25,
    max_components: int | None = None,
) -> CropDensityModel:
    """Fit crop means and a regularized pooled within-crop covariance.

    ``shrinkage`` blends the empirical covariance toward an isotropic matrix.
    Eigenvectors are retained rather than treating 128 learned dimensions as
    independent genetic loci.
    """

    matrix = _matrix(features, "features")
    crop_labels = _vector(labels, "labels", len(matrix)).astype(str)
    names = _names(crop_names, "crop_names")
    if not 0 < float(shrinkage) <= 1:
        raise ValueError("shrinkage must be in (0, 1]")
    if max_components is not None and not 1 <= int(max_components) <= matrix.shape[1]:
        raise ValueError("max_components is outside the feature dimensions")
    unknown = set(crop_labels).difference(names)
    if unknown:
        raise ValueError(f"labels contains unknown crops: {sorted(unknown)}")
    weights = _weights(sample_weights, len(matrix))

    raw_means = []
    residuals = np.empty_like(matrix)
    for crop in names:
        mask = crop_labels == crop
        if not mask.any():
            raise ValueError(f"crop {crop!r} has no reference samples")
        crop_weights = weights[mask]
        crop_weights /= crop_weights.sum()
        mean = np.average(matrix[mask], axis=0, weights=crop_weights)
        raw_means.append(mean)
        residuals[mask] = matrix[mask] - mean

    feature_origin = np.average(matrix, axis=0, weights=weights)
    covariance = (residuals * weights[:, None]).T @ residuals
    covariance /= weights.sum()
    average_variance = float(np.trace(covariance) / matrix.shape[1])
    if not np.isfinite(average_variance) or average_variance <= _EPSILON:
        raise ValueError("reference covariance has no usable variation")
    regularized = (
        (1.0 - float(shrinkage)) * covariance
        + float(shrinkage) * average_variance * np.eye(matrix.shape[1])
    )

    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    floor = max(average_variance * 1e-8, _EPSILON)
    supported = eigenvalues > floor
    if not supported.any():
        raise ValueError("regularized covariance has no supported components")
    eigenvalues = eigenvalues[supported]
    eigenvectors = eigenvectors[:, supported]
    if max_components is not None:
        eigenvalues = eigenvalues[: int(max_components)]
        eigenvectors = eigenvectors[:, : int(max_components)]

    whitener = eigenvectors / np.sqrt(eigenvalues)[None, :]
    raw_mean_matrix = np.vstack(raw_means)
    crop_centers = (raw_mean_matrix - feature_origin) @ whitener
    return CropDensityModel(
        crop_names=names,
        feature_origin=feature_origin,
        whitener=whitener,
        crop_centers=crop_centers,
        covariance_eigenvalues=eigenvalues,
        shrinkage=float(shrinkage),
    )


def crop_log_scores(model: CropDensityModel, features: object) -> np.ndarray:
    """Return equal-prior crop log scores, up to one shared constant."""

    matrix = _matrix(features, "features")
    if matrix.shape[1] != model.feature_count:
        raise ValueError("features and model have different feature counts")
    projected = (matrix - model.feature_origin) @ model.whitener
    difference = projected[:, None, :] - model.crop_centers[None, :, :]
    squared_distance = np.einsum("nck,nck->nc", difference, difference)
    return -0.5 * squared_distance / model.temperature


def crop_posteriors(log_scores: object) -> np.ndarray:
    scores = _matrix(log_scores, "log_scores")
    shifted = scores - scores.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def calibrate_temperature(
    raw_log_scores: object,
    true_labels: object,
    *,
    crop_names: Sequence[str],
    sample_weights: object | None = None,
    candidates: object | None = None,
) -> float:
    """Select a scalar temperature on held-out reference scores."""

    scores = _matrix(raw_log_scores, "raw_log_scores")
    names = _names(crop_names, "crop_names")
    if scores.shape[1] != len(names):
        raise ValueError("raw_log_scores columns do not match crop_names")
    labels = _vector(true_labels, "true_labels", len(scores)).astype(str)
    name_to_index = {name: index for index, name in enumerate(names)}
    try:
        targets = np.array([name_to_index[label] for label in labels], dtype=np.int64)
    except KeyError as error:
        raise ValueError(f"unknown true label: {error.args[0]}") from error
    weights = _weights(sample_weights, len(scores))
    grid = (
        np.geomspace(0.25, 8.0, 65)
        if candidates is None
        else np.asarray(candidates, dtype=np.float64)
    )
    if grid.ndim != 1 or grid.size == 0 or not np.isfinite(grid).all() or np.any(grid <= 0):
        raise ValueError("temperature candidates must be finite and positive")

    losses = []
    rows = np.arange(len(scores))
    for temperature in grid:
        scaled = scores / temperature
        maximum = scaled.max(axis=1)
        log_normalizer = maximum + np.log(
            np.exp(scaled - maximum[:, None]).sum(axis=1)
        )
        losses.append(float(np.sum(weights * (log_normalizer - scaled[rows, targets]))))
    return float(grid[int(np.argmin(losses))])


def with_temperature(model: CropDensityModel, temperature: float) -> CropDensityModel:
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return replace(model, temperature=float(temperature))


def fit_validated_crop_density(
    features: object,
    labels: object,
    field_ids: object,
    *,
    crop_names: Sequence[str],
    shrinkage_candidates: Sequence[float] = (0.10, 0.25, 0.50, 0.75),
    max_folds: int = 5,
    seed: int = 0,
    max_components: int | None = None,
) -> ValidatedCropDensity:
    """Choose covariance shrinkage and calibrate probabilities by field OOF.

    Folds are stratified by crop and keep every pixel from a field together.
    This measures held-out-field performance; it is intentionally not called a
    geographic generalization test.
    """

    matrix = _matrix(features, "features")
    crop_labels = _vector(labels, "labels", len(matrix)).astype(str)
    fields = _vector(field_ids, "field_ids", len(matrix)).astype(str)
    names = _names(crop_names, "crop_names")
    if not isinstance(max_folds, (int, np.integer)) or max_folds < 2:
        raise ValueError("max_folds must be an integer of at least 2")
    candidate_array = np.asarray(shrinkage_candidates, dtype=np.float64)
    if (
        candidate_array.ndim != 1
        or candidate_array.size == 0
        or not np.isfinite(candidate_array).all()
        or np.any(candidate_array <= 0)
        or np.any(candidate_array > 1)
    ):
        raise ValueError("shrinkage candidates must lie in (0, 1]")

    field_label: dict[str, str] = {}
    for field_id, label in zip(fields, crop_labels, strict=True):
        previous = field_label.setdefault(field_id, label)
        if previous != label:
            raise ValueError(f"field {field_id!r} has multiple crop labels")
    crop_fields = {
        crop: np.array(
            sorted(field for field, label in field_label.items() if label == crop),
            dtype=object,
        )
        for crop in names
    }
    missing = [crop for crop, values in crop_fields.items() if len(values) < 2]
    if missing:
        raise ValueError(f"crops need at least two reference fields: {missing}")
    fold_count = min(int(max_folds), *(len(values) for values in crop_fields.values()))
    rng = np.random.default_rng(seed)
    field_fold: dict[str, int] = {}
    for crop in names:
        values = crop_fields[crop].copy()
        rng.shuffle(values)
        for position, field_id in enumerate(values):
            field_fold[str(field_id)] = position % fold_count
    sample_folds = np.array([field_fold[field_id] for field_id in fields], dtype=np.int16)
    global_weights = field_balanced_weights(fields, crop_labels, names)

    candidate_scores: list[np.ndarray] = []
    candidate_losses = []
    target_index = np.array([names.index(label) for label in crop_labels], dtype=np.int64)
    rows = np.arange(len(matrix))
    for shrinkage in candidate_array:
        oof = np.empty((len(matrix), len(names)), dtype=np.float64)
        for fold in range(fold_count):
            train = sample_folds != fold
            holdout = ~train
            train_weights = field_balanced_weights(
                fields[train], crop_labels[train], names
            )
            fold_model = fit_crop_density(
                matrix[train],
                crop_labels[train],
                crop_names=names,
                sample_weights=train_weights,
                shrinkage=float(shrinkage),
                max_components=max_components,
            )
            oof[holdout] = crop_log_scores(fold_model, matrix[holdout])
        maximum = oof.max(axis=1)
        log_normalizer = maximum + np.log(
            np.exp(oof - maximum[:, None]).sum(axis=1)
        )
        loss = float(
            np.sum(global_weights * (log_normalizer - oof[rows, target_index]))
        )
        candidate_scores.append(oof)
        candidate_losses.append(loss)

    best = int(np.argmin(candidate_losses))
    chosen_shrinkage = float(candidate_array[best])
    raw_oof = candidate_scores[best]
    temperature = calibrate_temperature(
        raw_oof,
        crop_labels,
        crop_names=names,
        sample_weights=global_weights,
    )
    calibrated_oof = raw_oof / temperature

    field_probability: dict[str, np.ndarray] = {}
    for field_id in sorted(field_label):
        mask = fields == field_id
        field_probability[field_id] = crop_posteriors(calibrated_oof[mask]).mean(axis=0)
    recalls = []
    field_losses = []
    for crop_index, crop in enumerate(names):
        selected = [
            field_probability[field]
            for field, label in field_label.items()
            if label == crop
        ]
        probabilities = np.vstack(selected)
        recalls.append(float(np.mean(np.argmax(probabilities, axis=1) == crop_index)))
        field_losses.extend(-np.log(np.maximum(probabilities[:, crop_index], _EPSILON)))

    final_weights = field_balanced_weights(fields, crop_labels, names)
    final_model = fit_crop_density(
        matrix,
        crop_labels,
        crop_names=names,
        sample_weights=final_weights,
        shrinkage=chosen_shrinkage,
        max_components=max_components,
    )
    final_model = with_temperature(final_model, temperature)
    return ValidatedCropDensity(
        model=final_model,
        sample_folds=sample_folds,
        oof_log_scores=calibrated_oof,
        chosen_shrinkage=chosen_shrinkage,
        temperature=temperature,
        fold_count=fold_count,
        field_balanced_accuracy=float(np.mean(recalls)),
        field_log_loss=float(np.mean(field_losses)),
    )


@dataclass(frozen=True)
class ParentAxisFit:
    parent_names: tuple[str, str]
    raw_parent_a_share: np.ndarray
    parent_a_share: np.ndarray
    in_segment: np.ndarray
    tube_squared_distance: np.ndarray
    parent_squared_separation: float


def fit_parent_axis(
    model: CropDensityModel,
    features: object,
    parent_a: str,
    parent_b: str,
) -> ParentAxisFit:
    """Project embeddings onto a covariance-aware named-parent segment."""

    if parent_a == parent_b:
        raise ValueError("parents must be distinct")
    index = {name: position for position, name in enumerate(model.crop_names)}
    if parent_a not in index or parent_b not in index:
        raise ValueError("both parents must exist in the crop model")
    matrix = _matrix(features, "features")
    if matrix.shape[1] != model.feature_count:
        raise ValueError("features and model have different feature counts")
    projected = (matrix - model.feature_origin) @ model.whitener
    first = model.crop_centers[index[parent_a]]
    second = model.crop_centers[index[parent_b]]
    direction = first - second
    squared_separation = float(direction @ direction)
    if squared_separation <= _EPSILON:
        raise ValueError("named-parent references are not distinguishable")
    raw_share = ((projected - second) @ direction) / squared_separation
    share = np.clip(raw_share, 0.0, 1.0)
    reconstruction = second + share[:, None] * direction
    residual = projected - reconstruction
    tube_squared_distance = np.einsum("nk,nk->n", residual, residual)
    return ParentAxisFit(
        parent_names=(parent_a, parent_b),
        raw_parent_a_share=raw_share,
        parent_a_share=share,
        in_segment=(raw_share >= 0.0) & (raw_share <= 1.0),
        tube_squared_distance=tube_squared_distance,
        parent_squared_separation=squared_separation,
    )


@dataclass(frozen=True)
class MosaicShareFit:
    parent_a_share: float
    log_likelihood: float
    best_endpoint_log_likelihood: float
    log_evidence_over_pure: float
    pixel_parent_a_probability: np.ndarray
    grid: np.ndarray
    grid_log_likelihood: np.ndarray


def _log_mixture_grid(
    log_score_a: np.ndarray,
    log_score_b: np.ndarray,
    grid: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    with np.errstate(divide="ignore"):
        first = np.log(grid)[:, None] + log_score_a[None, :]
        second = np.log1p(-grid)[:, None] + log_score_b[None, :]
    mixed = np.logaddexp(first, second)
    return mixed @ weights


def fit_mosaic_share(
    log_score_a: object,
    log_score_b: object,
    *,
    sample_weights: object | None = None,
    grid_size: int = 201,
) -> MosaicShareFit:
    """Fit a field's A-like/B-like pixel mixture by maximum likelihood."""

    first = np.asarray(log_score_a, dtype=np.float64)
    second = np.asarray(log_score_b, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("parent log scores must be equally sized nonempty vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("parent log scores must be finite")
    if not isinstance(grid_size, (int, np.integer)) or grid_size < 3:
        raise ValueError("grid_size must be an integer of at least 3")
    if sample_weights is None:
        weights = np.ones(len(first), dtype=np.float64)
    else:
        weights = np.asarray(sample_weights, dtype=np.float64)
        if weights.shape != first.shape or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("sample_weights must be finite, positive, and match the scores")
    grid = np.linspace(0.0, 1.0, int(grid_size))
    likelihood = _log_mixture_grid(first, second, grid, weights)
    maximum = likelihood.max()
    tied = np.flatnonzero(np.isclose(likelihood, maximum, rtol=1e-12, atol=1e-12))
    best_index = int(tied[np.argmin(np.abs(grid[tied] - 0.5))])
    share = float(grid[best_index])
    endpoint = float(max(likelihood[0], likelihood[-1]))

    if share <= 0.0:
        responsibilities = np.zeros(len(first), dtype=np.float64)
    elif share >= 1.0:
        responsibilities = np.ones(len(first), dtype=np.float64)
    else:
        numerator = np.log(share) + first
        denominator = np.logaddexp(numerator, np.log1p(-share) + second)
        responsibilities = np.exp(numerator - denominator)
    return MosaicShareFit(
        parent_a_share=share,
        log_likelihood=float(likelihood[best_index]),
        best_endpoint_log_likelihood=endpoint,
        log_evidence_over_pure=float(likelihood[best_index] - endpoint),
        pixel_parent_a_probability=responsibilities,
        grid=grid,
        grid_log_likelihood=likelihood,
    )


def block_bootstrap_mosaic_share(
    log_score_a: object,
    log_score_b: object,
    block_ids: object,
    *,
    replicates: int = 200,
    seed: int = 0,
    grid_size: int = 201,
) -> np.ndarray:
    """Resample target spatial blocks and refit the mosaic balance."""

    first = np.asarray(log_score_a, dtype=np.float64)
    second = np.asarray(log_score_b, dtype=np.float64)
    blocks = _vector(block_ids, "block_ids", len(first)).astype(str)
    if first.ndim != 1 or first.shape != second.shape or first.size == 0:
        raise ValueError("parent log scores must be equally sized nonempty vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("parent log scores must be finite")
    if not isinstance(replicates, (int, np.integer)) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    unique_blocks = np.array(sorted(set(blocks)), dtype=object)
    positions = {block: np.flatnonzero(blocks == block) for block in unique_blocks}
    rng = np.random.default_rng(seed)
    shares = np.empty(int(replicates), dtype=np.float64)
    for replicate in range(int(replicates)):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        selected = np.concatenate([positions[block] for block in sampled])
        shares[replicate] = fit_mosaic_share(
            first[selected],
            second[selected],
            grid_size=grid_size,
        ).parent_a_share
    return shares


def empirical_upper_tail_probability(value: float, null_values: object) -> float:
    null = np.asarray(null_values, dtype=np.float64)
    if null.ndim != 1 or null.size == 0 or not np.isfinite(null).all():
        raise ValueError("null_values must be a nonempty finite vector")
    if not np.isfinite(value):
        raise ValueError("value must be finite")
    return float((1 + np.count_nonzero(null >= value)) / (len(null) + 1))


__all__ = [
    "CropDensityModel",
    "MosaicShareFit",
    "ParentAxisFit",
    "ValidatedCropDensity",
    "block_bootstrap_mosaic_share",
    "calibrate_temperature",
    "crop_log_scores",
    "crop_posteriors",
    "empirical_upper_tail_probability",
    "field_balanced_weights",
    "fit_crop_density",
    "fit_validated_crop_density",
    "fit_mosaic_share",
    "fit_parent_axis",
    "l2_normalize",
    "with_temperature",
]
