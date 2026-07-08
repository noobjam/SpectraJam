"""All-field, all-window intercropping parentage/admixture workflow.

This module keeps the notebook readable.  It validates the source artifacts via
``_pixel_analysis``, fits a field-held-out reference model, scores every
available canonical physical field, and maps those physical results back to
every original source record.

The output is deliberately split into evidence, attribution, and adequacy:

* ``log_evidence_over_pure`` asks whether two pixel-source distributions fit
  better than either named parent alone;
* ``mosaic_parent_a_share`` estimates the balance of A-like and B-like pixels;
* ``named_parent_mass`` and ``best_alternative_pair`` expose forced/off-family
  explanations.

None of these quantities is a calibrated planted-area or biomass percentage.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import reduce
from itertools import combinations
from pathlib import Path
from typing import Callable, Iterable
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd

from plain_tessera_incremental.notebooks._dna_likelihood import (
    CropDensityModel,
    ValidatedCropDensity,
    block_bootstrap_mosaic_share,
    crop_log_scores,
    crop_posteriors,
    empirical_upper_tail_probability,
    fit_mosaic_share,
    fit_parent_axis,
    fit_validated_crop_density,
    l2_normalize,
)
from plain_tessera_incremental.notebooks._pixel_analysis import (
    StaticArtifacts,
    WindowScan,
    load_embeddings,
    load_static,
    scan_window,
)


@dataclass(frozen=True)
class DNAWorkflowConfig:
    output_dir: Path
    export_dir: Path
    windows: tuple[str, ...] = ("w1", "w2", "w3", "w4")
    mono_crops: tuple[str, ...] = ("Maize", "Bean", "Irish Potato", "Rice")
    mixture_parents: tuple[tuple[str, str, str], ...] = (
        ("bean_maize", "Bean", "Maize"),
        ("potato_maize", "Irish Potato", "Maize"),
    )
    mixture_labels: tuple[tuple[str, str], ...] = (
        ("Bean and Maize", "bean_maize"),
        ("Irish Potato and Maize", "potato_maize"),
    )
    shrinkage_candidates: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    max_folds: int = 5
    bootstrap_replicates: int = 200
    spatial_block_pixels: int = 2
    mixture_grid_size: int = 201
    min_reference_fields_per_crop: int = 5
    min_balanced_accuracy: float = 0.65
    max_endpoint_mae: float = 0.25
    max_synthetic_mae: float = 0.20
    evidence_quantile: float = 0.95
    named_mass_quantile: float = 0.05
    endpoint_epsilon: float = 0.10
    random_seed: int = 20260708

    @property
    def pair_map(self) -> dict[str, tuple[str, str]]:
        return {key: (parent_a, parent_b) for key, parent_a, parent_b in self.mixture_parents}

    @property
    def mixture_label_map(self) -> dict[str, str]:
        return dict(self.mixture_labels)

    @property
    def target_labels(self) -> tuple[str, ...]:
        return (*self.mono_crops, *self.mixture_label_map)


@dataclass
class DNAWorkflowResult:
    config: DNAWorkflowConfig
    static: StaticArtifacts
    scans: dict[str, WindowScan]
    field_crosswalk: pd.DataFrame
    analysis_units: pd.DataFrame
    pixel_scores: pd.DataFrame
    physical_field_pair_scores: pd.DataFrame
    source_field_scores: pd.DataFrame
    reference_controls: pd.DataFrame
    synthetic_controls: pd.DataFrame
    validation_metrics: pd.DataFrame
    window_summary: pd.DataFrame
    models: dict[str, CropDensityModel]
    snapshot_id: str
    analysis_id: str


def default_config(
    output_dir: str | Path,
    export_dir: str | Path | None = None,
) -> DNAWorkflowConfig:
    output = Path(output_dir)
    export = (
        Path(export_dir)
        if export_dir is not None
        else output / "analysis" / "intercropping_parentage_likelihood_v1"
    )
    return DNAWorkflowConfig(output_dir=output, export_dir=export)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _pixel_count_bin(count: int) -> str:
    if count <= 1:
        return "1"
    if count == 2:
        return "2"
    if count <= 4:
        return "3-4"
    if count <= 8:
        return "5-8"
    if count <= 16:
        return "9-16"
    if count <= 32:
        return "17-32"
    if count <= 64:
        return "33-64"
    return "65+"


def _quantile(values: Iterable[float], probability: float, method: str) -> float:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("cannot take a calibrated quantile of empty/nonfinite values")
    return float(np.quantile(array, probability, method=method))


def _snapshot_id(scans: dict[str, WindowScan], run_fingerprint: str) -> str:
    payload = {
        "run_fingerprint": run_fingerprint,
        "windows": {
            window: [
                {
                    "name": path.name,
                    "task_fingerprint": scan.task_fingerprints.get(path.stem),
                }
                for path in scan.files
            ]
            for window, scan in sorted(scans.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _analysis_id(config: DNAWorkflowConfig) -> str:
    config_payload = asdict(config)
    config_payload.pop("output_dir", None)
    config_payload.pop("export_dir", None)
    code_hashes = {}
    for name in ("_dna_likelihood.py", "_dna_workflow.py", "_dna_reporting.py"):
        path = Path(__file__).with_name(name)
        code_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = {
        "analysis_name": "intercropping_parentage_likelihood_v1",
        "analysis_version": 1,
        "config": config_payload,
        "code_sha256": code_hashes,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_field_crosswalk(
    static: StaticArtifacts,
    target_labels: Iterable[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map every source row to one geometry+label analysis unit.

    Identical same-label geometries share one analysis result/report.  Exact
    geometries carrying conflicting labels are retained in reporting but are
    marked and never used as reference evidence.
    """

    targets = set(str(label) for label in target_labels)
    fields = static.fields.copy()
    fields["field_uid"] = fields["field_uid"].astype(str)
    fields["landcover"] = fields["landcover"].astype(str).str.strip()
    fields["geometry_label_count"] = fields.groupby("geometry_sha256")[
        "landcover"
    ].transform("nunique").astype(np.int64)
    fields["analysis_unit_uid"] = fields.groupby(
        ["geometry_sha256", "landcover"], sort=False
    )["field_uid"].transform("min")
    fields["source_replica_count"] = fields.groupby(
        ["geometry_sha256", "landcover"], sort=False
    )["field_uid"].transform("size").astype(np.int64)
    fields["is_analysis_unit"] = fields["field_uid"].eq(fields["analysis_unit_uid"])
    fields["supported_label"] = fields["landcover"].isin(targets)

    fields["record_role"] = "same_geometry_replica"
    fields.loc[fields["is_analysis_unit"], "record_role"] = "canonical"
    fields.loc[~fields["supported_label"], "record_role"] = "unsupported_label"
    conflict = fields["geometry_label_count"].gt(1) & fields["supported_label"]
    fields.loc[conflict & fields["is_analysis_unit"], "record_role"] = (
        "conflicting_label_canonical"
    )
    fields.loc[conflict & ~fields["is_analysis_unit"], "record_role"] = (
        "conflicting_label_replica"
    )

    columns = [
        "field_uid",
        "analysis_unit_uid",
        "id",
        "landcover",
        "wkt",
        "utm_epsg",
        "pixel_count",
        "geometry_sha256",
        "geometry_label_count",
        "source_replica_count",
        "record_role",
        "supported_label",
        "is_analysis_unit",
    ]
    crosswalk = fields[columns].copy()
    units = crosswalk[
        crosswalk["is_analysis_unit"]
        & crosswalk["supported_label"]
        & crosswalk["geometry_label_count"].eq(1)
    ].copy()
    if units["analysis_unit_uid"].duplicated().any():
        raise RuntimeError("analysis_unit_uid is not unique")
    return crosswalk.reset_index(drop=True), units.reset_index(drop=True)


def _window_rows(
    static: StaticArtifacts,
    scan: WindowScan,
    analysis_units: pd.DataFrame,
) -> pd.DataFrame:
    unit_columns = [
        "analysis_unit_uid",
        "landcover",
        "geometry_sha256",
        "geometry_label_count",
        "source_replica_count",
    ]
    units = analysis_units[unit_columns].rename(
        columns={"analysis_unit_uid": "field_uid", "landcover": "unit_landcover"}
    )
    flags = static.memberships[
        [
            "field_uid",
            "pixel_id",
            "utm_epsg",
            "pixel_x_index",
            "pixel_y_index",
            "pixel_longitude",
            "pixel_latitude",
            "overlap_field_count",
            "label_conflict",
        ]
    ].copy()
    rows = scan.index.merge(units, on="field_uid", how="inner", validate="many_to_one")
    rows = rows.merge(
        flags,
        on=["field_uid", "pixel_id"],
        how="left",
        validate="one_to_one",
    )
    if rows[["overlap_field_count", "label_conflict"]].isna().any().any():
        raise RuntimeError(f"membership flags disappeared in {scan.window_id}")
    if "landcover" in rows and not rows["landcover"].eq(rows["unit_landcover"]).all():
        raise RuntimeError(f"field labels disagree in {scan.window_id}")
    rows["landcover"] = rows["unit_landcover"]
    rows = rows.drop(columns="unit_landcover")
    rows["window_id"] = scan.window_id
    return rows


def _longitudinal_rows(
    rows_by_window: dict[str, pd.DataFrame],
    scans: dict[str, WindowScan],
    analysis_units: pd.DataFrame,
    config: DNAWorkflowConfig,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Freeze fully published fields and the same complete pixels in all windows."""

    unit_ids = set(analysis_units["analysis_unit_uid"].astype(str))
    common_published = unit_ids.intersection(
        *(set(scans[window].published_fields) for window in config.windows)
    )
    complete_sets = [
        set(
            rows_by_window[window].loc[
                rows_by_window[window]["field_uid"].isin(common_published)
                & rows_by_window[window]["outcome"].eq("complete"),
                ["field_uid", "pixel_id"],
            ].itertuples(index=False, name=None)
        )
        for window in config.windows
    ]
    common_pairs = reduce(set.intersection, complete_sets) if complete_sets else set()
    common_index = pd.MultiIndex.from_tuples(
        sorted(common_pairs), names=["field_uid", "pixel_id"]
    )
    frozen: dict[str, pd.DataFrame] = {}
    for window in config.windows:
        rows = rows_by_window[window].set_index(["field_uid", "pixel_id"])
        selected = rows.loc[rows.index.intersection(common_index)].reset_index()
        if set(selected[["field_uid", "pixel_id"]].itertuples(index=False, name=None)) != (
            common_pairs
        ):
            raise RuntimeError(f"{window} lost a longitudinal field/pixel membership")
        frozen[window] = selected.sort_values(
            ["field_uid", "pixel_id"], kind="stable"
        ).reset_index(drop=True)

    scoreable_units = {field_uid for field_uid, _ in common_pairs}
    availability_rows = []
    for field_uid in sorted(unit_ids):
        if field_uid not in common_published:
            status = "PENDING_NOT_FULLY_PUBLISHED_ALL_WINDOWS"
        elif field_uid not in scoreable_units:
            status = "EMPTY_OR_INCOMPLETE_WINDOW"
        else:
            status = "SCOREABLE"
        availability_rows.extend(
            {
                "analysis_unit_uid": field_uid,
                "window_id": window,
                "availability_status": status,
            }
            for window in config.windows
        )
    return frozen, pd.DataFrame(availability_rows)


def _common_reference_rows(
    static: StaticArtifacts,
    rows_by_window: dict[str, pd.DataFrame],
    config: DNAWorkflowConfig,
) -> dict[str, pd.DataFrame]:
    unit_fields = {
        field_uid: label
        for field_uid, label in rows_by_window[config.windows[0]][
            ["field_uid", "landcover"]
        ].drop_duplicates().itertuples(index=False, name=None)
    }
    membership = static.memberships[
        static.memberships["field_uid"].isin(unit_fields)
    ][["field_uid", "pixel_id", "label_conflict"]].copy()
    membership["landcover"] = membership["field_uid"].map(unit_fields)
    membership = membership[
        membership["landcover"].isin(config.mono_crops)
        & ~membership["label_conflict"].astype(bool)
    ].sort_values(["landcover", "pixel_id", "field_uid"], kind="stable")
    # One physical pixel may lie in multiple same-label fields.  It contributes
    # once to reference fitting and is assigned deterministically to one field.
    membership = membership.drop_duplicates(["landcover", "pixel_id"], keep="first")
    reference_keys = set(
        membership[["field_uid", "pixel_id"]].itertuples(index=False, name=None)
    )
    complete_sets = []
    for window in config.windows:
        rows = rows_by_window[window]
        complete_sets.append(
            set(
                rows[
                    rows["outcome"].eq("complete")
                    & rows["geometry_label_count"].eq(1)
                ][["field_uid", "pixel_id"]].itertuples(index=False, name=None)
            )
            & reference_keys
        )
    common = reduce(set.intersection, complete_sets)
    if not common:
        raise RuntimeError("no conflict-free monocrop reference pixels span every window")
    index = pd.MultiIndex.from_tuples(
        sorted(common), names=["field_uid", "pixel_id"]
    )
    result = {}
    for window in config.windows:
        rows = rows_by_window[window].set_index(["field_uid", "pixel_id"])
        selected = rows.loc[rows.index.intersection(index)].reset_index()
        selected = selected[selected["landcover"].isin(config.mono_crops)]
        counts = selected.groupby("landcover")["field_uid"].nunique().reindex(
            config.mono_crops, fill_value=0
        )
        if (counts < 2).any():
            raise RuntimeError(
                f"{window} lacks two reference fields for: "
                + ", ".join(f"{crop}={count}" for crop, count in counts.items() if count < 2)
            )
        result[window] = selected.sort_values(
            ["landcover", "field_uid", "pixel_id"], kind="stable"
        ).reset_index(drop=True)
    return result


def _crop_probability_columns(crop_names: Iterable[str]) -> dict[str, str]:
    return {crop: f"prob_{_slug(crop)}" for crop in crop_names}


def _fit_reference_window(
    static: StaticArtifacts,
    scan: WindowScan,
    rows: pd.DataFrame,
    config: DNAWorkflowConfig,
) -> tuple[ValidatedCropDensity, pd.DataFrame]:
    loaded = load_embeddings(static, scan, rows)
    features = l2_normalize(np.stack(loaded["_vector"].to_numpy()))
    validated = fit_validated_crop_density(
        features,
        loaded["landcover"].to_numpy(str),
        loaded["field_uid"].to_numpy(str),
        crop_names=config.mono_crops,
        shrinkage_candidates=config.shrinkage_candidates,
        max_folds=config.max_folds,
        seed=config.random_seed,
    )
    controls = loaded[
        [
            "field_uid",
            "pixel_id",
            "landcover",
            "pixel_x_index",
            "pixel_y_index",
        ]
    ].copy()
    controls["window_id"] = scan.window_id
    controls["fold"] = validated.sample_folds
    controls["_feature"] = list(features)
    controls["_log_scores"] = list(validated.oof_log_scores)
    controls["minimum_squared_mahalanobis"] = (
        -2.0
        * validated.temperature
        * validated.oof_log_scores.max(axis=1)
    )
    probabilities = crop_posteriors(validated.oof_log_scores)
    probability_columns = _crop_probability_columns(config.mono_crops)
    for crop_index, crop in enumerate(config.mono_crops):
        controls[probability_columns[crop]] = probabilities[:, crop_index]
    return validated, controls


def _pair_control_rows(
    controls: pd.DataFrame,
    validated: ValidatedCropDensity,
    config: DNAWorkflowConfig,
) -> pd.DataFrame:
    crop_index = {crop: index for index, crop in enumerate(config.mono_crops)}
    probability_columns = _crop_probability_columns(config.mono_crops)
    rows = []
    for pair_key, (parent_a, parent_b) in config.pair_map.items():
        endpoints = controls[controls["landcover"].isin((parent_a, parent_b))]
        for field_uid, group in endpoints.groupby("field_uid", sort=True):
            score_matrix = np.stack(group["_log_scores"].to_numpy())
            pair_fit = fit_mosaic_share(
                score_matrix[:, crop_index[parent_a]],
                score_matrix[:, crop_index[parent_b]],
                grid_size=config.mixture_grid_size,
            )
            true_label = str(group["landcover"].iloc[0])
            expected_share = 1.0 if true_label == parent_a else 0.0
            named_mass = (
                group[probability_columns[parent_a]].to_numpy(np.float64)
                + group[probability_columns[parent_b]].to_numpy(np.float64)
            )
            rows.append(
                {
                    "window_id": str(group["window_id"].iloc[0]),
                    "field_uid": str(field_uid),
                    "true_crop": true_label,
                    "pair_key": pair_key,
                    "parent_a": parent_a,
                    "parent_b": parent_b,
                    "pixel_count": len(group),
                    "pixel_count_bin": _pixel_count_bin(len(group)),
                    "expected_parent_a_share": expected_share,
                    "mosaic_parent_a_share": pair_fit.parent_a_share,
                    "endpoint_absolute_error": abs(
                        pair_fit.parent_a_share - expected_share
                    ),
                    "log_evidence_over_pure": pair_fit.log_evidence_over_pure,
                    "log_evidence_per_pixel": (
                        pair_fit.log_evidence_over_pure / len(group)
                    ),
                    "named_parent_mass": float(named_mass.mean()),
                    "minimum_mahalanobis_p90": float(
                        group["minimum_squared_mahalanobis"].quantile(0.90)
                    ),
                    "fold": int(group["fold"].iloc[0]),
                    "field_balanced_accuracy": validated.field_balanced_accuracy,
                }
            )
    return pd.DataFrame(rows)


def _synthetic_pair_controls(
    controls: pd.DataFrame,
    config: DNAWorkflowConfig,
) -> pd.DataFrame:
    """Combine held-out pure-field pixels at known ratios.

    These controls validate the mosaic estimator, not physical sub-pixel crop
    abundance or the behavior of a real intercropped canopy.
    """

    crop_index = {crop: index for index, crop in enumerate(config.mono_crops)}
    records = []
    target_pixels = 20
    for pair_index, (pair_key, (parent_a, parent_b)) in enumerate(
        config.pair_map.items()
    ):
        fields_a = sorted(
            controls.loc[controls["landcover"].eq(parent_a), "field_uid"].unique()
        )
        fields_b = sorted(
            controls.loc[controls["landcover"].eq(parent_b), "field_uid"].unique()
        )
        pair_count = min(len(fields_a), len(fields_b))
        rng = np.random.default_rng(config.random_seed + pair_index)
        for pair_position in range(pair_count):
            rows_a = controls[controls["field_uid"].eq(fields_a[pair_position])]
            rows_b = controls[controls["field_uid"].eq(fields_b[pair_position])]
            score_a = np.stack(rows_a["_log_scores"].to_numpy())
            score_b = np.stack(rows_b["_log_scores"].to_numpy())
            for true_share in (0.25, 0.50, 0.75):
                count_a = int(round(target_pixels * true_share))
                count_b = target_pixels - count_a
                selected_a = rng.choice(len(score_a), count_a, replace=len(score_a) < count_a)
                selected_b = rng.choice(len(score_b), count_b, replace=len(score_b) < count_b)
                combined = np.vstack((score_a[selected_a], score_b[selected_b]))
                fit = fit_mosaic_share(
                    combined[:, crop_index[parent_a]],
                    combined[:, crop_index[parent_b]],
                    grid_size=config.mixture_grid_size,
                )
                records.append(
                    {
                        "window_id": str(controls["window_id"].iloc[0]),
                        "pair_key": pair_key,
                        "parent_a": parent_a,
                        "parent_b": parent_b,
                        "source_field_a": fields_a[pair_position],
                        "source_field_b": fields_b[pair_position],
                        "true_parent_a_share": true_share,
                        "estimated_parent_a_share": fit.parent_a_share,
                        "absolute_error": abs(fit.parent_a_share - true_share),
                    }
                )
    return pd.DataFrame(records)


def _calibration_tables(
    control_rows: pd.DataFrame,
    synthetic_rows: pd.DataFrame,
    reference_pixel_controls: pd.DataFrame,
    reference_counts: pd.Series,
    validated: ValidatedCropDensity,
    config: DNAWorkflowConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    thresholds = []
    validation = []
    bins = ("1", "2", "3-4", "5-8", "9-16", "17-32", "33-64", "65+")
    for pair_key, (parent_a, parent_b) in config.pair_map.items():
        pair_controls = control_rows[control_rows["pair_key"].eq(pair_key)]
        if pair_controls.empty:
            raise RuntimeError(f"no endpoint controls for {pair_key}")
        all_evidence = pair_controls["log_evidence_per_pixel"].to_numpy(np.float64)
        mass_threshold = _quantile(
            pair_controls["named_parent_mass"],
            config.named_mass_quantile,
            "lower",
        )
        adequacy_threshold = _quantile(
            pair_controls["minimum_mahalanobis_p90"],
            0.95,
            "higher",
        )
        pixel_adequacy_threshold = _quantile(
            reference_pixel_controls["minimum_squared_mahalanobis"],
            0.95,
            "higher",
        )
        for pixel_bin in bins:
            selected = pair_controls[
                pair_controls["pixel_count_bin"].eq(pixel_bin)
            ]["log_evidence_per_pixel"].to_numpy(np.float64)
            calibration_source = "matching_pixel_bin"
            if len(selected) < 5:
                selected = all_evidence
                calibration_source = "all_parent_controls_fallback"
            thresholds.append(
                {
                    "window_id": str(pair_controls["window_id"].iloc[0]),
                    "pair_key": pair_key,
                    "pixel_count_bin": pixel_bin,
                    "evidence_per_pixel_threshold": _quantile(
                        selected, config.evidence_quantile, "higher"
                    ),
                    "named_parent_mass_threshold": mass_threshold,
                    "adequacy_distance_threshold": adequacy_threshold,
                    "pixel_adequacy_distance_threshold": pixel_adequacy_threshold,
                    "null_field_count": len(selected),
                    "calibration_source": calibration_source,
                }
            )

        synthetic = synthetic_rows[synthetic_rows["pair_key"].eq(pair_key)]
        endpoint_mae = float(pair_controls["endpoint_absolute_error"].mean())
        synthetic_mae = (
            float(synthetic["absolute_error"].mean())
            if not synthetic.empty
            else float("nan")
        )
        minimum_fields = int(min(reference_counts[parent_a], reference_counts[parent_b]))
        gate = bool(
            minimum_fields >= config.min_reference_fields_per_crop
            and validated.fold_count >= 2
            and validated.field_balanced_accuracy >= config.min_balanced_accuracy
            and endpoint_mae <= config.max_endpoint_mae
            and np.isfinite(synthetic_mae)
            and synthetic_mae <= config.max_synthetic_mae
        )
        validation.append(
            {
                "window_id": str(pair_controls["window_id"].iloc[0]),
                "pair_key": pair_key,
                "parent_a": parent_a,
                "parent_b": parent_b,
                "reference_fields_parent_a": int(reference_counts[parent_a]),
                "reference_fields_parent_b": int(reference_counts[parent_b]),
                "fold_count": validated.fold_count,
                "chosen_shrinkage": validated.chosen_shrinkage,
                "temperature": validated.temperature,
                "field_balanced_accuracy": validated.field_balanced_accuracy,
                "field_log_loss": validated.field_log_loss,
                "endpoint_mae": endpoint_mae,
                "synthetic_mosaic_mae": synthetic_mae,
                "validation_gate_passed": gate,
            }
        )
    return pd.DataFrame(thresholds), pd.DataFrame(validation)


def _score_window_pixels(
    static: StaticArtifacts,
    scan: WindowScan,
    rows: pd.DataFrame,
    model: CropDensityModel,
    config: DNAWorkflowConfig,
) -> pd.DataFrame:
    complete = rows[rows["outcome"].eq("complete")].copy()
    if complete.empty:
        return pd.DataFrame()
    loaded = load_embeddings(static, scan, complete)
    features = l2_normalize(np.stack(loaded["_vector"].to_numpy()))
    log_scores = crop_log_scores(model, features)
    probabilities = crop_posteriors(log_scores)
    loaded["_feature"] = list(features)
    loaded["_log_scores"] = list(log_scores)
    probability_columns = _crop_probability_columns(config.mono_crops)
    for crop_index, crop in enumerate(config.mono_crops):
        loaded[probability_columns[crop]] = probabilities[:, crop_index]
    loaded["best_crop"] = np.asarray(config.mono_crops)[
        np.argmax(probabilities, axis=1)
    ]
    loaded["best_crop_probability"] = probabilities.max(axis=1)
    loaded["minimum_squared_mahalanobis"] = -2.0 * model.temperature * log_scores.max(
        axis=1
    )

    crop_index = {crop: index for index, crop in enumerate(config.mono_crops)}
    for pair_key, (parent_a, parent_b) in config.pair_map.items():
        prefix = f"{pair_key}_"
        probability_a = probabilities[:, crop_index[parent_a]]
        probability_b = probabilities[:, crop_index[parent_b]]
        named_mass = probability_a + probability_b
        conditional = np.divide(
            probability_a,
            named_mass,
            out=np.full_like(named_mass, 0.5),
            where=named_mass > np.finfo(np.float64).eps,
        )
        axis = fit_parent_axis(model, features, parent_a, parent_b)
        loaded[prefix + "parent_a"] = parent_a
        loaded[prefix + "parent_b"] = parent_b
        loaded[prefix + "named_parent_mass"] = named_mass
        loaded[prefix + "conditional_parent_a_probability"] = conditional
        loaded[prefix + "axis_parent_a_share"] = axis.parent_a_share
        loaded[prefix + "axis_in_segment"] = axis.in_segment
        loaded[prefix + "tube_squared_distance"] = axis.tube_squared_distance
    return loaded


def _all_pair_likelihoods(
    score_matrix: np.ndarray,
    crop_names: tuple[str, ...],
    grid_size: int,
) -> dict[tuple[str, str], object]:
    index = {crop: position for position, crop in enumerate(crop_names)}
    return {
        (first, second): fit_mosaic_share(
            score_matrix[:, index[first]],
            score_matrix[:, index[second]],
            grid_size=grid_size,
        )
        for first, second in combinations(crop_names, 2)
    }


def _pair_fit(
    fits: dict[tuple[str, str], object],
    parent_a: str,
    parent_b: str,
):
    if (parent_a, parent_b) in fits:
        return fits[(parent_a, parent_b)], False
    return fits[(parent_b, parent_a)], True


def _score_window_fields(
    pixel_rows: pd.DataFrame,
    model: CropDensityModel,
    controls: pd.DataFrame,
    thresholds: pd.DataFrame,
    validation: pd.DataFrame,
    config: DNAWorkflowConfig,
) -> pd.DataFrame:
    if pixel_rows.empty:
        return pd.DataFrame()
    probability_columns = _crop_probability_columns(config.mono_crops)
    crop_index = {crop: index for index, crop in enumerate(config.mono_crops)}
    records = []
    for field_uid, group in pixel_rows.groupby("field_uid", sort=True):
        label = str(group["landcover"].iloc[0])
        score_matrix = np.stack(group["_log_scores"].to_numpy())
        feature_matrix = np.stack(group["_feature"].to_numpy())
        all_fits = _all_pair_likelihoods(
            score_matrix,
            config.mono_crops,
            config.mixture_grid_size,
        )
        likelihood_by_pair = {
            pair: fit.log_likelihood for pair, fit in all_fits.items()
        }
        pixel_count = len(group)
        pixel_bin = _pixel_count_bin(pixel_count)
        block_ids = (
            group["utm_epsg"].astype(str)
            + "-"
            + (group["pixel_x_index"].astype(np.int64) // config.spatial_block_pixels)
            .astype(str)
            + "-"
            + (group["pixel_y_index"].astype(np.int64) // config.spatial_block_pixels)
            .astype(str)
        ).to_numpy(str)
        named_mixture_pair = config.mixture_label_map.get(label)

        for pair_offset, (pair_key, (parent_a, parent_b)) in enumerate(
            config.pair_map.items()
        ):
            fit, reversed_pair = _pair_fit(all_fits, parent_a, parent_b)
            share = (
                1.0 - fit.parent_a_share if reversed_pair else fit.parent_a_share
            )
            log_evidence = fit.log_evidence_over_pure
            pair_tuple = (parent_a, parent_b)
            pair_likelihood = fit.log_likelihood
            alternatives = {
                other_pair: value
                for other_pair, value in likelihood_by_pair.items()
                if set(other_pair) != set(pair_tuple)
            }
            best_alternative, alternative_likelihood = max(
                alternatives.items(), key=lambda item: item[1]
            )
            pair_margin = float(pair_likelihood - alternative_likelihood)

            named_mass = (
                group[probability_columns[parent_a]].to_numpy(np.float64)
                + group[probability_columns[parent_b]].to_numpy(np.float64)
            )
            field_axis = fit_parent_axis(
                model,
                feature_matrix.mean(axis=0),
                parent_a,
                parent_b,
            )

            ci_low = float("nan")
            ci_high = float("nan")
            spatial_ci_available = len(set(block_ids)) >= 2
            if (
                named_mixture_pair == pair_key
                and spatial_ci_available
                and config.bootstrap_replicates > 0
            ):
                bootstrap = block_bootstrap_mosaic_share(
                    score_matrix[:, crop_index[parent_a]],
                    score_matrix[:, crop_index[parent_b]],
                    block_ids,
                    replicates=config.bootstrap_replicates,
                    seed=(
                        config.random_seed
                        + pair_offset
                        + int(hashlib.sha256(str(field_uid).encode()).hexdigest()[:8], 16)
                    ),
                    grid_size=config.mixture_grid_size,
                )
                ci_low, ci_high = np.quantile(bootstrap, [0.025, 0.975])

            threshold_row = thresholds[
                thresholds["pair_key"].eq(pair_key)
                & thresholds["pixel_count_bin"].eq(pixel_bin)
            ]
            if len(threshold_row) != 1:
                raise RuntimeError(
                    f"expected one threshold for {pair_key}/{pixel_bin}"
                )
            threshold_row = threshold_row.iloc[0]
            evidence_per_pixel = log_evidence / pixel_count
            evidence_per_pixel_threshold = float(
                threshold_row["evidence_per_pixel_threshold"]
            )
            evidence_threshold = evidence_per_pixel_threshold * pixel_count
            mass_threshold = float(threshold_row["named_parent_mass_threshold"])
            adequacy_threshold = float(
                threshold_row["adequacy_distance_threshold"]
            )
            pixel_adequacy_threshold = float(
                threshold_row["pixel_adequacy_distance_threshold"]
            )
            null = controls[
                controls["pair_key"].eq(pair_key)
                & controls["pixel_count_bin"].eq(pixel_bin)
            ]["log_evidence_per_pixel"].to_numpy(np.float64)
            if len(null) < 5:
                null = controls[controls["pair_key"].eq(pair_key)][
                    "log_evidence_per_pixel"
                ].to_numpy(np.float64)
            evidence_tail_probability = empirical_upper_tail_probability(
                evidence_per_pixel, null
            )

            validation_row = validation[validation["pair_key"].eq(pair_key)]
            if len(validation_row) != 1:
                raise RuntimeError(f"expected one validation row for {pair_key}")
            gate = bool(validation_row["validation_gate_passed"].iloc[0])
            mean_named_mass = float(named_mass.mean())
            minimum_mahalanobis_p90 = float(
                group["minimum_squared_mahalanobis"].quantile(0.90)
            )
            is_named_mixture = named_mixture_pair == pair_key
            out_of_model = minimum_mahalanobis_p90 > adequacy_threshold
            out_of_family = bool(
                mean_named_mass < mass_threshold or pair_margin < 0.0
            )
            if not gate:
                call_status = "NO_CALL_VALIDATION_FAILED"
            elif pixel_count < 2:
                call_status = "NO_CALL_ONE_PIXEL"
            elif is_named_mixture and not spatial_ci_available:
                call_status = "NO_CALL_INSUFFICIENT_SPATIAL_BLOCKS"
            elif out_of_model:
                call_status = "OUT_OF_MODEL"
            elif out_of_family:
                call_status = "OUT_OF_FAMILY"
            elif (
                evidence_per_pixel > evidence_per_pixel_threshold
                and config.endpoint_epsilon < share < 1.0 - config.endpoint_epsilon
            ):
                interval_is_interior = bool(
                    not is_named_mixture
                    or (
                        np.isfinite(ci_low)
                        and np.isfinite(ci_high)
                        and ci_low > config.endpoint_epsilon
                        and ci_high < 1.0 - config.endpoint_epsilon
                    )
                )
                call_status = (
                    "INTERCROP_SUPPORTED"
                    if interval_is_interior
                    else "AMBIGUOUS_WIDE_INTERVAL"
                )
            elif share >= 1.0 - config.endpoint_epsilon:
                call_status = "PARENT_A_LIKE"
            elif share <= config.endpoint_epsilon:
                call_status = "PARENT_B_LIKE"
            else:
                call_status = "AMBIGUOUS_NO_MIXTURE_EVIDENCE"

            record = {
                "field_uid": str(field_uid),
                "window_id": str(group["window_id"].iloc[0]),
                "landcover": label,
                "pair_key": pair_key,
                "parent_a": parent_a,
                "parent_b": parent_b,
                "pixel_count": pixel_count,
                "spatial_block_count": len(set(block_ids)),
                "pixel_count_bin": pixel_bin,
                "mosaic_parent_a_share": share,
                "mosaic_ci_low": ci_low,
                "mosaic_ci_high": ci_high,
                "spatial_ci_available": spatial_ci_available,
                "field_axis_parent_a_share": float(field_axis.parent_a_share[0]),
                "field_axis_in_segment": bool(field_axis.in_segment[0]),
                "field_axis_tube_squared_distance": float(
                    field_axis.tube_squared_distance[0]
                ),
                "named_parent_mass": mean_named_mass,
                "other_known_crop_mass": 1.0 - mean_named_mass,
                "minimum_mahalanobis_p90": minimum_mahalanobis_p90,
                "adequacy_distance_threshold": adequacy_threshold,
                "pixel_adequacy_distance_threshold": pixel_adequacy_threshold,
                "out_of_model": out_of_model,
                "log_evidence_over_pure": log_evidence,
                "log_evidence_per_pixel": evidence_per_pixel,
                "evidence_threshold": evidence_threshold,
                "evidence_per_pixel_threshold": evidence_per_pixel_threshold,
                "evidence_tail_probability": evidence_tail_probability,
                "pair_log_margin": pair_margin,
                "best_alternative_pair": " + ".join(best_alternative),
                "validation_gate_passed": gate,
                "out_of_family": out_of_family,
                "call_status": call_status,
            }
            for crop in config.mono_crops:
                record[probability_columns[crop]] = float(
                    group[probability_columns[crop]].mean()
                )
            records.append(record)
    return pd.DataFrame(records)


def _display_pair_for_label(label: str, config: DNAWorkflowConfig) -> str:
    mixture_pair = config.mixture_label_map.get(label)
    if mixture_pair is not None:
        return mixture_pair
    if label == "Irish Potato":
        return "potato_maize"
    return "bean_maize"


def _expand_source_field_scores(
    crosswalk: pd.DataFrame,
    physical_scores: pd.DataFrame,
    availability: pd.DataFrame,
    config: DNAWorkflowConfig,
) -> pd.DataFrame:
    source = crosswalk.copy()
    source["pair_key"] = source["landcover"].map(
        lambda label: _display_pair_for_label(str(label), config)
    )
    source["_join"] = 1
    window_frame = pd.DataFrame({"window_id": config.windows, "_join": 1})
    source = source.merge(window_frame, on="_join", how="outer").drop(columns="_join")

    scores = physical_scores.rename(columns={"field_uid": "analysis_unit_uid"})
    source = source.merge(
        scores,
        on=["analysis_unit_uid", "window_id", "pair_key"],
        how="left",
        suffixes=("_source", ""),
        validate="many_to_one",
    )
    source = source.merge(
        availability,
        on=["analysis_unit_uid", "window_id"],
        how="left",
        validate="many_to_one",
    )
    source["availability_status"] = source["availability_status"].fillna("PENDING")

    missing_score = source["call_status"].isna()
    source.loc[missing_score, "call_status"] = source.loc[
        missing_score, "availability_status"
    ]
    unsupported = ~source["supported_label"]
    source.loc[unsupported, "call_status"] = "UNSUPPORTED_LABEL"
    conflict = source["geometry_label_count"].gt(1) & source["supported_label"]
    source.loc[conflict, "call_status"] = "NO_CALL_CONFLICTING_LABEL_GEOMETRY"

    if len(source) != len(crosswalk) * len(config.windows):
        raise RuntimeError("source field expansion lost field/window rows")
    if source.duplicated(["field_uid", "window_id"]).any():
        raise RuntimeError("source field scores contain duplicate field/window rows")
    source["is_replica_inherited_result"] = source["field_uid"].ne(
        source["analysis_unit_uid"]
    ) & source["mosaic_parent_a_share"].notna()
    return source.sort_values(["field_uid", "window_id"], kind="stable").reset_index(
        drop=True
    )


def run_workflow(
    config: DNAWorkflowConfig,
    *,
    progress: Callable[[str], None] | None = None,
) -> DNAWorkflowResult:
    """Run one frozen analysis snapshot without modifying pipeline artifacts."""

    report = (lambda message: None) if progress is None else progress
    report(f"Loading validated pipeline artifacts from {config.output_dir}")
    static = load_static(config.output_dir)
    crosswalk, analysis_units = build_field_crosswalk(static, config.target_labels)
    retained = set(analysis_units["analysis_unit_uid"].astype(str))
    report(
        f"Frozen source universe: {len(crosswalk):,} source records -> "
        f"{len(analysis_units):,} unique geometry+label analysis units"
    )
    scans = {
        window: scan_window(static, window, retained_field_uids=retained)
        for window in config.windows
    }
    snapshot_id = _snapshot_id(scans, static.run_fingerprint)
    analysis_id = _analysis_id(config)
    raw_rows_by_window = {
        window: _window_rows(static, scans[window], analysis_units)
        for window in config.windows
    }
    rows_by_window, availability = _longitudinal_rows(
        raw_rows_by_window,
        scans,
        analysis_units,
        config,
    )
    reference_rows = _common_reference_rows(static, rows_by_window, config)

    pixel_parts = []
    field_parts = []
    control_parts = []
    synthetic_parts = []
    validation_parts = []
    summary_rows = []
    models: dict[str, CropDensityModel] = {}

    for window in config.windows:
        report(
            f"{window}: fitting held-out references and scoring the frozen longitudinal cohort"
        )
        rows = rows_by_window[window]
        validated, reference_pixel_controls = _fit_reference_window(
            static,
            scans[window],
            reference_rows[window],
            config,
        )
        models[window] = validated.model
        pair_controls = _pair_control_rows(
            reference_pixel_controls,
            validated,
            config,
        )
        synthetic = _synthetic_pair_controls(reference_pixel_controls, config)
        reference_counts = reference_pixel_controls.groupby("landcover")[
            "field_uid"
        ].nunique().reindex(config.mono_crops, fill_value=0)
        thresholds, validation = _calibration_tables(
            pair_controls,
            synthetic,
            reference_pixel_controls,
            reference_counts,
            validated,
            config,
        )

        scored_pixels = _score_window_pixels(
            static,
            scans[window],
            rows,
            validated.model,
            config,
        )
        scored_fields = _score_window_fields(
            scored_pixels,
            validated.model,
            pair_controls,
            thresholds,
            validation,
            config,
        )
        public_pixels = scored_pixels.drop(
            columns=["_feature", "_log_scores", "_vector"],
            errors="ignore",
        )
        pixel_parts.append(public_pixels)
        field_parts.append(scored_fields)
        control_parts.append(pair_controls)
        synthetic_parts.append(synthetic)
        validation_parts.append(validation.merge(thresholds, on=["window_id", "pair_key"]))
        summary_rows.append(
            {
                "window_id": window,
                "published_analysis_units": int(
                    analysis_units["analysis_unit_uid"].isin(scans[window].published_fields).sum()
                ),
                "scoreable_analysis_units": int(scored_fields["field_uid"].nunique()),
                "scoreable_canonical_pixels": int(
                    public_pixels[["field_uid", "pixel_id"]].drop_duplicates().shape[0]
                ),
                "reference_pixels": len(reference_pixel_controls),
                "reference_fields": reference_pixel_controls["field_uid"].nunique(),
                "pipeline_complete_at_snapshot": (config.output_dir / "COMPLETED.json").is_file(),
                "snapshot_id": snapshot_id,
            }
        )
        report(
            f"{window}: scored {scored_fields['field_uid'].nunique():,} physical fields / "
            f"{len(scored_pixels):,} field-pixel rows"
        )

    pixel_scores = pd.concat(pixel_parts, ignore_index=True)
    physical_scores = pd.concat(field_parts, ignore_index=True)
    reference_controls = pd.concat(control_parts, ignore_index=True)
    synthetic_controls = pd.concat(synthetic_parts, ignore_index=True)
    validation_metrics = pd.concat(validation_parts, ignore_index=True)
    source_scores = _expand_source_field_scores(
        crosswalk,
        physical_scores,
        availability,
        config,
    )
    window_summary = pd.DataFrame(summary_rows)

    if pixel_scores.duplicated(["field_uid", "pixel_id", "window_id"]).any():
        raise RuntimeError("pixel scores contain duplicate canonical pixel/window rows")
    if physical_scores.duplicated(["field_uid", "window_id", "pair_key"]).any():
        raise RuntimeError("physical field scores contain duplicate field/window/pair rows")
    report("Numerical analysis complete; tables are ready to save")
    return DNAWorkflowResult(
        config=config,
        static=static,
        scans=scans,
        field_crosswalk=crosswalk,
        analysis_units=analysis_units,
        pixel_scores=pixel_scores,
        physical_field_pair_scores=physical_scores,
        source_field_scores=source_scores,
        reference_controls=reference_controls,
        synthetic_controls=synthetic_controls,
        validation_metrics=validation_metrics,
        window_summary=window_summary,
        models=models,
        snapshot_id=snapshot_id,
        analysis_id=analysis_id,
    )


def snapshot_export_dir(result: DNAWorkflowResult) -> Path:
    return (
        result.config.export_dir
        / result.static.run_fingerprint[:12]
        / result.snapshot_id
        / result.analysis_id
    )


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, path)


def _atomic_json(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def save_workflow_tables(result: DNAWorkflowResult) -> tuple[Path, dict[str, object]]:
    """Persist every numerical result before potentially long figure export."""

    root = snapshot_export_dir(result)
    root.mkdir(parents=True, exist_ok=True)
    # A rerun must not inherit success from an earlier interrupted export.
    (root / "COMPLETED.json").unlink(missing_ok=True)
    (root / "SMOKE_COMPLETE.json").unlink(missing_ok=True)
    (root / "TABLES_COMPLETE.json").unlink(missing_ok=True)
    (root / "analysis_manifest.json").unlink(missing_ok=True)
    tables = root / "tables"
    for window in result.config.windows:
        _atomic_parquet(
            result.pixel_scores[result.pixel_scores["window_id"].eq(window)],
            tables / "pixel_scores" / f"window_id={window}" / "part-00000.parquet",
        )
    _atomic_parquet(
        result.physical_field_pair_scores,
        tables / "physical_field_pair_scores.parquet",
    )
    _atomic_parquet(result.source_field_scores, tables / "source_field_scores.parquet")
    _atomic_parquet(result.field_crosswalk, tables / "source_field_manifest.parquet")
    _atomic_parquet(result.reference_controls, tables / "reference_controls.parquet")
    _atomic_parquet(result.synthetic_controls, tables / "synthetic_controls.parquet")
    _atomic_parquet(result.validation_metrics, tables / "validation_metrics.parquet")
    _atomic_parquet(result.window_summary, tables / "window_summary.parquet")

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    for window, model in result.models.items():
        target = model_dir / f"{window}.npz"
        temporary = model_dir / f"{window}.npz.part"
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                crop_names=np.asarray(model.crop_names),
                feature_origin=model.feature_origin,
                whitener=model.whitener,
                crop_centers=model.crop_centers,
                covariance_eigenvalues=model.covariance_eigenvalues,
                shrinkage=np.asarray(model.shrinkage),
                temperature=np.asarray(model.temperature),
            )
        os.replace(temporary, target)

    config_payload = asdict(result.config)
    config_payload["output_dir"] = str(result.config.output_dir)
    config_payload["export_dir"] = str(result.config.export_dir)
    manifest: dict[str, object] = {
        "analysis_name": "intercropping_parentage_likelihood_v1",
        "analysis_version": 1,
        "run_fingerprint": result.static.run_fingerprint,
        "snapshot_id": result.snapshot_id,
        "analysis_id": result.analysis_id,
        "pipeline_complete_at_snapshot": (
            result.config.output_dir / "COMPLETED.json"
        ).is_file(),
        "config": config_payload,
        "windows": list(result.config.windows),
        "longitudinal_policy": (
            "fully published physical fields; exact complete field/pixel memberships "
            "intersected across every window"
        ),
        "source_field_rows": len(result.field_crosswalk),
        "source_field_window_rows": len(result.source_field_scores),
        "canonical_analysis_units": len(result.analysis_units),
        "scored_canonical_field_windows": int(
            result.physical_field_pair_scores[["field_uid", "window_id"]]
            .drop_duplicates()
            .shape[0]
        ),
        "scored_canonical_pixel_windows": len(result.pixel_scores),
        "shards": {
            window: [path.name for path in scan.files]
            for window, scan in result.scans.items()
        },
        "scientific_interpretation": (
            "Shares are relative TESSERA embedding-signature attribution. "
            "They are not calibrated planted-area, plant-count, biomass, yield, "
            "or sub-pixel abundance percentages. Evidence tail probabilities are "
            "held-out-reference diagnostics, not formal exchangeable p-values."
        ),
        "tables_complete": True,
        "figures_complete": False,
    }
    _atomic_json(manifest, root / "analysis_manifest.json")
    _atomic_json(
        {
            "snapshot_id": result.snapshot_id,
            "analysis_id": result.analysis_id,
            "tables_complete": True,
        },
        root / "TABLES_COMPLETE.json",
    )
    return root, manifest


def finalize_workflow_export(
    root: Path,
    manifest: dict[str, object],
    *,
    field_plot_count: int,
    plot_manifest_path: str,
    gallery_complete: bool = True,
    report_limit: int | None = None,
) -> None:
    completed = dict(manifest)
    completed["figures_complete"] = bool(gallery_complete)
    completed["gallery_complete"] = bool(gallery_complete)
    completed["field_plot_count"] = int(field_plot_count)
    completed["plot_manifest_path"] = plot_manifest_path
    completed["report_limit"] = report_limit
    _atomic_json(completed, root / "analysis_manifest.json")
    marker = {
        "snapshot_id": completed["snapshot_id"],
        "analysis_id": completed["analysis_id"],
        "tables_complete": True,
        "figures_complete": bool(gallery_complete),
        "gallery_complete": bool(gallery_complete),
        "field_plot_count": int(field_plot_count),
        "report_limit": report_limit,
    }
    if gallery_complete:
        (root / "SMOKE_COMPLETE.json").unlink(missing_ok=True)
        _atomic_json(marker, root / "COMPLETED.json")
    else:
        (root / "COMPLETED.json").unlink(missing_ok=True)
        _atomic_json(marker, root / "SMOKE_COMPLETE.json")


def save_field_plot_index(
    result: DNAWorkflowResult,
    root: Path,
    field_paths: dict[str, str],
    *,
    report_limited: bool = False,
) -> pd.DataFrame:
    """Map every original source row to its canonical report or a reason."""

    index = result.field_crosswalk[
        [
            "field_uid",
            "analysis_unit_uid",
            "landcover",
            "record_role",
            "source_replica_count",
            "geometry_label_count",
        ]
    ].copy()
    resolved_root = root.resolve()

    def relative_report(unit: str) -> str | None:
        value = field_paths.get(str(unit))
        if value is None:
            return None
        return str(Path(value).resolve().relative_to(resolved_root))

    index["report_path"] = index["analysis_unit_uid"].map(relative_report)
    index["report_available"] = index["report_path"].notna()
    index["report_status"] = "AVAILABLE"
    index.loc[~index["report_available"], "report_status"] = (
        "UNAVAILABLE_NOT_COMPLETE_IN_ALL_WINDOWS"
    )
    complete_units = set(
        result.physical_field_pair_scores.groupby("field_uid")["window_id"]
        .nunique()
        .loc[lambda counts: counts.eq(len(result.config.windows))]
        .index.astype(str)
    )
    if report_limited:
        limited = (
            ~index["report_available"]
            & index["analysis_unit_uid"].astype(str).isin(complete_units)
        )
        index.loc[limited, "report_status"] = "NOT_RENDERED_REPORT_LIMIT"
    index.loc[index["record_role"].eq("unsupported_label"), "report_status"] = (
        "UNSUPPORTED_LABEL"
    )
    index.loc[index["geometry_label_count"].gt(1), "report_status"] = (
        "NO_CALL_CONFLICTING_LABEL_GEOMETRY"
    )
    _atomic_parquet(index, root / "tables" / "field_plot_index.parquet")
    return index


__all__ = [
    "DNAWorkflowConfig",
    "DNAWorkflowResult",
    "build_field_crosswalk",
    "default_config",
    "finalize_workflow_export",
    "run_workflow",
    "save_field_plot_index",
    "save_workflow_tables",
    "snapshot_export_dir",
]
