"""Read-only evidence selection for the intercropping presentation notebook.

The functions in this module consume a finalized DNA-analysis export. They do
not load embeddings, fit models, or rerun inference. Representative fields are
selected at the last in-contract window (``w3`` when available) using cohort
medians rather than maximum evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd

from plain_tessera_incremental.notebooks._dna_reporting import choose_display_pair


DEFAULT_ANALYSIS_ROOT = Path(
    "/mnt/noobjam/harvard_tessera_incremental_v2/analysis/"
    "intercropping_parentage_likelihood_v1"
)
WINDOW_PERIODS = {
    "w1": ("2024-09-01", "2025-01-01", 122, "in contract"),
    "w2": ("2024-09-01", "2025-05-01", 242, "in contract"),
    "w3": ("2024-09-01", "2025-09-01", 365, "in contract"),
    "w4": ("2024-09-01", "2026-01-01", 487, "OOD sensitivity"),
}
MIXTURE_LABELS = ("Bean and Maize", "Irish Potato and Maize")


@dataclass(frozen=True)
class PDFEvidencePack:
    """Small, finalized analysis snapshot needed for the presentation."""

    root: Path
    manifest: dict[str, object]
    completion: dict[str, object]
    plot_manifest: dict[str, object]
    field_scores: pd.DataFrame
    display_scores: pd.DataFrame
    validation_metrics: pd.DataFrame
    window_summary: pd.DataFrame
    source_fields: pd.DataFrame


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: set[str], name: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def find_completed_analysis_dir(
    search_root: str | Path = DEFAULT_ANALYSIS_ROOT,
    *,
    explicit: str | Path | None = None,
) -> Path:
    """Return an explicit or most-recent finalized full-gallery directory."""

    if explicit is not None:
        candidate = Path(explicit).expanduser().resolve()
        if candidate.name == "COMPLETED.json":
            candidate = candidate.parent
        if not (candidate / "COMPLETED.json").is_file():
            raise FileNotFoundError(f"no COMPLETED.json in {candidate}")
        return candidate

    root = Path(search_root).expanduser().resolve()
    markers = list(root.glob("**/COMPLETED.json"))
    if not markers:
        raise FileNotFoundError(
            f"no completed intercropping analysis beneath {root}; "
            "finish the parentage-likelihood notebook first"
        )
    marker = max(markers, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    return marker.parent


def display_pair_rows(field_scores: pd.DataFrame) -> pd.DataFrame:
    """Keep the same deterministic display pair used by the field dashboards."""

    _require_columns(
        field_scores,
        {
            "field_uid",
            "window_id",
            "landcover",
            "pair_key",
            "named_parent_mass",
            "log_evidence_over_pure",
        },
        "physical field scores",
    )
    parts = []
    for _, rows in field_scores.groupby("field_uid", sort=True):
        pair_key = choose_display_pair(rows)
        selected = rows[rows["pair_key"].astype(str).eq(pair_key)].copy()
        if selected["window_id"].duplicated().any():
            raise ValueError(f"duplicate display rows for field {selected.iloc[0]['field_uid']}")
        parts.append(selected)
    if not parts:
        raise ValueError("physical field scores are empty")
    return pd.concat(parts, ignore_index=True)


def load_evidence_pack(root: str | Path) -> PDFEvidencePack:
    """Load and cross-check the compact artifacts from one completed export."""

    analysis_root = Path(root).expanduser().resolve()
    completion = _read_json(analysis_root / "COMPLETED.json")
    manifest = _read_json(analysis_root / "analysis_manifest.json")
    if not completion.get("gallery_complete") or not completion.get("figures_complete"):
        raise ValueError("the selected export is not a completed full-gallery run")
    if completion.get("analysis_id") != manifest.get("analysis_id"):
        raise ValueError("COMPLETED.json and analysis_manifest.json disagree")

    plot_manifest_relative = str(
        manifest.get("plot_manifest_path", "figures/plot_manifest.json")
    )
    plot_manifest = _read_json(analysis_root / plot_manifest_relative)
    tables = analysis_root / "tables"
    field_scores = pd.read_parquet(tables / "physical_field_pair_scores.parquet")
    validation = pd.read_parquet(tables / "validation_metrics.parquet")
    window_summary = pd.read_parquet(tables / "window_summary.parquet")
    source_fields = pd.read_parquet(tables / "source_field_manifest.parquet")
    display_scores = display_pair_rows(field_scores)

    _require_columns(
        display_scores,
        {
            "field_uid",
            "window_id",
            "landcover",
            "pixel_count",
            "mosaic_parent_a_share",
            "mosaic_ci_low",
            "mosaic_ci_high",
            "named_parent_mass",
            "out_of_model",
            "log_evidence_over_pure",
            "evidence_threshold",
            "validation_gate_passed",
            "call_status",
        },
        "display field scores",
    )
    _require_columns(
        validation,
        {
            "window_id",
            "pair_key",
            "parent_a",
            "parent_b",
            "reference_fields_parent_a",
            "reference_fields_parent_b",
            "fold_count",
            "field_balanced_accuracy",
            "endpoint_mae",
            "synthetic_mosaic_mae",
            "validation_gate_passed",
        },
        "validation metrics",
    )
    windows = tuple(str(window) for window in manifest.get("windows", ()))
    if not windows:
        raise ValueError("analysis manifest contains no windows")
    observed = set(display_scores["window_id"].astype(str))
    if set(windows) != observed:
        raise ValueError(f"manifest windows {windows} disagree with scores {sorted(observed)}")

    reports = plot_manifest.get("field_reports", [])
    report_ids = {str(record["field_uid"]) for record in reports}
    score_ids = set(display_scores["field_uid"].astype(str))
    if report_ids != score_ids:
        raise ValueError("field-report manifest does not match the scored physical fields")

    return PDFEvidencePack(
        root=analysis_root,
        manifest=manifest,
        completion=completion,
        plot_manifest=plot_manifest,
        field_scores=field_scores,
        display_scores=display_scores,
        validation_metrics=validation,
        window_summary=window_summary,
        source_fields=source_fields,
    )


def mixture_outcomes(pack: PDFEvidencePack) -> pd.DataFrame:
    """Summarize mixture labels without dropping OOD or ambiguous fields."""

    rows = pack.display_scores[
        pack.display_scores["landcover"].astype(str).isin(MIXTURE_LABELS)
    ].copy()
    rows["supported"] = rows["call_status"].astype(str).eq("INTERCROP_SUPPORTED")
    rows["evidence_gap"] = (
        rows["log_evidence_over_pure"] - rows["evidence_threshold"]
    )
    summary = (
        rows.groupby(["landcover", "window_id"], as_index=False)
        .agg(
            fields=("field_uid", "nunique"),
            supported_fields=("supported", "sum"),
            out_of_model_fields=("out_of_model", "sum"),
            median_parent_a_share=("mosaic_parent_a_share", "median"),
            q25_parent_a_share=("mosaic_parent_a_share", lambda x: x.quantile(0.25)),
            q75_parent_a_share=("mosaic_parent_a_share", lambda x: x.quantile(0.75)),
            median_evidence_gap=("evidence_gap", "median"),
            median_named_parent_mass=("named_parent_mass", "median"),
        )
        .reset_index(drop=True)
    )
    summary["supported_rate"] = summary["supported_fields"] / summary["fields"]
    summary["out_of_model_rate"] = summary["out_of_model_fields"] / summary["fields"]
    order = {window: index for index, window in enumerate(pack.manifest["windows"])}
    summary["_window_order"] = summary["window_id"].astype(str).map(order)
    return summary.sort_values(
        ["landcover", "_window_order"], kind="stable"
    ).drop(columns="_window_order")


def _dataframe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    return json.loads(frame.to_json(orient="records", double_precision=5))


def presentation_facts(pack: PDFEvidencePack) -> dict[str, object]:
    """Return one JSON-safe payload containing every numerical PDF claim."""

    windows = tuple(str(window) for window in pack.manifest["windows"])
    physical_fields = pack.display_scores[["field_uid", "landcover"]].drop_duplicates()
    field_counts = (
        physical_fields.groupby("landcover", as_index=False)
        .agg(physical_fields=("field_uid", "nunique"))
        .sort_values("physical_fields", ascending=False, kind="stable")
    )

    statuses = (
        pack.display_scores.groupby(["window_id", "call_status"], as_index=False)
        .agg(fields=("field_uid", "nunique"))
    )
    statuses["_window_order"] = statuses["window_id"].astype(str).map(
        {window: index for index, window in enumerate(windows)}
    )
    statuses = statuses.sort_values(
        ["_window_order", "call_status"], kind="stable"
    ).drop(columns="_window_order")

    controls = pack.display_scores[
        ~pack.display_scores["landcover"].astype(str).str.contains(" and ", regex=False)
    ].copy()
    controls["false_positive"] = controls["call_status"].astype(str).eq(
        "INTERCROP_SUPPORTED"
    )
    control_summary = (
        controls.groupby("window_id", as_index=False)
        .agg(
            monocrop_fields=("field_uid", "nunique"),
            false_positive_fields=("false_positive", "sum"),
        )
    )
    control_summary["false_positive_rate"] = (
        control_summary["false_positive_fields"] / control_summary["monocrop_fields"]
    )

    validation_columns = [
        "window_id",
        "pair_key",
        "parent_a",
        "parent_b",
        "reference_fields_parent_a",
        "reference_fields_parent_b",
        "fold_count",
        "field_balanced_accuracy",
        "endpoint_mae",
        "synthetic_mosaic_mae",
        "validation_gate_passed",
    ]
    periods = []
    for window in windows:
        start, end, duration, status = WINDOW_PERIODS.get(
            window, (None, None, None, "unspecified")
        )
        periods.append(
            {
                "window_id": window,
                "start": start,
                "end_exclusive": end,
                "duration_days": duration,
                "contract_status": status,
            }
        )

    return {
        "analysis": {
            "analysis_name": pack.manifest.get("analysis_name"),
            "run_fingerprint": pack.manifest.get("run_fingerprint"),
            "snapshot_id": pack.manifest.get("snapshot_id"),
            "analysis_id": pack.manifest.get("analysis_id"),
            "pipeline_complete_at_snapshot": pack.manifest.get(
                "pipeline_complete_at_snapshot"
            ),
            "source_field_rows": pack.manifest.get("source_field_rows"),
            "canonical_analysis_units": pack.manifest.get("canonical_analysis_units"),
            "scored_physical_fields": int(physical_fields["field_uid"].nunique()),
            "field_reports": pack.completion.get("field_plot_count"),
        },
        "windows": periods,
        "field_counts": _dataframe_records(field_counts),
        "window_summary": _dataframe_records(pack.window_summary),
        "validation": _dataframe_records(
            pack.validation_metrics[validation_columns].sort_values(
                ["window_id", "pair_key"], kind="stable"
            )
        ),
        "mixture_outcomes": _dataframe_records(mixture_outcomes(pack)),
        "monocrop_negative_control": _dataframe_records(control_summary),
        "all_call_statuses": _dataframe_records(statuses),
        "interpretation_boundary": pack.manifest.get("scientific_interpretation"),
    }


def _typical_row(rows: pd.DataFrame) -> pd.Series | None:
    if rows.empty:
        return None
    ranked = rows.copy()
    ranked["_evidence_gap"] = (
        ranked["log_evidence_over_pure"] - ranked["evidence_threshold"]
    )
    target = float(ranked["_evidence_gap"].median())
    ranked["_median_distance"] = (ranked["_evidence_gap"] - target).abs()
    ranked = ranked.sort_values(
        ["_median_distance", "pixel_count", "field_uid"],
        ascending=[True, False, True],
        kind="stable",
    )
    return ranked.iloc[0]


def select_representative_fields(pack: PDFEvidencePack) -> pd.DataFrame:
    """Choose typical mixture, negative-control, and guardrail dashboards."""

    windows = tuple(str(window) for window in pack.manifest["windows"])
    anchor = "w3" if "w3" in windows else windows[-1]
    available = {
        str(record["field_uid"]) for record in pack.plot_manifest["field_reports"]
    }
    rows = pack.display_scores[
        pack.display_scores["window_id"].astype(str).eq(anchor)
        & pack.display_scores["field_uid"].astype(str).isin(available)
    ].copy()
    rows["_evidence_gap"] = rows["log_evidence_over_pure"] - rows["evidence_threshold"]
    selected: list[dict[str, object]] = []
    used: set[str] = set()

    for label in MIXTURE_LABELS:
        label_rows = rows[rows["landcover"].astype(str).eq(label)]
        supported = label_rows[
            label_rows["call_status"].astype(str).eq("INTERCROP_SUPPORTED")
            & ~label_rows["out_of_model"].astype(bool)
        ]
        pool = supported
        basis = "cohort-median evidence among supported in-model fields"
        if pool.empty:
            pool = label_rows[~label_rows["out_of_model"].astype(bool)]
            basis = "cohort-median evidence among in-model fields; none were supported"
        row = _typical_row(pool)
        if row is not None:
            field_uid = str(row["field_uid"])
            used.add(field_uid)
            selected.append(
                {
                    "role": f"representative {label}",
                    "selection_basis": basis,
                    **row.to_dict(),
                }
            )

    control_rows = rows[
        ~rows["landcover"].astype(str).str.contains(" and ", regex=False)
        & ~rows["out_of_model"].astype(bool)
        & ~rows["call_status"].astype(str).eq("INTERCROP_SUPPORTED")
        & ~rows["field_uid"].astype(str).isin(used)
    ].copy()
    if not control_rows.empty:
        cutoff = float(control_rows["pixel_count"].quantile(0.75))
        high_pixel_controls = control_rows[control_rows["pixel_count"].ge(cutoff)]
        row = _typical_row(high_pixel_controls)
        if row is not None:
            field_uid = str(row["field_uid"])
            used.add(field_uid)
            selected.append(
                {
                    "role": "monocrop negative control",
                    "selection_basis": (
                        "cohort-median evidence within the top pixel-count quartile "
                        "of in-model monocrop negatives"
                    ),
                    **row.to_dict(),
                }
            )

    guardrail_rows = rows[
        rows["out_of_model"].astype(bool)
        & ~rows["field_uid"].astype(str).isin(used)
    ].copy()
    guardrail_basis = "highest-pixel OUT_OF_MODEL field"
    if guardrail_rows.empty:
        guardrail_rows = rows[
            rows["call_status"].astype(str).str.startswith("AMBIGUOUS")
            & ~rows["field_uid"].astype(str).isin(used)
        ].copy()
        guardrail_basis = "highest-pixel ambiguous field; no OUT_OF_MODEL field available"
    if not guardrail_rows.empty:
        row = guardrail_rows.sort_values(
            ["pixel_count", "field_uid"], ascending=[False, True], kind="stable"
        ).iloc[0]
        selected.append(
            {
                "role": "model guardrail example",
                "selection_basis": guardrail_basis,
                **row.to_dict(),
            }
        )

    if not selected:
        raise ValueError(f"no representative dashboards were available at {anchor}")
    result = pd.DataFrame(selected)
    result.insert(1, "selection_window", anchor)
    columns = [
        "role",
        "selection_window",
        "field_uid",
        "landcover",
        "pair_key",
        "pixel_count",
        "mosaic_parent_a_share",
        "mosaic_ci_low",
        "mosaic_ci_high",
        "named_parent_mass",
        "_evidence_gap",
        "call_status",
        "selection_basis",
    ]
    return result[columns].rename(columns={"_evidence_gap": "evidence_gap"})


def _safe_figure_path(figures_root: Path, relative: str) -> Path:
    path = (figures_root / relative).resolve()
    try:
        path.relative_to(figures_root.resolve())
    except ValueError as error:
        raise ValueError(f"figure path escapes the export: {relative}") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def overview_figure_path(pack: PDFEvidencePack) -> Path:
    record = pack.plot_manifest["overview"]
    return _safe_figure_path(pack.root / "figures", str(record["path"]))


def field_figure_path(pack: PDFEvidencePack, field_uid: str) -> Path:
    records = {
        str(record["field_uid"]): record
        for record in pack.plot_manifest["field_reports"]
    }
    if str(field_uid) not in records:
        raise KeyError(f"no field report for {field_uid}")
    return _safe_figure_path(
        pack.root / "figures", str(records[str(field_uid)]["path"])
    )


def plot_mixture_outcomes(
    summary: pd.DataFrame,
    *,
    windows: Sequence[str],
) -> Figure:
    """Plot supported/OOD rates and the longitudinal parent-A balance."""

    order = tuple(str(window) for window in windows)
    x = np.arange(len(order))
    colors = ("#2E8B57", "#8E63B0")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for color, label in zip(colors, MIXTURE_LABELS, strict=True):
        rows = summary[summary["landcover"].astype(str).eq(label)].set_index("window_id")
        rows = rows.reindex(order)
        axes[0].plot(
            x,
            100 * rows["supported_rate"],
            color=color,
            marker="o",
            linewidth=2,
            label=f"{label}: supported",
        )
        axes[0].plot(
            x,
            100 * rows["out_of_model_rate"],
            color=color,
            marker="x",
            linestyle="--",
            linewidth=1.5,
            label=f"{label}: OOD",
        )
        median = rows["median_parent_a_share"].to_numpy(np.float64)
        low = rows["q25_parent_a_share"].to_numpy(np.float64)
        high = rows["q75_parent_a_share"].to_numpy(np.float64)
        axes[1].fill_between(x, low, high, color=color, alpha=0.16)
        axes[1].plot(x, median, color=color, marker="o", linewidth=2, label=label)

    axes[0].set_xticks(x, order)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("fields (%)")
    axes[0].set_title("Supported intercropping calls and model guardrails")
    axes[0].legend(fontsize=7)
    axes[1].set_xticks(x, order)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("median parent-A signature share")
    axes[1].set_title("Field balance through cumulative windows (IQR shaded)")
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.18)
    figure.tight_layout()
    return figure


__all__ = [
    "DEFAULT_ANALYSIS_ROOT",
    "PDFEvidencePack",
    "display_pair_rows",
    "field_figure_path",
    "find_completed_analysis_dir",
    "load_evidence_pack",
    "mixture_outcomes",
    "overview_figure_path",
    "plot_mixture_outcomes",
    "presentation_facts",
    "select_representative_fields",
]
