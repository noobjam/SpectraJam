"""Meaningful, auditable figures for the intercropping DNA analysis.

This module only renders already-computed evidence tables.  It does not fit or
recalibrate the likelihood model.  Pixel coordinates follow the pipeline's
global 10 m UTM convention: ``pixel_x_index`` and ``pixel_y_index`` are integer
cell indices, so a cell occupies ``[x*10, (x+1)*10]`` by
``[y*10, (y+1)*10]`` projected metres in ``utm_epsg``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt as shapely_wkt
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shapely_transform


DEFAULT_WINDOWS = ("w1", "w2", "w3", "w4")
PIXEL_SIZE_M = 10.0
PAIR_PREFIXES = ("bean_maize", "potato_maize")
CROP_PROBABILITY_COLUMNS = {
    "Maize": "prob_maize",
    "Bean": "prob_bean",
    "Irish Potato": "prob_irish_potato",
    "Rice": "prob_rice",
}
CROP_COLORS = {
    "Maize": "#E3A51A",
    "Bean": "#2E8B57",
    "Irish Potato": "#8E63B0",
    "Rice": "#3B82C4",
}
STATUS_COLORS = (
    "#2E8B57",
    "#3B82C4",
    "#E3A51A",
    "#C44E52",
    "#8172B2",
    "#7F7F7F",
)

PIXEL_BASE_COLUMNS = {
    "field_uid",
    "pixel_id",
    "window_id",
    "landcover",
    "utm_epsg",
    "pixel_x_index",
    "pixel_y_index",
    "overlap_field_count",
    "label_conflict",
    "minimum_squared_mahalanobis",
    *CROP_PROBABILITY_COLUMNS.values(),
}
FIELD_COLUMNS = {
    "field_uid",
    "window_id",
    "landcover",
    "pair_key",
    "parent_a",
    "parent_b",
    "pixel_count",
    "mosaic_parent_a_share",
    "mosaic_ci_low",
    "mosaic_ci_high",
    "field_axis_parent_a_share",
    "named_parent_mass",
    "other_known_crop_mass",
    "minimum_mahalanobis_p90",
    "adequacy_distance_threshold",
    "pixel_adequacy_distance_threshold",
    "out_of_model",
    "log_evidence_over_pure",
    "evidence_threshold",
    "evidence_tail_probability",
    "pair_log_margin",
    "best_alternative_pair",
    "validation_gate_passed",
    "call_status",
    *CROP_PROBABILITY_COLUMNS.values(),
}


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _validate_windows(windows: Sequence[str]) -> tuple[str, ...]:
    result = tuple(str(window) for window in windows)
    if not result or any(not window for window in result) or len(set(result)) != len(result):
        raise ValueError("windows must contain unique nonempty names")
    return result


def _single_value(values: pd.Series, name: str) -> object:
    unique = values.dropna().unique()
    if len(unique) != 1:
        raise ValueError(f"expected one {name}, found {unique.tolist()}")
    return unique[0]


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not safe:
        safe = hashlib.sha256(str(value).encode()).hexdigest()[:16]
    return safe


def _pair_pixel_columns(pair_key: str) -> dict[str, str]:
    if pair_key not in PAIR_PREFIXES:
        raise ValueError(f"unsupported pair_key {pair_key!r}; expected {PAIR_PREFIXES}")
    return {
        name: f"{pair_key}_{name}"
        for name in (
            "parent_a",
            "parent_b",
            "named_parent_mass",
            "conditional_parent_a_probability",
            "axis_parent_a_share",
            "tube_squared_distance",
        )
    }


def choose_display_pair(field_rows: pd.DataFrame) -> str:
    """Choose one dashboard pair for a field with deterministic control rules."""

    _require_columns(
        field_rows,
        {
            "field_uid",
            "landcover",
            "pair_key",
            "named_parent_mass",
            "log_evidence_over_pure",
            "window_id",
        },
        "field_rows",
    )
    available = set(field_rows["pair_key"].astype(str))
    if not available:
        raise ValueError("field_rows contains no pair candidates")
    label = str(_single_value(field_rows["landcover"], "landcover")).strip().lower()
    preferred = {
        "bean and maize": "bean_maize",
        "bean": "bean_maize",
        "irish potato and maize": "potato_maize",
        "irish potato": "potato_maize",
    }.get(label)
    if preferred in available:
        return str(preferred)
    if len(available) == 1:
        return next(iter(available))

    ranked = (
        field_rows.assign(pair_key=field_rows["pair_key"].astype(str))
        .groupby("pair_key", as_index=False)
        .agg(
            mean_named_parent_mass=("named_parent_mass", "mean"),
            mean_evidence=("log_evidence_over_pure", "mean"),
        )
        .sort_values(
            ["mean_named_parent_mass", "mean_evidence", "pair_key"],
            ascending=[False, False, True],
            kind="stable",
        )
    )
    return str(ranked.iloc[0]["pair_key"])


def _display_field_rows(field_evidence: pd.DataFrame) -> pd.DataFrame:
    _require_columns(field_evidence, FIELD_COLUMNS, "field_evidence")
    parts = []
    for _, rows in field_evidence.groupby("field_uid", sort=True):
        pair_key = choose_display_pair(rows)
        selected = rows[rows["pair_key"].astype(str).eq(pair_key)].copy()
        if selected["window_id"].duplicated().any():
            raise ValueError(
                f"field {selected['field_uid'].iloc[0]!r} has duplicate pair/window rows"
            )
        parts.append(selected)
    if not parts:
        raise ValueError("field_evidence is empty")
    return pd.concat(parts, ignore_index=True)


def _field_ordered_rows(
    field_uid: str,
    field_evidence: pd.DataFrame,
    windows: tuple[str, ...],
    pair_key: str | None,
) -> tuple[pd.DataFrame, str]:
    rows = field_evidence[field_evidence["field_uid"].astype(str).eq(str(field_uid))]
    if rows.empty:
        raise ValueError(f"field_evidence contains no field {field_uid!r}")
    selected_pair = choose_display_pair(rows) if pair_key is None else str(pair_key)
    rows = rows[rows["pair_key"].astype(str).eq(selected_pair)].copy()
    if rows.empty:
        raise ValueError(f"field {field_uid!r} has no evidence for pair {selected_pair!r}")
    if rows["window_id"].duplicated().any():
        raise ValueError(f"field {field_uid!r} has duplicate rows for {selected_pair!r}")
    missing = set(windows).difference(rows["window_id"].astype(str))
    if missing:
        raise ValueError(f"field {field_uid!r} is missing windows {sorted(missing)}")
    order = {window: index for index, window in enumerate(windows)}
    rows = rows[rows["window_id"].astype(str).isin(windows)].copy()
    rows["_window_order"] = rows["window_id"].astype(str).map(order)
    return rows.sort_values("_window_order", kind="stable"), selected_pair


def _project_field_geometry(fields: pd.DataFrame, field_uid: str, epsg: int) -> BaseGeometry:
    _require_columns(fields, {"field_uid", "wkt"}, "fields")
    selected = fields[fields["field_uid"].astype(str).eq(str(field_uid))]
    if len(selected) != 1:
        raise ValueError(f"expected one fields row for {field_uid!r}, found {len(selected)}")
    geometry = shapely_wkt.loads(str(selected.iloc[0]["wkt"]))
    transformer = Transformer.from_crs(4326, epsg, always_xy=True)
    return shapely_transform(transformer.transform, geometry)


def _iter_polygons(geometry: BaseGeometry):
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in {"MultiPolygon", "GeometryCollection"}:
        for child in geometry.geoms:
            yield from _iter_polygons(child)


def _draw_outline(axis, geometry: BaseGeometry) -> None:
    for polygon in _iter_polygons(geometry):
        x, y = polygon.exterior.xy
        axis.plot(x, y, color="black", linewidth=1.25, zorder=5)
        for interior in polygon.interiors:
            x, y = interior.xy
            axis.plot(x, y, color="black", linewidth=0.8, zorder=5)


def _map_limits(geometry: BaseGeometry, pixels: pd.DataFrame) -> tuple[float, float, float, float]:
    left = pixels["pixel_x_index"].to_numpy(np.float64) * PIXEL_SIZE_M
    bottom = pixels["pixel_y_index"].to_numpy(np.float64) * PIXEL_SIZE_M
    min_x, min_y, max_x, max_y = geometry.bounds
    min_x = min(min_x, float(left.min()))
    min_y = min(min_y, float(bottom.min()))
    max_x = max(max_x, float((left + PIXEL_SIZE_M).max()))
    max_y = max(max_y, float((bottom + PIXEL_SIZE_M).max()))
    padding = max(PIXEL_SIZE_M, 0.04 * max(max_x - min_x, max_y - min_y))
    return min_x - padding, max_x + padding, min_y - padding, max_y + padding


def _draw_pixel_map(
    axis,
    rows: pd.DataFrame,
    geometry: BaseGeometry,
    probability_column: str,
    mass_column: str,
    limits: tuple[float, float, float, float],
) -> PatchCollection:
    values = rows[probability_column].to_numpy(np.float64)
    finite = np.isfinite(values)
    if np.any((values[finite] < 0) | (values[finite] > 1)):
        raise ValueError(f"{probability_column} values must lie in [0, 1]")
    rectangles = [
        Rectangle(
            (
                float(row.pixel_x_index) * PIXEL_SIZE_M,
                float(row.pixel_y_index) * PIXEL_SIZE_M,
            ),
            PIXEL_SIZE_M,
            PIXEL_SIZE_M,
        )
        for row in rows.itertuples(index=False)
    ]
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#D9D9D9")
    collection = PatchCollection(
        rectangles,
        cmap=cmap,
        norm=Normalize(0.0, 1.0),
        edgecolor="white",
        linewidth=0.35,
        zorder=2,
    )
    collection.set_array(np.ma.masked_invalid(values))
    axis.add_collection(collection)

    low_mass = rows[mass_column].to_numpy(np.float64) < 0.5
    shared = rows["overlap_field_count"].to_numpy(np.int64) > 1
    conflict = rows["label_conflict"].astype(bool).to_numpy()
    out_of_model = (
        rows.get("_pixel_out_of_model", pd.Series(False, index=rows.index))
        .astype(bool)
        .to_numpy()
    )
    overlays = (
        (low_mass, "..", "#666666", 0.5),
        (out_of_model, "++", "#7B2CBF", 0.8),
        (shared, "///", "#333333", 0.7),
        (conflict, "xxx", "crimson", 1.0),
    )
    for mask, hatch, color, width in overlays:
        if not mask.any():
            continue
        overlay = PatchCollection(
            [rectangles[index] for index in np.flatnonzero(mask)],
            facecolor="none",
            edgecolor=color,
            linewidth=width,
            hatch=hatch,
            zorder=4,
        )
        axis.add_collection(overlay)

    _draw_outline(axis, geometry)
    axis.set_xlim(limits[0], limits[1])
    axis.set_ylim(limits[2], limits[3])
    axis.set_aspect("equal")
    axis.set_xticks([])
    axis.set_yticks([])
    return collection


def plot_cohort_overview(
    field_evidence: pd.DataFrame,
    *,
    windows: Sequence[str] = DEFAULT_WINDOWS,
) -> Figure:
    """Build a four-panel overview from one selected pair per unique field.

    The panels expose cohort composition, evidence above the empirical pure-crop
    threshold, longitudinal parent balance, and call-status denominators.  The
    function returns an unsaved Matplotlib figure.
    """

    window_names = _validate_windows(windows)
    rows = _display_field_rows(field_evidence)
    rows = rows[rows["window_id"].astype(str).isin(window_names)].copy()
    if rows.empty:
        raise ValueError("field_evidence has no rows in the requested windows")
    order = {window: index for index, window in enumerate(window_names)}
    rows["_window_order"] = rows["window_id"].astype(str).map(order)

    figure, axes = plt.subplots(2, 2, figsize=(15, 10))

    field_table = rows[["field_uid", "landcover"]].drop_duplicates()
    field_counts = field_table["landcover"].value_counts().sort_values()
    axes[0, 0].barh(
        field_counts.index,
        field_counts.to_numpy(),
        color="#4C78A8",
    )
    for position, count in enumerate(field_counts):
        axes[0, 0].text(count, position, f" {count}", va="center", fontsize=9)
    axes[0, 0].set_title("Unique physical fields in this analysis")
    axes[0, 0].set_xlabel("fields")

    gap_values = []
    for window in window_names:
        selected = rows[rows["window_id"].astype(str).eq(window)]
        gap_values.append(
            (
                selected["log_evidence_over_pure"]
                - selected["evidence_threshold"]
            ).dropna().to_numpy(np.float64)
        )
    if any(len(values) for values in gap_values):
        boxes = axes[0, 1].boxplot(
            gap_values,
            patch_artist=True,
            showfliers=True,
        )
        axes[0, 1].set_xticks(range(1, len(window_names) + 1), window_names)
        for box in boxes["boxes"]:
            box.set_facecolor("#72B7B2")
            box.set_alpha(0.75)
    axes[0, 1].axhline(0, color="crimson", linestyle="--", linewidth=1)
    axes[0, 1].set_title("Pair evidence relative to the pure-crop threshold")
    axes[0, 1].set_ylabel("log evidence − threshold")

    mixture_labels = [
        label
        for label in sorted(rows["landcover"].astype(str).unique())
        if " and " in label.lower()
    ]
    for index, label in enumerate(mixture_labels):
        selected = rows[rows["landcover"].astype(str).eq(label)]
        summary = (
            selected.groupby("_window_order")["mosaic_parent_a_share"]
            .agg(
                median="median",
                low=lambda values: values.quantile(0.25),
                high=lambda values: values.quantile(0.75),
            )
            .reindex(range(len(window_names)))
        )
        x = np.arange(len(window_names))
        color = STATUS_COLORS[index % len(STATUS_COLORS)]
        axes[1, 0].fill_between(x, summary["low"], summary["high"], color=color, alpha=0.16)
        axes[1, 0].plot(x, summary["median"], marker="o", color=color, label=label)
    axes[1, 0].set_xticks(range(len(window_names)), window_names)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].set_ylabel("mosaic parent-A share")
    axes[1, 0].set_title("Intercrop balance through cumulative windows")
    if mixture_labels:
        axes[1, 0].legend(fontsize=8)

    status = pd.crosstab(rows["window_id"].astype(str), rows["call_status"].astype(str))
    status = status.reindex(window_names, fill_value=0)
    bottom = np.zeros(len(status), dtype=np.float64)
    for index, column in enumerate(status.columns):
        values = status[column].to_numpy(np.float64)
        axes[1, 1].bar(
            np.arange(len(status)),
            values,
            bottom=bottom,
            color=STATUS_COLORS[index % len(STATUS_COLORS)],
            label=str(column),
        )
        bottom += values
    axes[1, 1].set_xticks(range(len(window_names)), window_names)
    axes[1, 1].set_ylabel("unique fields")
    axes[1, 1].set_title("Every field remains in the status denominator")
    axes[1, 1].legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")

    for axis in axes.ravel():
        axis.grid(alpha=0.15, axis="y")
    figure.suptitle(
        f"Intercropping DNA cohort overview · {field_table['field_uid'].nunique():,} fields",
        fontsize=15,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    return figure


def plot_field_dashboard(
    field_uid: str,
    pixel_evidence: pd.DataFrame,
    field_evidence: pd.DataFrame,
    fields: pd.DataFrame,
    *,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    pair_key: str | None = None,
) -> Figure:
    """Render one physical field across all four windows.

    Pixel rows use the exact 10 m grid convention documented at module level;
    the WKT in ``fields`` is assumed to be EPSG:4326 and is projected to the
    single ``utm_epsg`` carried by those pixels.  Colors show conditional
    parent-A probability, while hatching exposes low named-parent mass, shared
    pixels, and label conflicts rather than hiding them.
    """

    _require_columns(pixel_evidence, PIXEL_BASE_COLUMNS, "pixel_evidence")
    _require_columns(field_evidence, FIELD_COLUMNS, "field_evidence")
    window_names = _validate_windows(windows)
    if len(window_names) != 4:
        raise ValueError("a field dashboard requires exactly four ordered windows")
    field_rows, selected_pair = _field_ordered_rows(
        str(field_uid), field_evidence, window_names, pair_key
    )
    pair_columns = _pair_pixel_columns(selected_pair)
    _require_columns(pixel_evidence, set(pair_columns.values()), "pixel_evidence")

    pixels = pixel_evidence[
        pixel_evidence["field_uid"].astype(str).eq(str(field_uid))
        & pixel_evidence["window_id"].astype(str).isin(window_names)
    ].copy()
    if pixels.empty:
        raise ValueError(f"pixel_evidence contains no field {field_uid!r}")
    for window in window_names:
        if not pixels["window_id"].astype(str).eq(window).any():
            raise ValueError(f"pixel_evidence for {field_uid!r} is missing {window}")
    if pixels.duplicated(["pixel_id", "window_id"]).any():
        raise ValueError(f"pixel_evidence for {field_uid!r} has duplicate pixel/window rows")
    epsg = int(_single_value(pixels["utm_epsg"], "utm_epsg"))
    geometry = _project_field_geometry(fields, str(field_uid), epsg)
    limits = _map_limits(geometry, pixels)
    parent_a = str(_single_value(field_rows["parent_a"], "parent_a"))
    parent_b = str(_single_value(field_rows["parent_b"], "parent_b"))

    figure = plt.figure(figsize=(18, 11))
    grid = figure.add_gridspec(3, 4, height_ratios=(1.6, 1.0, 0.9))
    map_axes = [figure.add_subplot(grid[0, column]) for column in range(4)]
    collection = None
    field_by_window = field_rows.set_index(field_rows["window_id"].astype(str))
    for axis, window in zip(map_axes, window_names, strict=True):
        window_pixels = pixels[pixels["window_id"].astype(str).eq(window)].copy()
        evidence_row = field_by_window.loc[window]
        window_pixels["_pixel_out_of_model"] = window_pixels[
            "minimum_squared_mahalanobis"
        ].gt(float(evidence_row["pixel_adequacy_distance_threshold"]))
        collection = _draw_pixel_map(
            axis,
            window_pixels,
            geometry,
            pair_columns["conditional_parent_a_probability"],
            pair_columns["named_parent_mass"],
            limits,
        )
        axis.set_title(
            f"{window}{' · OOD' if window == 'w4' else ''} · "
            f"{len(window_pixels):,} exact pixels\n"
            f"q={evidence_row['mosaic_parent_a_share']:.2f}, "
            f"evidence={evidence_row['log_evidence_over_pure']:.2f}"
        )
    if collection is not None:
        colorbar = figure.colorbar(
            collection,
            ax=map_axes,
            orientation="horizontal",
            fraction=0.035,
            pad=0.05,
        )
        colorbar.set_label(f"conditional probability: {parent_b} (0) ← pixel → {parent_a} (1)")
    map_axes[-1].legend(
        handles=[
            Patch(
                facecolor="none",
                edgecolor="#666666",
                hatch="..",
                label="named-parent mass < 0.5",
            ),
            Patch(
                facecolor="none",
                edgecolor="#7B2CBF",
                hatch="++",
                label="outside held-out reference distance",
            ),
            Patch(facecolor="none", edgecolor="#333333", hatch="///", label="shared pixel"),
            Patch(facecolor="none", edgecolor="crimson", hatch="xxx", label="label conflict"),
        ],
        fontsize=7,
        loc="lower left",
        bbox_to_anchor=(1.02, 0),
    )

    x = np.arange(len(window_names))
    q_axis = figure.add_subplot(grid[1, :2])
    q = field_rows["mosaic_parent_a_share"].to_numpy(np.float64)
    low = field_rows["mosaic_ci_low"].to_numpy(np.float64)
    high = field_rows["mosaic_ci_high"].to_numpy(np.float64)
    q_axis.fill_between(x, low, high, color="#4C78A8", alpha=0.22, label="95% block bootstrap")
    q_axis.plot(x, q, marker="o", linewidth=2.0, color="#4C78A8", label="mosaic MLE")
    q_axis.plot(
        x,
        field_rows["field_axis_parent_a_share"],
        marker="s",
        linestyle="--",
        color="#F58518",
        label="mean covariance-aware axis share",
    )
    q_axis.set_xticks(x, window_names)
    q_axis.set_ylim(0, 1)
    q_axis.set_ylabel(f"{parent_a} share (0={parent_b}, 1={parent_a})")
    q_axis.set_title("Field-level parent balance and uncertainty")
    q_axis.legend(fontsize=8)
    q_axis.grid(alpha=0.15)

    evidence_axis = figure.add_subplot(grid[1, 2:])
    evidence = field_rows["log_evidence_over_pure"].to_numpy(np.float64)
    threshold = field_rows["evidence_threshold"].to_numpy(np.float64)
    evidence_axis.plot(x, evidence, marker="o", linewidth=2, label="field evidence")
    evidence_axis.plot(
        x,
        threshold,
        linestyle="--",
        linewidth=1.5,
        color="crimson",
        label="pure-crop threshold",
    )
    evidence_axis.fill_between(
        x,
        threshold,
        evidence,
        where=evidence >= threshold,
        color="#55A868",
        alpha=0.18,
    )
    evidence_axis.set_xticks(x, window_names)
    evidence_axis.set_ylabel("log evidence over best pure endpoint")
    evidence_axis.set_title("Named pair must exceed the empirical pure-crop null")
    evidence_axis.legend(fontsize=8)
    evidence_axis.grid(alpha=0.15)

    probability_axis = figure.add_subplot(grid[2, :3])
    bottom = np.zeros(len(window_names), dtype=np.float64)
    for crop, column in CROP_PROBABILITY_COLUMNS.items():
        values = field_rows[column].to_numpy(np.float64)
        probability_axis.bar(x, values, bottom=bottom, color=CROP_COLORS[crop], label=crop)
        bottom += values
    probability_axis.set_xticks(x, window_names)
    probability_axis.set_ylim(0, max(1.0, float(np.nanmax(bottom)) * 1.03))
    probability_axis.set_ylabel("mean calibrated crop probability")
    probability_axis.set_title(
        "All reference crops remain visible; off-pair signal is not discarded"
    )
    probability_axis.legend(ncol=4, fontsize=8, loc="upper center")
    probability_axis.grid(alpha=0.15, axis="y")

    status_axis = figure.add_subplot(grid[2, 3])
    status_axis.axis("off")
    status_lines = [
        f"field: {field_uid}",
        f"label: {_single_value(field_rows['landcover'], 'landcover')}",
        f"display pair: {parent_a} + {parent_b}",
        "",
    ]
    for row in field_rows.itertuples(index=False):
        status_lines.extend(
            [
                f"{row.window_id}: {row.call_status}",
                f"  null tail={row.evidence_tail_probability:.3g}, "
                f"pair margin={row.pair_log_margin:.2f}",
                f"  distance p90={row.minimum_mahalanobis_p90:.2f} / "
                f"limit={row.adequacy_distance_threshold:.2f}",
                f"  alternative={row.best_alternative_pair}",
            ]
        )
    status_axis.text(
        0,
        1,
        "\n".join(status_lines),
        va="top",
        ha="left",
        fontsize=8.5,
        family="monospace",
        linespacing=1.25,
    )
    status_axis.set_title("Evidence card", loc="left", fontweight="bold")

    figure.suptitle(
        f"Intercropping DNA field dashboard · {field_uid} · {parent_a} + {parent_b}",
        fontsize=15,
    )
    figure.subplots_adjust(top=0.92, bottom=0.06, left=0.05, right=0.94, hspace=0.48, wspace=0.28)
    return figure


def save_figure_atomic(
    figure: Figure,
    output_path: str | Path,
    *,
    dpi: int = 170,
) -> Path:
    """Atomically save a Matplotlib figure as a PNG in its destination directory."""

    path = Path(output_path)
    if path.suffix.lower() != ".png":
        raise ValueError("report figures must use a .png path")
    if not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".png.part",
        dir=path.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        figure.savefig(temporary, format="png", dpi=dpi, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def save_cohort_overview(
    field_evidence: pd.DataFrame,
    output_path: str | Path,
    *,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    dpi: int = 170,
) -> tuple[Path, Figure]:
    """Create and atomically save the cohort overview, returning path and figure."""

    figure = plot_cohort_overview(field_evidence, windows=windows)
    return save_figure_atomic(figure, output_path, dpi=dpi), figure


def save_field_dashboard(
    field_uid: str,
    pixel_evidence: pd.DataFrame,
    field_evidence: pd.DataFrame,
    fields: pd.DataFrame,
    output_path: str | Path,
    *,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    pair_key: str | None = None,
    dpi: int = 170,
) -> tuple[Path, Figure]:
    """Create and atomically save one field dashboard, returning path and figure."""

    figure = plot_field_dashboard(
        field_uid,
        pixel_evidence,
        field_evidence,
        fields,
        windows=windows,
        pair_key=pair_key,
    )
    return save_figure_atomic(figure, output_path, dpi=dpi), figure


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path, base: Path) -> dict[str, object]:
    return {
        "path": str(path.relative_to(base)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def write_plot_manifest_atomic(manifest: Mapping[str, object], path: str | Path) -> Path:
    """Atomically write the JSON manifest after every listed PNG exists."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(manifest), indent=2, sort_keys=True, default=str).encode()
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.",
        suffix=".json.part",
        dir=destination.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
        handle.close()
        os.replace(temporary, destination)
    finally:
        if not handle.closed:
            handle.close()
        temporary.unlink(missing_ok=True)
    return destination


def save_all_field_reports(
    pixel_evidence: pd.DataFrame,
    field_evidence: pd.DataFrame,
    fields: pd.DataFrame,
    output_dir: str | Path,
    *,
    windows: Sequence[str] = DEFAULT_WINDOWS,
    dpi: int = 170,
    metadata: Mapping[str, object] | None = None,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 100,
) -> dict[str, object]:
    """Save the overview and one four-window PNG per unique physical field.

    The function fails if any field cannot produce a complete dashboard.  The
    manifest is written last, so its presence certifies that every listed PNG
    was atomically published.  Returned paths are absolute strings; manifest
    entries use paths relative to ``output_dir`` for portability.
    """

    if progress_every < 1:
        raise ValueError("progress_every must be positive")
    report = (lambda message: None) if progress is None else progress
    _require_columns(pixel_evidence, PIXEL_BASE_COLUMNS, "pixel_evidence")
    _require_columns(field_evidence, FIELD_COLUMNS, "field_evidence")
    _require_columns(fields, {"field_uid", "wkt"}, "fields")
    window_names = _validate_windows(windows)
    root = Path(output_dir).expanduser().resolve()
    field_dir = root / "field_reports"
    field_dir.mkdir(parents=True, exist_ok=True)

    overview_path = root / "cohort_overview.png"
    overview_figure = plot_cohort_overview(field_evidence, windows=window_names)
    try:
        save_figure_atomic(overview_figure, overview_path, dpi=dpi)
    finally:
        plt.close(overview_figure)

    report_records = []
    field_paths: dict[str, str] = {}
    field_ids = sorted(field_evidence["field_uid"].astype(str).unique())
    pixel_groups = {
        str(field_uid): rows.copy()
        for field_uid, rows in pixel_evidence.groupby("field_uid", sort=False)
    }
    evidence_groups = {
        str(field_uid): rows.copy()
        for field_uid, rows in field_evidence.groupby("field_uid", sort=False)
    }
    field_groups = {
        str(field_uid): rows.copy()
        for field_uid, rows in fields.groupby("field_uid", sort=False)
    }
    report(f"Rendering {len(field_ids):,} unique physical-field dashboards")
    for position, field_uid in enumerate(field_ids, start=1):
        field_rows = evidence_groups[field_uid]
        selected_pair = choose_display_pair(field_rows)
        filename = f"{_safe_filename(field_uid)}.png"
        report_path = field_dir / filename
        figure = plot_field_dashboard(
            field_uid,
            pixel_groups[field_uid],
            field_rows,
            field_groups[field_uid],
            windows=window_names,
            pair_key=selected_pair,
        )
        try:
            save_figure_atomic(figure, report_path, dpi=dpi)
        finally:
            plt.close(figure)
        label = str(_single_value(field_rows["landcover"], "landcover"))
        record = {
            "field_uid": field_uid,
            "landcover": label,
            "pair_key": selected_pair,
            "windows": list(window_names),
            "pixel_rows": int(
                pixel_groups[field_uid]["window_id"].astype(str).isin(window_names).sum()
            ),
            **_file_record(report_path, root),
        }
        report_records.append(record)
        field_paths[field_uid] = str(report_path)
        if position % progress_every == 0 or position == len(field_ids):
            report(f"Rendered {position:,}/{len(field_ids):,} field dashboards")

    manifest = {
        "schema_version": "1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "windows": list(window_names),
        "unique_field_count": len(field_ids),
        "overview": _file_record(overview_path, root),
        "field_reports": report_records,
        "metadata": {} if metadata is None else dict(metadata),
    }
    manifest_path = write_plot_manifest_atomic(manifest, root / "plot_manifest.json")
    return {
        "overview_path": str(overview_path),
        "manifest_path": str(manifest_path),
        "field_paths": field_paths,
        "manifest": manifest,
    }


__all__ = [
    "CROP_PROBABILITY_COLUMNS",
    "DEFAULT_WINDOWS",
    "PAIR_PREFIXES",
    "choose_display_pair",
    "plot_cohort_overview",
    "plot_field_dashboard",
    "save_all_field_reports",
    "save_cohort_overview",
    "save_field_dashboard",
    "save_figure_atomic",
    "write_plot_manifest_atomic",
]
