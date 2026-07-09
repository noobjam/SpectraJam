from __future__ import annotations

import ast
from dataclasses import replace
from datetime import date
import json
from pathlib import Path

import numpy as np
import pandas as pd
import shapely
from pyproj import Transformer

from plain_tessera_incremental.inference import WindowEmbeddings
from plain_tessera_incremental.notebooks._dna_workflow import (
    default_config,
    finalize_workflow_export,
    run_workflow,
    save_field_plot_index,
    save_workflow_tables,
)
from plain_tessera_incremental.notebooks._dna_reporting import save_all_field_reports
from plain_tessera_incremental.storage import (
    canonical_sha256,
    write_dataframe_atomic,
    write_embedding_shard,
    write_json_atomic,
)
from plain_tessera_incremental.windows import PrefixWindow


def test_presentation_notebooks_are_standalone() -> None:
    notebook_dir = Path(__file__).parents[1] / "notebooks"
    notebook_names = (
        "intercropping_pdf_evidence_pack.ipynb",
        "intercropping_temporal_separability.ipynb",
        "intercropping_parent_evidence_v2.ipynb",
        "intercropping_harvard_multilens.ipynb",
        "intercropping_harvard_only_workbench.ipynb",
        "intercropping_harvard_one_click.ipynb",
    )
    for notebook_name in notebook_names:
        notebook = json.loads((notebook_dir / notebook_name).read_text())
        code = "\n".join(
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        tree = ast.parse(code)
        project_imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                project_imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("plain_tessera_incremental")
                )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("plain_tessera_incremental"):
                    project_imports.append(module)
        assert project_imports == []
        assert sum(len(cell.get("outputs", [])) for cell in notebook["cells"]) == 0

    temporal_source = (
        notebook_dir / "intercropping_temporal_separability.ipynb"
    ).read_text()
    for figure_number in range(1, 8):
        assert f"0{figure_number}_" in temporal_source
    assert "PDF_HANDOFF_BEGIN" in temporal_source
    assert "PDF_HANDOFF_END" in temporal_source
    assert "PDF_HANDOFF.json" in temporal_source

    parent_evidence_source = (
        notebook_dir / "intercropping_parent_evidence_v2.ipynb"
    ).read_text()
    for figure_number in range(1, 8):
        assert f"0{figure_number}_" in parent_evidence_source
    assert "PDF_HANDOFF_V2_BEGIN" in parent_evidence_source
    assert "PDF_HANDOFF_V2_END" in parent_evidence_source
    assert "mixture_fields_in_parent_fit" in parent_evidence_source
    assert "TARGET_TRAIN_FPR" in parent_evidence_source
    assert "hybrid_parent_evidence" in parent_evidence_source

    harvard_multilens_source = (
        notebook_dir / "intercropping_harvard_multilens.ipynb"
    ).read_text()
    for figure_number in range(1, 8):
        assert f"0{figure_number}_" in harvard_multilens_source
    assert "HARVARD_MULTILENS_HANDOFF_BEGIN" in harvard_multilens_source
    assert "HARVARD_MULTILENS_HANDOFF_END" in harvard_multilens_source
    assert 'SAFE_END = pd.Timestamp(\\"2025-05-06\\")' in harvard_multilens_source
    assert "StratifiedGroupKFold" in harvard_multilens_source
    assert "not_spatially_testable" in harvard_multilens_source
    assert "raw_full_retrospective" in harvard_multilens_source
    assert "No result estimates crop fraction" in harvard_multilens_source

    harvard_workbench_source = (
        notebook_dir / "intercropping_harvard_only_workbench.ipynb"
    ).read_text()
    for figure_number in range(1, 8):
        assert f"0{figure_number}_" in harvard_workbench_source
    assert "HARVARD_WORKBENCH_HANDOFF_BEGIN" in harvard_workbench_source
    assert "HARVARD_WORKBENCH_HANDOFF_END" in harvard_workbench_source
    assert "data_bundle = load_harvard_data()" in harvard_workbench_source
    assert (
        "feature_bundle = build_harvard_features(data_bundle)"
        in harvard_workbench_source
    )
    assert "results = run_harvard_evaluation(feature_bundle)" in harvard_workbench_source
    assert "ANALYSIS_DIR" not in harvard_workbench_source
    assert "load_handoff_exports" not in harvard_workbench_source

    one_click_notebook = json.loads(
        (notebook_dir / "intercropping_harvard_one_click.ipynb").read_text()
    )
    one_click_cells = [
        cell for cell in one_click_notebook["cells"] if cell["cell_type"] == "code"
    ]
    assert len(one_click_cells) == 1
    one_click_source = "".join(one_click_cells[0]["source"])
    assert "data_bundle = load_harvard_data()" in one_click_source
    assert "handoff = finalize_harvard_workbench(" in one_click_source
    assert 'root / "COMPLETED.json"' not in one_click_source
    assert "ANALYSIS_DIR" not in one_click_source
    assert "HARVARD_WORKBENCH_HANDOFF_BEGIN" in one_click_source
    assert "HARVARD_WORKBENCH_HANDOFF_END" in one_click_source


def test_harvard_multilens_handoff_recovers_after_kernel_restart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "intercropping_harvard_multilens.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    output = tmp_path / "pipeline"
    output.mkdir()
    fingerprint = "a" * 64
    write_json_atomic(output / "run.json", {"run_fingerprint": fingerprint})
    write_json_atomic(output / "COMPLETED.json", {"status": "complete"})
    for filename in ("fields.parquet", "pixels.parquet", "field_pixels.parquet"):
        pd.DataFrame({"placeholder": [1]}).to_parquet(output / filename, index=False)

    analysis_root = tmp_path / "analysis"
    analysis_dir = analysis_root / fingerprint[:16]
    figure_dir = analysis_dir / "figures"
    figure_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "field_uid": ["field-1"],
            "spatial_block": ["block-1"],
            "landcover": ["Maize"],
        }
    ).to_parquet(analysis_dir / "field_features.parquet", index=False)
    pd.DataFrame(
        {"field_uid": ["field-1"], "pixel_id": ["pixel-1"]}
    ).to_parquet(analysis_dir / "field_pixel_index.parquet", index=False)
    pd.DataFrame(
        {
            "field_uid": ["field-1"],
            "task": ["generic_intercrop"],
            "lens": ["raw_safe"],
        }
    ).to_parquet(analysis_dir / "field_predictions.parquet", index=False)
    pd.DataFrame(
        {
            "task": ["generic_intercrop"],
            "lens": ["raw_safe"],
            "n": [1],
            "positive_n": [0],
            "auroc": [np.nan],
            "auroc_ci_low": [np.nan],
            "auroc_ci_high": [np.nan],
            "average_precision": [np.nan],
            "ap_lift": [np.nan],
            "recall": [np.nan],
            "false_positive_rate": [np.nan],
        }
    ).to_parquet(analysis_dir / "performance.parquet", index=False)
    pd.DataFrame({"task": ["generic_intercrop"], "delta": [0.0]}).to_parquet(
        analysis_dir / "paired_lens_deltas.parquet", index=False
    )
    pd.DataFrame({"hypothesis": ["fixture"], "flag": [False]}).to_parquet(
        analysis_dir / "hypothesis_scorecard.parquet", index=False
    )
    pd.DataFrame({"criterion": ["fixture"], "passed": [False]}).to_parquet(
        analysis_dir / "bean_maize_exploratory_gates.parquet", index=False
    )
    for filename in (
        "cohort_attrition.parquet",
        "cache_task_audit.parquet",
        "spatial_fold_assignments.parquet",
        "evaluation_audit.parquet",
        "resolution_sensitivity.parquet",
    ):
        pd.DataFrame({"placeholder": [1]}).to_parquet(
            analysis_dir / filename, index=False
        )
    for figure_number in range(1, 8):
        (figure_dir / f"0{figure_number}_fixture.png").write_bytes(b"fixture")

    monkeypatch.setenv("TESSERA_OUTPUT_DIR", str(output))
    monkeypatch.setenv("HARVARD_MULTILENS_EXPORT_DIR", str(analysis_root))
    namespace: dict[str, object] = {}
    setup_source = "".join(notebook["cells"][1]["source"])
    handoff_source = "".join(notebook["cells"][23]["source"])
    exec(compile(setup_source, str(notebook_path), "exec"), namespace)
    assert namespace["ANALYSIS_DIR"] == analysis_dir
    exec(compile(handoff_source, str(notebook_path), "exec"), namespace)

    completion = json.loads((analysis_dir / "COMPLETED.json").read_text())
    assert completion["status"] == "complete"
    assert completion["figure_count"] == 7
    assert (analysis_dir / "HARVARD_MULTILENS_HANDOFF.json").is_file()


def test_harvard_workbench_hidden_setup_defines_complete_api() -> None:
    notebook_path = (
        Path(__file__).parents[1]
        / "notebooks"
        / "intercropping_harvard_only_workbench.ipynb"
    )
    notebook = json.loads(notebook_path.read_text())
    setup_source = "".join(notebook["cells"][1]["source"])
    namespace: dict[str, object] = {}
    exec(compile(setup_source, str(notebook_path), "exec"), namespace)

    required_api = {
        "load_harvard_data",
        "build_harvard_features",
        "run_harvard_evaluation",
        "figure_cohort_support",
        "figure_geography",
        "figure_multilens_performance",
        "figure_crop_presence",
        "figure_resolution",
        "figure_phenology",
        "figure_falsification",
        "finalize_harvard_workbench",
    }
    assert required_api <= namespace.keys()


def _fixture(root: Path) -> tuple[Path, int]:
    output = root / "pipeline"
    fingerprint = "f" * 64
    crop_fields = {
        "Maize": ("maize-0", "maize-1"),
        "Bean": ("bean-0", "bean-1", "conflict-bean"),
        "Irish Potato": ("potato-0", "potato-1"),
        "Rice": ("rice-0", "rice-1"),
        "Bean and Maize": ("bean-maize", "outlier-mix"),
        "Irish Potato and Maize": ("potato-maize",),
    }
    field_rows = []
    membership_rows = []
    pixel_rows = []
    field_crop = {}
    x_index = 1000
    to_wgs84 = Transformer.from_crs(32735, 4326, always_xy=True)
    for crop, field_ids in crop_fields.items():
        for field_id in field_ids:
            field_crop[field_id] = crop
            geometry_hash = f"geometry-{field_id}"
            pixel_ids = []
            for local in range(4):
                pixel_id = f"utm-32735-10m-{x_index + local}-2000"
                pixel_ids.append(pixel_id)
                longitude, latitude = to_wgs84.transform(
                    (x_index + local + 0.5) * 10.0,
                    (2000 + 0.5) * 10.0,
                )
                pixel_rows.append(
                    {
                        "pixel_id": pixel_id,
                        "utm_epsg": 32735,
                        "pixel_x_index": x_index + local,
                        "pixel_y_index": 2000,
                        "pixel_longitude": longitude,
                        "pixel_latitude": latitude,
                    }
                )
                membership_rows.append(
                    {
                        "field_uid": field_id,
                        "source_id": field_id,
                        "landcover": crop,
                        "quadkey": "q",
                        "pixel_id": pixel_id,
                        "utm_epsg": 32735,
                        "pixel_x_index": x_index + local,
                        "pixel_y_index": 2000,
                        "pixel_longitude": longitude,
                        "pixel_latitude": latitude,
                        "work_x_index": 0,
                        "work_y_index": 0,
                        "chunk_x_index": 0,
                        "chunk_y_index": 0,
                        "overlap_field_count": 1,
                        "label_conflict": False,
                    }
                )
            field_rows.append(
                {
                    "field_uid": field_id,
                    "id": field_id,
                    "landcover": crop,
                    "wkt": shapely.transform(
                        shapely.box(
                            x_index * 10.0,
                            2000 * 10.0,
                            (x_index + 4) * 10.0,
                            (2000 + 1) * 10.0,
                        ),
                        to_wgs84.transform,
                        interleaved=False,
                    ).wkt,
                    "utm_epsg": 32735,
                    "pixel_count": len(pixel_ids),
                    "duplicate_count": 1,
                    "geometry_sha256": geometry_hash,
                }
            )
            x_index += 10

    # One same-label source replica shares the exact physical geometry/pixels.
    original = next(row for row in field_rows if row["field_uid"] == "bean-0")
    replica = dict(original)
    replica["field_uid"] = "bean-0-replica"
    replica["id"] = "bean-0-replica"
    replica["duplicate_count"] = 2
    original["duplicate_count"] = 2
    field_rows.append(replica)
    for row in [row for row in membership_rows if row["field_uid"] == "bean-0"]:
        copy = dict(row)
        copy["field_uid"] = "bean-0-replica"
        copy["source_id"] = "bean-0-replica"
        copy["overlap_field_count"] = 2
        row["overlap_field_count"] = 2
        membership_rows.append(copy)
    field_crop["bean-0-replica"] = "Bean"

    # One exact geometry carries conflicting labels and must remain a no-call.
    conflict_source = next(
        row for row in field_rows if row["field_uid"] == "conflict-bean"
    )
    conflict_replica = dict(conflict_source)
    conflict_replica["field_uid"] = "conflict-maize"
    conflict_replica["id"] = "conflict-maize"
    conflict_replica["landcover"] = "Maize"
    field_rows.append(conflict_replica)
    for row in [
        row for row in membership_rows if row["field_uid"] == "conflict-bean"
    ]:
        copy = dict(row)
        copy["field_uid"] = "conflict-maize"
        copy["source_id"] = "conflict-maize"
        copy["landcover"] = "Maize"
        copy["overlap_field_count"] = 2
        copy["label_conflict"] = True
        row["overlap_field_count"] = 2
        row["label_conflict"] = True
        membership_rows.append(copy)
    field_crop["conflict-maize"] = "Maize"

    fields = pd.DataFrame(field_rows)
    pixels = pd.DataFrame(pixel_rows).drop_duplicates("pixel_id").reset_index(drop=True)
    memberships = pd.DataFrame(membership_rows)
    write_dataframe_atomic(fields, output / "fields.parquet", fingerprint, "fields")
    write_dataframe_atomic(pixels, output / "pixels.parquet", fingerprint, "pixels")
    write_dataframe_atomic(
        memberships,
        output / "field_pixels.parquet",
        fingerprint,
        "field_pixels",
    )

    windows = (
        PrefixWindow("w1", 1, date(2024, 9, 1), date(2025, 1, 1)),
        PrefixWindow("w2", 2, date(2024, 9, 1), date(2025, 5, 1)),
        PrefixWindow("w3", 3, date(2024, 9, 1), date(2025, 9, 1)),
        PrefixWindow("w4", 4, date(2024, 9, 1), date(2026, 1, 1)),
    )
    write_json_atomic(
        output / "run.json",
        {
            "run_fingerprint": fingerprint,
            "config": {
                "windows": [
                    {
                        "window_id": window.window_id,
                        "ordinal": window.ordinal,
                        "start": window.start.isoformat(),
                        "end_exclusive": window.end_exclusive.isoformat(),
                    }
                    for window in windows
                ]
            },
        },
    )

    pixel_position = {pixel_id: index for index, pixel_id in enumerate(pixels["pixel_id"])}
    shard_memberships = memberships.copy()
    shard_memberships["pixel_position"] = shard_memberships["pixel_id"].map(pixel_position)
    task_key = canonical_sha256(
        {"epsg": 32735, "work_x": 0, "work_y": 0, "chunk_x": 0, "chunk_y": 0}
    )[:24]
    centers = {
        "Maize": 0,
        "Bean": 1,
        "Irish Potato": 2,
        "Rice": 3,
    }
    rng = np.random.default_rng(19)
    base_vectors = np.empty((len(pixels), 128), dtype=np.float32)
    pixel_crops = {}
    for row in memberships.itertuples(index=False):
        if row.pixel_id in pixel_crops:
            continue
        crop = str(row.landcover)
        if crop == "Bean and Maize":
            crop = "Bean" if row.pixel_x_index % 2 == 0 else "Maize"
        elif crop == "Irish Potato and Maize":
            crop = "Irish Potato" if row.pixel_x_index % 2 == 0 else "Maize"
        pixel_crops[row.pixel_id] = crop
    for pixel_id, position in pixel_position.items():
        vector = rng.normal(scale=0.04, size=128)
        source_fields = set(
            memberships.loc[memberships["pixel_id"].eq(pixel_id), "field_uid"]
        )
        if "outlier-mix" in source_fields:
            vector[127] += 2.0
        else:
            vector[centers[pixel_crops[pixel_id]]] += 2.0
        base_vectors[position] = vector

    missing_longitudinal_pixel = memberships.loc[
        memberships["field_uid"].eq("bean-maize"), "pixel_id"
    ].iloc[0]
    missing_position = pixel_position[missing_longitudinal_pixel]

    for window in windows:
        embeddings = base_vectors + rng.normal(
            scale=0.005 * window.ordinal, size=base_vectors.shape
        ).astype(np.float32)
        outcome = np.array(["complete"] * len(pixels), dtype=object)
        if window.window_id == "w1":
            outcome[missing_position] = "empty_window"
        results = WindowEmbeddings(
            embeddings=embeddings,
            outcome=outcome,
            s2_valid_count=np.full(len(pixels), window.ordinal, dtype=np.int32),
            s1_valid_count=np.full(len(pixels), window.ordinal, dtype=np.int32),
            s2_input_count=np.full(len(pixels), 8, dtype=np.int32),
            s1_input_count=np.full(len(pixels), 8, dtype=np.int32),
            s2_source_count=window.ordinal,
            s1_source_count=window.ordinal,
        )
        write_embedding_shard(
            shard_memberships,
            results,
            window,
            output / "embeddings" / f"window_id={window.window_id}" / f"{task_key}.parquet",
            fingerprint,
            task_key,
            "task-fingerprint",
        )
    return output, len(fields)


def test_all_field_all_window_workflow_and_exports(tmp_path: Path) -> None:
    output, source_fields = _fixture(tmp_path)
    config = replace(
        default_config(output, tmp_path / "export"),
        shrinkage_candidates=(0.5,),
        max_folds=2,
        bootstrap_replicates=4,
        mixture_grid_size=41,
        min_reference_fields_per_crop=2,
        min_balanced_accuracy=0.0,
        max_endpoint_mae=1.0,
        max_synthetic_mae=1.0,
    )

    result = run_workflow(config)
    root, manifest = save_workflow_tables(result)
    report_output = save_all_field_reports(
        result.pixel_scores,
        result.physical_field_pair_scores,
        result.analysis_units,
        root / "figures",
        windows=config.windows,
        dpi=40,
    )
    plot_index = save_field_plot_index(
        result,
        root,
        report_output["field_paths"],
    )
    finalize_workflow_export(
        root,
        manifest,
        field_plot_count=len(report_output["field_paths"]),
        plot_manifest_path="figures/plot_manifest.json",
    )

    assert len(result.source_field_scores) == source_fields * 4
    assert not result.source_field_scores.duplicated(["field_uid", "window_id"]).any()
    assert set(result.pixel_scores["window_id"]) == {"w1", "w2", "w3", "w4"}
    assert set(result.physical_field_pair_scores["pair_key"]) == {
        "bean_maize",
        "potato_maize",
    }
    longitudinal_counts = (
        result.pixel_scores[result.pixel_scores["field_uid"].eq("bean-maize")]
        .groupby("window_id")["pixel_id"]
        .nunique()
    )
    assert longitudinal_counts.to_dict() == {"w1": 3, "w2": 3, "w3": 3, "w4": 3}
    replica = result.source_field_scores[
        result.source_field_scores["field_uid"].eq("bean-0-replica")
    ]
    assert replica["is_replica_inherited_result"].all()
    conflicts = result.source_field_scores[
        result.source_field_scores["field_uid"].isin(
            ["conflict-bean", "conflict-maize"]
        )
    ]
    assert conflicts["call_status"].eq(
        "NO_CALL_CONFLICTING_LABEL_GEOMETRY"
    ).all()
    assert not result.physical_field_pair_scores["field_uid"].isin(
        ["conflict-bean", "conflict-maize"]
    ).any()
    outlier = result.physical_field_pair_scores[
        result.physical_field_pair_scores["field_uid"].eq("outlier-mix")
        & result.physical_field_pair_scores["pair_key"].eq("bean_maize")
    ]
    assert outlier["call_status"].eq("OUT_OF_MODEL").all()
    assert manifest["source_field_window_rows"] == source_fields * 4
    assert len(plot_index) == source_fields
    assert not plot_index.loc[
        plot_index["field_uid"].isin(["conflict-bean", "conflict-maize"]),
        "report_available",
    ].any()
    assert plot_index.loc[
        plot_index["field_uid"].isin(["conflict-bean", "conflict-maize"]),
        "report_status",
    ].eq("NO_CALL_CONFLICTING_LABEL_GEOMETRY").all()
    assert (root / "tables" / "source_field_scores.parquet").is_file()
    assert (root / "tables" / "pixel_scores" / "window_id=w4" / "part-00000.parquet").is_file()
    assert (root / "figures" / "cohort_overview.png").is_file()
    assert (root / "COMPLETED.json").is_file()

    rerun_root, _ = save_workflow_tables(result)
    assert rerun_root == root
    assert not (root / "COMPLETED.json").exists()

    smoke_root = tmp_path / "smoke-export"
    finalize_workflow_export(
        smoke_root,
        manifest,
        field_plot_count=2,
        plot_manifest_path="figures/plot_manifest.json",
        gallery_complete=False,
        report_limit=2,
    )
    assert (smoke_root / "SMOKE_COMPLETE.json").is_file()
    assert not (smoke_root / "COMPLETED.json").exists()
