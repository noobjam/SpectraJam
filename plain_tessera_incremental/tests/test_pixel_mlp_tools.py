from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from plain_tessera_incremental.tools.build_pixel_classification_dataset import (
    build,
    choose_spatial_split,
    collapse_unambiguous_pixels,
)
from plain_tessera_incremental.tools.download_rwanda_worldcover import (
    RWANDA_SOURCE_KEYS,
)
from plain_tessera_incremental.tools.prepare_harvard_large_field_input import (
    prepare as prepare_large_fields,
)
from plain_tessera_incremental.tools.prepare_worldcover_noncrop_input import (
    DEFAULT_NONCROP_CODES,
    _keep_smallest,
    _patch_geometry,
)
from plain_tessera_incremental.tools.train_pixel_mlp import classification_metrics
from spectrajam.frame_sources import FRAME_SOURCES


def test_rwanda_download_uses_only_pinned_boundary_and_cover_tiles() -> None:
    assert RWANDA_SOURCE_KEYS == (
        "world_bank_admin0",
        "worldcover_2021_s03e027",
        "worldcover_2021_s03e030",
    )
    assert set(RWANDA_SOURCE_KEYS) <= set(FRAME_SOURCES)
    assert DEFAULT_NONCROP_CODES == (10, 20, 30, 50, 60, 80, 90)


def test_deterministic_sampler_retains_smallest_hash_ranks() -> None:
    heaps = {}
    for index in range(20):
        _keep_smallest(
            heaps,
            class_code=10,
            record={"pixel_id": f"pixel-{index}"},
            limit=5,
            seed=17,
        )

    selected = {entry[1] for entry in heaps[10]}
    expected = {
        pixel_id
        for _, pixel_id in sorted(
            (
                (
                    int.from_bytes(
                        hashlib.sha256(f"17:pixel-{index}".encode()).digest()[:8],
                        "big",
                    ),
                    f"pixel-{index}",
                )
                for index in range(20)
            )
        )[:5]
    }
    assert selected == expected


def test_patch_geometry_contains_exactly_the_requested_10m_cells() -> None:
    geometry = _patch_geometry(
        {"pixel_x_index": 100, "pixel_y_index": 200},
        width_m=100,
        shapely=shapely,
    )
    assert geometry.bounds == (950.0, 1950.0, 1050.0, 2050.0)


def test_large_field_selector_keeps_largest_fields_per_class_deterministically() -> None:
    rows = []
    for label, counts in {
        "Bean": [500, 400, 100],
        "Maize": [800, 300],
    }.items():
        for ordinal, count in enumerate(counts):
            rows.append(
                {
                    "field_uid": f"{label}-{ordinal}",
                    "geometry_status": (
                        "repaired" if label == "Bean" and ordinal == 1 else "valid"
                    ),
                    "center_pixel_count": count,
                    "LONGITUDE": 30.0 + ordinal / 100,
                    "LATITUDE": -2.0,
                    "QUADKEY": "q",
                    "landcover": label,
                    "wkt": shapely.box(
                        30.0 + ordinal / 100,
                        -2.0,
                        30.001 + ordinal / 100,
                        -1.999,
                    ).wkt,
                    "id": f"{label}-{ordinal}",
                }
            )
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        fields = root / "fields.parquet"
        output = root / "large.parquet"
        pd.DataFrame(rows).to_parquet(fields, index=False)
        result = prepare_large_fields(
            argparse.Namespace(
                fields=str(fields),
                min_pixels=256,
                max_fields_per_class=2,
                crop_labels=["Bean", "Maize"],
                output=str(output),
                manifest=None,
            )
        )
        selected = pd.read_parquet(output)

    assert selected["id"].tolist() == ["Bean-0", "Bean-1", "Maize-0", "Maize-1"]
    assert result["selected_field_counts"] == {"Bean": 2, "Maize": 2}
    assert result["selected_pixel_count_estimates"] == {"Bean": 900, "Maize": 1100}


def test_collapse_removes_conflicting_and_invalid_pixel_rows() -> None:
    vector = np.arange(128, dtype=np.float32)
    rows = pd.DataFrame(
        {
            "pixel_id": ["same", "same", "conflict", "conflict", "invalid"],
            "landcover": ["Maize", "Maize", "Maize", "Bean", "Maize"],
            "outcome": ["complete"] * 5,
            "embedding": [vector, vector, vector, vector, np.zeros(3, dtype=np.float32)],
        }
    )

    clean, audit = collapse_unambiguous_pixels(
        rows,
        allowed_labels={"Bean", "Maize"},
        source="harvard",
        np=np,
    )

    assert clean[["pixel_id", "label", "source"]].to_dict(orient="records") == [
        {"pixel_id": "same", "label": "Maize", "source": "harvard"}
    ]
    assert audit == {
        "invalid_embedding_rows": 1,
        "ambiguous_pixels": 1,
        "duplicate_membership_rows": 1,
        "retained_pixels": 1,
    }


def test_spatial_split_keeps_blocks_whole_and_every_class_in_each_split() -> None:
    rows = []
    for block_index in range(30):
        for label in ("Bean", "Non-crop: Tree cover"):
            rows.append(
                {
                    "pixel_id": f"{block_index}-{label}",
                    "label": label,
                    "spatial_block": f"block-{block_index}",
                }
            )
    frame = pd.DataFrame(rows)

    split, selected_seed = choose_spatial_split(
        frame,
        ratios=(0.6, 0.2, 0.2),
        seed=31,
        attempts=256,
    )

    assert 31 <= selected_seed < 31 + 256
    assert split.groupby("spatial_block")["split"].nunique().eq(1).all()
    counts = split.groupby(["label", "split"]).size().unstack(fill_value=0)
    assert (counts[["train", "validation", "test"]] > 0).all().all()


def test_classification_metrics_reports_macro_and_per_class_values() -> None:
    metrics, matrix = classification_metrics(
        np.asarray([0, 0, 1, 1]),
        np.asarray([0, 1, 1, 1]),
        ["crop", "noncrop"],
        np,
    )

    assert matrix.tolist() == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75
    assert metrics["balanced_accuracy"] == 0.75
    assert [row["support"] for row in metrics["per_class"]] == [2, 2]


def _write_embedding_fixture(root: Path, fingerprint: str, label: str, offset: int) -> None:
    run = {
        "schema_version": "fixture",
        "preprocessing_version": "fixture",
        "checkpoint_sha256": "a" * 64,
        "run_fingerprint": fingerprint,
        "config": {
            "windows": [
                {
                    "window_id": "w2",
                    "ordinal": 2,
                    "start": "2024-09-01",
                    "end_exclusive": "2025-05-01",
                }
            ],
            "pixel_size_m": 10,
            "work_tile_m": 20_000,
            "stac_endpoint": "fixture",
            "s2_collection": "s2",
            "s1_collection": "s1",
        },
    }
    root.mkdir(parents=True)
    (root / "run.json").write_text(json.dumps(run))
    (root / "COMPLETED.json").write_text(
        json.dumps({"completed": True, "run_fingerprint": fingerprint})
    )
    rows = []
    for block in range(30):
        x_index = block * 1_000 + offset
        rows.append(
            {
                "run_fingerprint": fingerprint,
                "field_uid": f"field-{block}-{offset}",
                "pixel_id": f"utm-32735-10m-{x_index}-480000",
                "utm_epsg": 32735,
                "pixel_x_index": x_index,
                "pixel_y_index": 480000,
                "pixel_longitude": 30.0,
                "pixel_latitude": -2.0,
                "window_id": "w2",
                "outcome": "complete",
                "landcover": label,
                "embedding": np.full(128, block + offset, dtype=np.float32),
            }
        )
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    vectors = pa.array(
        [row["embedding"].tolist() for row in rows], type=pa.list_(pa.float32())
    )
    table = table.set_column(table.schema.get_field_index("embedding"), "embedding", vectors)
    table = table.replace_schema_metadata(
        {
            b"run_fingerprint": fingerprint.encode(),
            b"artifact": b"pixel_embeddings",
            b"window_id": b"w2",
        }
    )
    shard = root / "embeddings" / "window_id=w2" / "fixture.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(table, shard)


def test_dataset_builder_merges_sources_and_writes_spatial_splits() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        crop_root = root / "crop"
        noncrop_root = root / "noncrop"
        _write_embedding_fixture(crop_root, "crop-run", "Maize", 0)
        _write_embedding_fixture(
            noncrop_root,
            "noncrop-run",
            "Non-crop: Tree cover",
            100,
        )
        output = root / "dataset.parquet"
        result = build(
            argparse.Namespace(
                crop_root=str(crop_root),
                noncrop_root=str(noncrop_root),
                window_id="w2",
                crop_labels=["Maize"],
                max_pixels_per_class=10_000,
                spatial_block_km=10.0,
                train_ratio=0.6,
                validation_ratio=0.2,
                test_ratio=0.2,
                split_attempts=256,
                seed=31,
                output=str(output),
                manifest=None,
            )
        )

        dataset = pd.read_parquet(output)
        assert result["rows"] == 60
        assert dataset["embedding"].map(len).eq(128).all()
        assert dataset.groupby("spatial_block")["split"].nunique().eq(1).all()
        assert set(dataset["label"]) == {"Maize", "Non-crop: Tree cover"}
