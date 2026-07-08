"""Shared, read-only loaders for the plain-TESSERA analysis notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from plain_tessera_incremental.storage import EMBEDDING_COLUMNS, canonical_sha256


EMBEDDING_DIM = 128
PIXEL_SIZE_M = 10


def _read_parquet(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    return pq.ParquetFile(path).read(columns=columns).to_pandas()


def _metadata(path: Path) -> dict[str, str]:
    raw = pq.ParquetFile(path).metadata.metadata or {}
    return {key.decode(): value.decode() for key, value in raw.items()}


def _expected_row_id(
    run_fingerprint: str,
    field_uid: str,
    pixel_id: str,
    window_id: str,
) -> str:
    payload = f"{run_fingerprint}:{field_uid}:{pixel_id}:{window_id}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _vector(value: object) -> np.ndarray | None:
    if value is None:
        return None
    result = np.asarray(value, dtype=np.float32)
    if result.shape != (EMBEDDING_DIM,) or not np.isfinite(result).all():
        return None
    return result


@dataclass(frozen=True)
class StaticArtifacts:
    output_dir: Path
    run: dict[str, object]
    run_fingerprint: str
    fields: pd.DataFrame
    pixels: pd.DataFrame
    memberships: pd.DataFrame
    field_lookup: pd.DataFrame
    pixel_lookup: pd.DataFrame
    expected_pair_index: pd.MultiIndex
    expected_pixels: pd.Series
    expected_task_positions: dict[str, np.ndarray]
    window_specs: dict[str, dict[str, object]]


@dataclass(frozen=True)
class WindowScan:
    window_id: str
    files: tuple[Path, ...]
    index: pd.DataFrame
    published_fields: frozenset[str]
    task_fingerprints: dict[str, str]


@dataclass(frozen=True)
class CleanWindow:
    rows: pd.DataFrame
    diagnostics: pd.Series
    canonical_fields: frozenset[str]


def load_static(output_dir: str | Path) -> StaticArtifacts:
    output_dir = Path(output_dir)
    required = ["run.json", "fields.parquet", "pixels.parquet", "field_pixels.parquet"]
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pipeline artifacts: {missing}")

    run = json.loads((output_dir / "run.json").read_text())
    run_fingerprint = str(run["run_fingerprint"])
    for artifact in ("fields", "pixels", "field_pixels"):
        path = output_dir / f"{artifact}.parquet"
        expected = {
            "run_fingerprint": run_fingerprint,
            "artifact": artifact,
            "schema_version": "1",
        }
        actual = _metadata(path)
        mismatched = {
            key: (actual.get(key), value)
            for key, value in expected.items()
            if actual.get(key) != value
        }
        if mismatched:
            raise RuntimeError(f"Artifact metadata mismatch for {path}: {mismatched}")

    fields = _read_parquet(
        output_dir / "fields.parquet",
        columns=[
            "field_uid",
            "id",
            "landcover",
            "wkt",
            "utm_epsg",
            "pixel_count",
            "duplicate_count",
            "geometry_sha256",
        ],
    )
    pixels = _read_parquet(
        output_dir / "pixels.parquet",
        columns=[
            "pixel_id",
            "utm_epsg",
            "pixel_x_index",
            "pixel_y_index",
            "pixel_longitude",
            "pixel_latitude",
        ],
    )
    memberships = _read_parquet(
        output_dir / "field_pixels.parquet",
        columns=[
            "field_uid",
            "source_id",
            "landcover",
            "quadkey",
            "pixel_id",
            "utm_epsg",
            "pixel_x_index",
            "pixel_y_index",
            "pixel_longitude",
            "pixel_latitude",
            "work_x_index",
            "work_y_index",
            "chunk_x_index",
            "chunk_y_index",
            "overlap_field_count",
            "label_conflict",
        ],
    )

    if fields["field_uid"].duplicated().any():
        raise RuntimeError("fields.parquet contains duplicate field_uid values")
    if pixels["pixel_id"].duplicated().any():
        raise RuntimeError("pixels.parquet contains duplicate pixel_id values")
    if memberships.duplicated(["field_uid", "pixel_id"]).any():
        raise RuntimeError("field_pixels.parquet contains duplicate memberships")

    fields["field_uid"] = fields["field_uid"].astype(str)
    fields["landcover"] = fields["landcover"].map(lambda value: str(value).strip())
    memberships["field_uid"] = memberships["field_uid"].astype(str)
    memberships["pixel_id"] = memberships["pixel_id"].astype(str)
    memberships["landcover"] = memberships["landcover"].map(
        lambda value: str(value).strip()
    )
    pixels["pixel_id"] = pixels["pixel_id"].astype(str)
    if fields["landcover"].eq("").any():
        raise RuntimeError("fields.parquet contains an empty label")

    field_lookup = fields.set_index("field_uid")
    pixel_lookup = pixels.set_index("pixel_id")
    if not memberships["field_uid"].isin(field_lookup.index).all():
        raise RuntimeError("field_pixels.parquet references an unknown field_uid")
    if not memberships["pixel_id"].isin(pixel_lookup.index).all():
        raise RuntimeError("field_pixels.parquet references an unknown pixel_id")
    expected_labels = memberships["field_uid"].map(field_lookup["landcover"])
    if not memberships["landcover"].eq(expected_labels).all():
        raise RuntimeError("field_pixels.parquet labels disagree with fields.parquet")

    expected_pixels = memberships.groupby("field_uid", sort=False).size()
    declared_pixels = field_lookup["pixel_count"].astype(np.int64)
    if not expected_pixels.sort_index().equals(
        declared_pixels.reindex(expected_pixels.index).sort_index()
    ):
        raise RuntimeError("Declared field pixel counts disagree with memberships")

    reconstructed_ids = (
        "utm-"
        + pixels["utm_epsg"].astype(np.int64).astype(str)
        + f"-{PIXEL_SIZE_M}m-"
        + pixels["pixel_x_index"].astype(np.int64).astype(str)
        + "-"
        + pixels["pixel_y_index"].astype(np.int64).astype(str)
    )
    if not pixels["pixel_id"].eq(reconstructed_ids).all():
        raise RuntimeError("pixels.parquet grid coordinates do not reproduce pixel_id")

    for column in ("utm_epsg", "pixel_x_index", "pixel_y_index"):
        expected = memberships["pixel_id"].map(pixel_lookup[column]).to_numpy(np.int64)
        if not np.array_equal(memberships[column].to_numpy(np.int64), expected):
            raise RuntimeError(f"field_pixels.parquet {column} disagrees with pixels.parquet")
    for column in ("pixel_longitude", "pixel_latitude"):
        expected = memberships["pixel_id"].map(pixel_lookup[column]).to_numpy(np.float64)
        if not np.allclose(
            memberships[column].to_numpy(np.float64), expected, rtol=0.0, atol=1e-10
        ):
            raise RuntimeError(f"field_pixels.parquet {column} disagrees with pixels.parquet")

    recomputed_overlap = memberships.groupby("pixel_id")["field_uid"].transform("nunique")
    recomputed_conflict = memberships.groupby("pixel_id")["landcover"].transform(
        "nunique"
    ).gt(1)
    if not memberships["overlap_field_count"].astype(np.int64).eq(
        recomputed_overlap.astype(np.int64)
    ).all():
        raise RuntimeError("field_pixels.parquet overlap counts are inconsistent")
    if not memberships["label_conflict"].astype(bool).eq(recomputed_conflict).all():
        raise RuntimeError("field_pixels.parquet label conflicts are inconsistent")

    group_columns = [
        "utm_epsg",
        "work_x_index",
        "work_y_index",
        "chunk_x_index",
        "chunk_y_index",
    ]
    expected_task_positions: dict[str, np.ndarray] = {}
    for group_key, positions in memberships.groupby(
        group_columns, sort=True
    ).indices.items():
        epsg, work_x, work_y, chunk_x, chunk_y = (int(value) for value in group_key)
        task_key = canonical_sha256(
            {
                "epsg": epsg,
                "work_x": work_x,
                "work_y": work_y,
                "chunk_x": chunk_x,
                "chunk_y": chunk_y,
            }
        )[:24]
        if task_key in expected_task_positions:
            raise RuntimeError(f"Task-key collision in field_pixels.parquet: {task_key}")
        position_array = np.asarray(positions, dtype=np.int64)
        position_array.sort()
        expected_task_positions[task_key] = position_array

    raw_specs = run.get("config", {}).get("windows", [])
    window_specs = {
        str(spec["window_id"]): dict(spec)
        for spec in raw_specs
        if isinstance(spec, dict) and "window_id" in spec
    }
    if not window_specs:
        window_specs = {
            path.name.split("=", 1)[1]: {"window_id": path.name.split("=", 1)[1]}
            for path in (output_dir / "embeddings").glob("window_id=w*")
        }

    return StaticArtifacts(
        output_dir=output_dir,
        run=run,
        run_fingerprint=run_fingerprint,
        fields=fields,
        pixels=pixels,
        memberships=memberships,
        field_lookup=field_lookup,
        pixel_lookup=pixel_lookup,
        expected_pair_index=pd.MultiIndex.from_frame(
            memberships[["field_uid", "pixel_id"]]
        ),
        expected_pixels=expected_pixels,
        expected_task_positions=expected_task_positions,
        window_specs=window_specs,
    )


def scan_window(
    static: StaticArtifacts,
    window_id: str,
    retained_field_uids: set[str] | frozenset[str] | None = None,
) -> WindowScan:
    """Validate every published shard, retaining optional field rows in the index."""
    if window_id not in static.window_specs:
        raise RuntimeError(f"Window {window_id!r} is not present in run.json")
    retained_fields = (
        None
        if retained_field_uids is None
        else {str(field_uid) for field_uid in retained_field_uids}
    )
    if retained_fields is not None:
        unknown_fields = retained_fields - set(static.field_lookup.index)
        if unknown_fields:
            raise RuntimeError(
                "Requested retained fields are unknown: "
                + ", ".join(sorted(unknown_fields))
            )
    shard_dir = static.output_dir / "embeddings" / f"window_id={window_id}"
    files = tuple(
        sorted(
            path
            for path in shard_dir.glob("*.parquet")
            if not path.name.endswith(".part")
        )
    )
    if not files:
        raise RuntimeError(f"No published {window_id} shards are available")

    index_columns = [
        "row_id",
        "run_fingerprint",
        "field_uid",
        "landcover",
        "pixel_id",
        "utm_epsg",
        "pixel_x_index",
        "pixel_y_index",
        "pixel_longitude",
        "pixel_latitude",
        "window_id",
        "window_ordinal",
        "window_start",
        "window_end_exclusive",
        "window_duration_days",
        "s2_source_count",
        "s1_source_count",
        "s2_valid_count",
        "s1_valid_count",
        "s2_input_count",
        "s1_input_count",
        "outcome",
    ]
    seen = np.zeros(len(static.memberships), dtype=bool)
    parts: list[pd.DataFrame] = []
    seen_tasks: set[str] = set()
    task_fingerprints: dict[str, str] = {}
    spec = static.window_specs[window_id]
    field_positions = pd.Series(
        np.arange(len(static.expected_pixels), dtype=np.int64),
        index=static.expected_pixels.index,
    )
    actual_field_counts = np.zeros(len(static.expected_pixels), dtype=np.int64)
    count_columns = [
        "s2_source_count",
        "s1_source_count",
        "s2_valid_count",
        "s1_valid_count",
        "s2_input_count",
        "s1_input_count",
    ]
    compact_columns = [
        "field_uid",
        "pixel_id",
        "outcome",
        "window_ordinal",
        "window_start",
        "window_end_exclusive",
        "window_duration_days",
        *count_columns,
    ]

    for path_code, path in enumerate(files):
        parquet = pq.ParquetFile(path)
        metadata = _metadata(path)
        expected_metadata = {
            "run_fingerprint": static.run_fingerprint,
            "artifact": "pixel_embeddings",
            "schema_version": "1",
            "window_id": window_id,
        }
        mismatch = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatch:
            raise RuntimeError(f"Shard metadata mismatch for {path}: {mismatch}")
        task_key = metadata.get("task_key")
        task_fingerprint = metadata.get("task_fingerprint")
        if not task_key or not task_fingerprint:
            raise RuntimeError(f"Shard task metadata is missing: {path}")
        if task_key in seen_tasks or path.stem != task_key:
            raise RuntimeError(f"Duplicate or misnamed task shard: {path}")
        seen_tasks.add(task_key)
        task_fingerprints[task_key] = task_fingerprint
        missing_columns = EMBEDDING_COLUMNS - set(parquet.schema_arrow.names)
        if missing_columns:
            raise RuntimeError(f"Shard {path} is missing {sorted(missing_columns)}")
        embedding_type = parquet.schema_arrow.field("embedding").type
        if not pa.types.is_list(embedding_type) or embedding_type.value_type != pa.float32():
            raise RuntimeError(f"Shard {path} has embedding type {embedding_type}")

        part = _read_parquet(path, columns=index_columns)
        if part.empty:
            raise RuntimeError(f"Published shard is empty: {path}")
        part["field_uid"] = part["field_uid"].astype(str)
        part["pixel_id"] = part["pixel_id"].astype(str)
        part["landcover"] = part["landcover"].map(lambda value: str(value).strip())
        if not part["run_fingerprint"].eq(static.run_fingerprint).all():
            raise RuntimeError(f"Shard row run fingerprint mismatch: {path}")
        if not part["window_id"].eq(window_id).all():
            raise RuntimeError(f"Shard row window mismatch: {path}")
        if not set(part["outcome"].unique()) <= {"complete", "empty_window"}:
            raise RuntimeError(f"Unknown outcome in {path}")

        positions = static.expected_pair_index.get_indexer(
            pd.MultiIndex.from_frame(part[["field_uid", "pixel_id"]])
        )
        if (positions < 0).any():
            raise RuntimeError(f"Shard contains an unknown membership: {path}")
        if (
            part["row_id"].duplicated().any()
            or len(np.unique(positions)) != len(positions)
            or seen[positions].any()
        ):
            raise RuntimeError(f"Duplicate field/pixel/window row in {path}")
        expected_positions = static.expected_task_positions.get(task_key)
        if (
            expected_positions is None
            or len(positions) != len(expected_positions)
            or not np.array_equal(np.sort(positions), expected_positions)
        ):
            raise RuntimeError(f"Shard membership set disagrees with task {task_key}")
        calculated_ids = [
            _expected_row_id(static.run_fingerprint, field_uid, pixel_id, window_id)
            for field_uid, pixel_id in part[
                ["field_uid", "pixel_id"]
            ].itertuples(index=False, name=None)
        ]
        if not part["row_id"].astype(str).eq(calculated_ids).all():
            raise RuntimeError(f"Deterministic row_id mismatch in {path}")

        field_labels = part["field_uid"].map(static.field_lookup["landcover"])
        if field_labels.isna().any() or not part["landcover"].eq(field_labels).all():
            raise RuntimeError(f"Shard labels disagree with fields.parquet: {path}")
        expected_grid = static.pixel_lookup.reindex(part["pixel_id"])
        for column in ("utm_epsg", "pixel_x_index", "pixel_y_index"):
            if not np.array_equal(
                part[column].to_numpy(np.int64),
                expected_grid[column].to_numpy(np.int64),
            ):
                raise RuntimeError(f"Shard {column} disagrees with pixels.parquet: {path}")
        for column in ("pixel_longitude", "pixel_latitude"):
            if not np.allclose(
                part[column].to_numpy(np.float64),
                expected_grid[column].to_numpy(np.float64),
                rtol=0.0,
                atol=1e-10,
            ):
                raise RuntimeError(f"Shard {column} disagrees with pixels.parquet: {path}")

        if "ordinal" in spec and not part["window_ordinal"].eq(int(spec["ordinal"])).all():
            raise RuntimeError(f"Shard ordinal mismatch: {path}")
        if "start" in spec and not part["window_start"].astype(str).eq(str(spec["start"])).all():
            raise RuntimeError(f"Shard start mismatch: {path}")
        if "end_exclusive" in spec and not part["window_end_exclusive"].astype(str).eq(
            str(spec["end_exclusive"])
        ).all():
            raise RuntimeError(f"Shard end mismatch: {path}")

        if (part[count_columns].to_numpy(np.int64) < 0).any():
            raise RuntimeError(f"Shard contains a negative observation count: {path}")
        seen[positions] = True
        part_field_positions = part["field_uid"].map(field_positions)
        if part_field_positions.isna().any():
            raise RuntimeError(f"Shard references an unknown field: {path}")
        np.add.at(
            actual_field_counts,
            part_field_positions.to_numpy(dtype=np.int64),
            1,
        )
        retained_part = (
            part
            if retained_fields is None
            else part[part["field_uid"].isin(retained_fields)]
        )
        if retained_part.empty:
            continue
        compact = retained_part[compact_columns].copy()
        compact["_path_code"] = np.int32(path_code)
        parts.append(compact)

    index = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=[*compact_columns, "_path_code"])
    )
    published_fields = static.expected_pixels.index[
        actual_field_counts == static.expected_pixels.to_numpy(dtype=np.int64)
    ]
    return WindowScan(
        window_id=window_id,
        files=files,
        index=index,
        published_fields=frozenset(published_fields.astype(str)),
        task_fingerprints=task_fingerprints,
    )


def clean_window_rows(
    static: StaticArtifacts,
    scan: WindowScan,
    published_fields: set[str] | frozenset[str] | None = None,
) -> CleanWindow:
    published = set(scan.published_fields if published_fields is None else published_fields)
    if not published <= set(scan.published_fields):
        raise RuntimeError(f"Requested fields are not all published in {scan.window_id}")

    fields = static.fields
    geometry_label_counts = fields.groupby("geometry_sha256")["landcover"].nunique()
    conflicting_hashes = set(geometry_label_counts[geometry_label_counts.gt(1)].index)
    conflicting_fields = set(
        fields.loc[fields["geometry_sha256"].isin(conflicting_hashes), "field_uid"]
    )
    conflicting_pixels = set(
        static.memberships.loc[
            static.memberships["field_uid"].isin(conflicting_fields), "pixel_id"
        ]
    )

    unanimous = fields[~fields["geometry_sha256"].isin(conflicting_hashes)].copy()
    unanimous["_published_priority"] = unanimous["field_uid"].isin(published)
    unanimous = unanimous.sort_values(
        ["geometry_sha256", "_published_priority", "field_uid"],
        ascending=[True, False, True],
        kind="stable",
    )
    canonical = set(
        unanimous.groupby("geometry_sha256", sort=False).head(1)["field_uid"]
    )
    replicas = set(unanimous["field_uid"]) - canonical
    excluded_fields = conflicting_fields | replicas

    effective = static.memberships[
        ~static.memberships["field_uid"].isin(excluded_fields)
    ]
    effective_overlap = effective.groupby("pixel_id")["field_uid"].nunique()
    effective_label_count = effective.groupby("pixel_id")["landcover"].nunique()

    rows = scan.index.copy()
    rows["landcover"] = rows["field_uid"].map(static.field_lookup["landcover"])
    rows["pixel_count"] = rows["field_uid"].map(static.field_lookup["pixel_count"])
    rows["_effective_overlap"] = rows["pixel_id"].map(effective_overlap).fillna(0)
    rows["_effective_label_conflict"] = rows["pixel_id"].map(
        effective_label_count
    ).fillna(0).gt(1)
    clean = rows[
        rows["field_uid"].isin(published)
        & rows["outcome"].eq("complete")
        & ~rows["field_uid"].isin(excluded_fields)
        & ~rows["pixel_id"].isin(conflicting_pixels)
        & rows["_effective_overlap"].eq(1)
        & ~rows["_effective_label_conflict"]
    ].copy()
    if clean["pixel_id"].duplicated().any():
        raise RuntimeError("Clean modelling rows contain a repeated physical pixel")

    touching_overlap = set(
        static.memberships.loc[
            static.memberships["overlap_field_count"].gt(1), "field_uid"
        ]
    )
    diagnostics = pd.Series(
        {
            "all_fields": len(fields),
            f"published_fields_{scan.window_id}": len(published),
            "fields_touching_shared_pixels": len(touching_overlap & published),
            "conflicting_geometry_fields_excluded": len(conflicting_fields & published),
            "same_label_geometry_replicas_excluded": len(replicas & published),
            "clean_complete_fields": clean["field_uid"].nunique(),
            "clean_complete_pixels": clean["pixel_id"].nunique(),
        },
        dtype=np.int64,
    )
    return CleanWindow(
        rows=clean.reset_index(drop=True),
        diagnostics=diagnostics,
        canonical_fields=frozenset(canonical),
    )


def balanced_sample(
    rows: pd.DataFrame,
    labels: list[str] | tuple[str, ...],
    max_fields_per_label: int | None,
    max_pixels_per_field: int | None,
    seed: int,
    min_fields_per_label: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = [str(label) for label in labels]
    available = (
        rows[rows["landcover"].isin(labels)]
        .groupby("landcover")["field_uid"]
        .nunique()
        .reindex(labels, fill_value=0)
    )
    missing = available[available.lt(min_fields_per_label)]
    if not missing.empty:
        raise RuntimeError(
            "Labels lack enough completed clean fields: "
            + ", ".join(f"{label}={count}" for label, count in missing.items())
        )

    rng = np.random.default_rng(seed)
    selected_fields: list[str] = []
    for label in labels:
        candidates = np.array(
            sorted(rows.loc[rows["landcover"].eq(label), "field_uid"].unique()),
            dtype=object,
        )
        if (
            max_fields_per_label is not None
            and len(candidates) > max_fields_per_label
        ):
            candidates = np.sort(
                rng.choice(candidates, size=max_fields_per_label, replace=False)
            )
        selected_fields.extend(candidates.tolist())

    parts: list[pd.DataFrame] = []
    for _, group in rows[rows["field_uid"].isin(selected_fields)].groupby(
        "field_uid", sort=True
    ):
        ordered = group.sort_values("pixel_id", kind="stable")
        if max_pixels_per_field is not None and len(ordered) > max_pixels_per_field:
            positions = np.linspace(
                0, len(ordered) - 1, max_pixels_per_field, dtype=np.int64
            )
            ordered = ordered.iloc[positions]
        parts.append(ordered)
    sampled = pd.concat(parts, ignore_index=True)
    if sampled["pixel_id"].duplicated().any():
        raise RuntimeError("Balanced sample contains a repeated physical pixel")
    summary = sampled.groupby("landcover").agg(
        fields=("field_uid", "nunique"),
        pixel_samples=("pixel_id", "nunique"),
    ).reindex(labels)
    return sampled, summary


def load_embeddings(
    static: StaticArtifacts,
    scan: WindowScan,
    selected_rows: pd.DataFrame,
) -> pd.DataFrame:
    selected = selected_rows.copy()
    if selected.duplicated(["field_uid", "pixel_id"]).any():
        raise RuntimeError("Selected rows contain duplicate memberships")
    selected["row_id"] = [
        _expected_row_id(static.run_fingerprint, field_uid, pixel_id, scan.window_id)
        for field_uid, pixel_id in selected[["field_uid", "pixel_id"]].itertuples(
            index=False, name=None
        )
    ]

    parts: list[pd.DataFrame] = []
    for path_code, group in selected.groupby("_path_code", sort=False):
        row_ids = sorted(set(group["row_id"].astype(str)))
        part = pq.read_table(
            scan.files[int(path_code)],
            columns=["row_id", "embedding"],
            filters=[("row_id", "in", row_ids)],
            partitioning=None,
        ).to_pandas()
        parts.append(part)
    embeddings = pd.concat(parts, ignore_index=True)
    if embeddings["row_id"].duplicated().any():
        raise RuntimeError("Filtered embedding read returned duplicate row IDs")
    result = selected.merge(
        embeddings,
        on="row_id",
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if not result["_merge"].eq("both").all():
        raise RuntimeError("Filtered embedding read did not return every selected row")
    result = result.drop(columns="_merge")
    result["_vector"] = result["embedding"].map(_vector)
    if result["_vector"].isna().any():
        raise RuntimeError("A selected complete row lacks a finite float32[128] vector")
    return result.drop(columns="embedding")


def l2_features(rows: pd.DataFrame) -> np.ndarray:
    matrix = np.stack(rows["_vector"].to_numpy()).astype(np.float64)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if (norms <= 0).any():
        raise RuntimeError("A selected embedding has zero norm")
    return matrix / norms
