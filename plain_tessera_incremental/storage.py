from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .inference import WindowEmbeddings
from .windows import PrefixWindow


EMBEDDING_COLUMNS = {
    "row_id",
    "run_fingerprint",
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
    "embedding",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True, default=str).encode()
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _metadata(run_fingerprint: str, artifact: str, **extra: str) -> dict[bytes, bytes]:
    values = {
        "run_fingerprint": run_fingerprint,
        "artifact": artifact,
        "schema_version": "1",
        **extra,
    }
    return {key.encode(): value.encode() for key, value in values.items()}


def parquet_matches(
    path: Path,
    expected: dict[str, str],
    expected_rows: int | None = None,
    required_columns: set[str] | None = None,
) -> bool:
    if not path.is_file():
        return False
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        file_metadata = pq.read_metadata(path)
        metadata = file_metadata.metadata or {}
        schema = pq.read_schema(path)
    except Exception as error:
        raise RuntimeError(f"existing Parquet shard is unreadable: {path}") from error
    actual = {key.decode(): value.decode() for key, value in metadata.items()}
    if any(actual.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"existing Parquet shard belongs to another run: {path}")
    if expected_rows is not None and file_metadata.num_rows != expected_rows:
        raise RuntimeError(
            f"existing Parquet row count is wrong: {path} "
            f"({file_metadata.num_rows} != {expected_rows})"
        )
    if required_columns is not None and not required_columns <= set(schema.names):
        missing = sorted(required_columns - set(schema.names))
        raise RuntimeError(f"existing Parquet shard is missing columns {missing}: {path}")
    if expected.get("artifact") == "pixel_embeddings":
        embedding_type = schema.field("embedding").type
        if not pa.types.is_list(embedding_type) or embedding_type.value_type != pa.float32():
            raise RuntimeError(f"existing embedding column has the wrong type: {path}")
    return True


def write_dataframe_atomic(
    frame: pd.DataFrame,
    path: Path,
    run_fingerprint: str,
    artifact: str,
) -> None:
    expected = {"run_fingerprint": run_fingerprint, "artifact": artifact}
    if parquet_matches(
        path,
        expected,
        expected_rows=len(frame),
        required_columns=set(frame.columns),
    ):
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for pipeline output") from error
    table = pa.Table.from_pandas(frame, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), **_metadata(run_fingerprint, artifact)}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pq.write_table(table, temporary, compression="zstd")
    if pq.read_metadata(temporary).num_rows != len(frame):
        raise RuntimeError(f"Parquet row-count validation failed: {temporary}")
    temporary.replace(path)


def _embedding_array(values: np.ndarray, null_mask: np.ndarray):
    import pyarrow as pa

    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 128:
        raise ValueError("complete embeddings must have exactly 128 float32 values")
    # Some Arrow/Parquet releases cannot round-trip null fixed-size-list rows;
    # use a nullable list and enforce width above instead of writing fake vectors.
    rows = [None if missing else row.tolist() for row, missing in zip(values, null_mask, strict=True)]
    return pa.array(rows, type=pa.list_(pa.float32()))


def write_embedding_shard(
    memberships: pd.DataFrame,
    results: WindowEmbeddings,
    window: PrefixWindow,
    path: Path,
    run_fingerprint: str,
    task_key: str,
    task_fingerprint: str,
) -> None:
    count = len(memberships)
    expected = {
        "run_fingerprint": run_fingerprint,
        "artifact": "pixel_embeddings",
        "window_id": window.window_id,
        "task_key": task_key,
        "task_fingerprint": task_fingerprint,
    }
    if parquet_matches(
        path,
        expected,
        expected_rows=count,
        required_columns=EMBEDDING_COLUMNS,
    ):
        return
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required for pipeline output") from error

    positions = memberships["pixel_position"].to_numpy(dtype=np.int64)
    outcome = results.outcome[positions]
    null_embedding = outcome != "complete"
    row_ids = [
        hashlib.sha256(
            f"{run_fingerprint}:{field}:{pixel}:{window.window_id}".encode()
        ).hexdigest()
        for field, pixel in zip(
            memberships["field_uid"], memberships["pixel_id"], strict=True
        )
    ]
    columns = {
        "row_id": pa.array(row_ids, pa.string()),
        "run_fingerprint": pa.array([run_fingerprint] * count, pa.string()),
        "field_uid": pa.array(memberships["field_uid"].astype(str), pa.string()),
        "source_id": pa.array(memberships["source_id"].astype(str), pa.string()),
        "landcover": pa.array(memberships["landcover"].astype(str), pa.string()),
        "quadkey": pa.array(memberships["quadkey"].astype(str), pa.string()),
        "pixel_id": pa.array(memberships["pixel_id"].astype(str), pa.string()),
        "utm_epsg": pa.array(memberships["utm_epsg"], pa.int32()),
        "pixel_x_index": pa.array(memberships["pixel_x_index"], pa.int64()),
        "pixel_y_index": pa.array(memberships["pixel_y_index"], pa.int64()),
        "pixel_longitude": pa.array(memberships["pixel_longitude"], pa.float64()),
        "pixel_latitude": pa.array(memberships["pixel_latitude"], pa.float64()),
        "window_id": pa.array([window.window_id] * count, pa.string()),
        "window_ordinal": pa.array([window.ordinal] * count, pa.uint8()),
        "window_start": pa.array([window.start] * count, pa.date32()),
        "window_end_exclusive": pa.array([window.end_exclusive] * count, pa.date32()),
        "window_duration_days": pa.array([window.duration_days] * count, pa.int32()),
        "s2_source_count": pa.array([results.s2_source_count] * count, pa.int32()),
        "s1_source_count": pa.array([results.s1_source_count] * count, pa.int32()),
        "s2_valid_count": pa.array(results.s2_valid_count[positions], pa.int32()),
        "s1_valid_count": pa.array(results.s1_valid_count[positions], pa.int32()),
        "s2_input_count": pa.array(results.s2_input_count[positions], pa.int32()),
        "s1_input_count": pa.array(results.s1_input_count[positions], pa.int32()),
        "outcome": pa.array(outcome, pa.string()),
        "embedding": _embedding_array(results.embeddings[positions], null_embedding),
    }
    table = pa.table(columns).replace_schema_metadata(
        _metadata(
            run_fingerprint,
            "pixel_embeddings",
            window_id=window.window_id,
            task_key=task_key,
            task_fingerprint=task_fingerprint,
        )
    )
    if table.num_rows != count or len(set(row_ids)) != count:
        raise RuntimeError("embedding shard has duplicate or missing row IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    pq.write_table(table, temporary, compression="zstd", use_dictionary=True)
    metadata = pq.read_metadata(temporary)
    if metadata.num_rows != count:
        raise RuntimeError(f"embedding shard row-count validation failed: {temporary}")
    temporary.replace(path)
