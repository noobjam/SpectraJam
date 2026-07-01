from __future__ import annotations

import operator
import os
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from .contracts import (
    CANONICAL_S1_BANDS,
    CANONICAL_S2_BANDS,
    TESSERA_UPSTREAM_COMMIT,
    ContractError,
    sha256_file,
)

Modality = Literal["s2", "s1"]
S1_ORBIT_ASCENDING = 1
S1_ORBIT_DESCENDING = -1

_EPOCH = date(1970, 1, 1)
_SCHEMA_VERSION = "1"
_COMMON_FIELDS = (
    "sample_id",
    "country",
    "year",
    "epoch_day",
    "day_of_year",
    "source_item_id",
    "source_item_sha256",
    "catalog_query_sha256",
    "bands",
)
_SCL_INVALID = {0, 1, 2, 3, 8, 9}


@dataclass(frozen=True, slots=True)
class PointStoreWriteResult:
    path: Path
    modality: Modality
    row_count: int
    size_bytes: int
    sha256: str


@lru_cache(maxsize=1)
def _require_pyarrow() -> tuple[Any, Any]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "PyArrow is required for point-store support; install it with "
            "`python -m pip install 'spectrajam[data]'`."
        ) from error
    return pa, pq


def _schema_metadata(modality: Modality) -> dict[bytes, bytes]:
    bands = CANONICAL_S2_BANDS if modality == "s2" else CANONICAL_S1_BANDS
    metadata = {
        b"spectrajam.pointstore.schema_version": _SCHEMA_VERSION.encode(),
        b"spectrajam.pointstore.modality": modality.encode(),
        b"spectrajam.pointstore.band_order": ",".join(bands).encode(),
        b"spectrajam.pointstore.provider_profile": b"mpc-v1.1",
        b"spectrajam.pointstore.upstream_commit": TESSERA_UPSTREAM_COMMIT.encode(),
        b"spectrajam.pointstore.preprocessing": b"tessera-v1.1-mpc-compatibility",
    }
    if modality == "s1":
        metadata[b"spectrajam.pointstore.orbit_encoding"] = (
            b"-1=descending,1=ascending"
        )
    return metadata


def s2_schema() -> Any:
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("country", pa.string(), nullable=False),
            pa.field("year", pa.int16(), nullable=False),
            pa.field("epoch_day", pa.int32(), nullable=False),
            pa.field("day_of_year", pa.int16(), nullable=False),
            pa.field("source_item_id", pa.string(), nullable=False),
            pa.field("source_item_sha256", pa.string(), nullable=False),
            pa.field("catalog_query_sha256", pa.string(), nullable=False),
            pa.field(
                "bands",
                pa.list_(pa.field("element", pa.uint16()), 10),
                nullable=False,
            ),
            pa.field("scl", pa.uint8(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ],
        metadata=_schema_metadata("s2"),
    )


def s1_schema() -> Any:
    pa, _ = _require_pyarrow()
    return pa.schema(
        [
            pa.field("sample_id", pa.string(), nullable=False),
            pa.field("country", pa.string(), nullable=False),
            pa.field("year", pa.int16(), nullable=False),
            pa.field("epoch_day", pa.int32(), nullable=False),
            pa.field("day_of_year", pa.int16(), nullable=False),
            pa.field("source_item_id", pa.string(), nullable=False),
            pa.field("source_item_sha256", pa.string(), nullable=False),
            pa.field("catalog_query_sha256", pa.string(), nullable=False),
            pa.field(
                "bands",
                pa.list_(pa.field("element", pa.int16()), 2),
                nullable=False,
            ),
            pa.field("orbit", pa.int8(), nullable=False),
            pa.field("valid", pa.bool_(), nullable=False),
        ],
        metadata=_schema_metadata("s1"),
    )


def point_store_schema(modality: Modality) -> Any:
    if modality == "s2":
        return s2_schema()
    if modality == "s1":
        return s1_schema()
    raise ContractError(f"unsupported point-store modality: {modality}")


def _integer(value: object, name: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool):
        raise ContractError(f"{name} must be an integer, not bool")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ContractError(f"{name} must be an integer") from error
    if not minimum <= result <= maximum:
        raise ContractError(f"{name} must be in [{minimum}, {maximum}], got {result}")
    return result


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be a non-empty string")
    return value


def _sha256(value: object, name: str) -> str:
    digest = _text(value, name).lower()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ContractError(f"{name} must be a hexadecimal SHA-256 digest")
    return digest


def _bands(value: object, modality: Modality) -> list[int]:
    if isinstance(value, (str, bytes)):
        raise ContractError("bands must be an integer sequence")
    try:
        values = list(value)  # type: ignore[arg-type]
    except TypeError as error:
        raise ContractError("bands must be an integer sequence") from error
    size = 10 if modality == "s2" else 2
    if len(values) != size:
        raise ContractError(f"{modality} bands must contain exactly {size} values")
    limits = (0, 65_535) if modality == "s2" else (0, 32_767)
    return [
        _integer(item, f"bands[{index}]", limits[0], limits[1])
        for index, item in enumerate(values)
    ]


def _normalize_record(record: Mapping[str, object], modality: Modality) -> dict[str, object]:
    sensor_fields = ("scl", "valid") if modality == "s2" else ("orbit", "valid")
    expected = set((*_COMMON_FIELDS, *sensor_fields))
    actual = set(record)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ContractError(f"point-store fields differ: missing={missing}, extra={extra}")

    sample_id = _text(record["sample_id"], "sample_id")
    country = _text(record["country"], "country")
    if country not in {"RWA", "ISR"}:
        raise ContractError(f"unsupported point-store country: {country}")
    year = _integer(record["year"], "year", 2017, 2100)
    epoch_day = _integer(record["epoch_day"], "epoch_day", -(2**31), 2**31 - 1)
    day_of_year = _integer(record["day_of_year"], "day_of_year", 1, 366)
    try:
        observed_date = _EPOCH + timedelta(days=epoch_day)
    except OverflowError as error:
        raise ContractError(f"epoch_day is outside the supported calendar: {epoch_day}") from error
    expected_day_of_year = observed_date.timetuple().tm_yday
    if observed_date.year != year or expected_day_of_year != day_of_year:
        raise ContractError(
            "year/day_of_year do not match epoch_day: "
            f"expected {observed_date.year}/{expected_day_of_year}"
        )
    source_item_id = _text(record["source_item_id"], "source_item_id")
    source_item_sha256 = _sha256(record["source_item_sha256"], "source_item_sha256")
    catalog_query_sha256 = _sha256(
        record["catalog_query_sha256"], "catalog_query_sha256"
    )
    bands = _bands(record["bands"], modality)
    if not isinstance(record["valid"], bool):
        raise ContractError("valid must be bool")

    normalized: dict[str, object] = {
        "sample_id": sample_id,
        "country": country,
        "year": year,
        "epoch_day": epoch_day,
        "day_of_year": day_of_year,
        "source_item_id": source_item_id,
        "source_item_sha256": source_item_sha256,
        "catalog_query_sha256": catalog_query_sha256,
        "bands": bands,
    }
    if modality == "s2":
        scl = _integer(record["scl"], "scl", 0, 11)
        if record["valid"] is not (scl not in _SCL_INVALID):
            raise ContractError("S2 valid must match the MPC compatibility SCL mask")
        normalized["scl"] = scl
    else:
        orbit = _integer(record["orbit"], "orbit", -128, 127)
        if orbit not in {S1_ORBIT_ASCENDING, S1_ORBIT_DESCENDING}:
            raise ContractError("orbit must be -1 (descending) or 1 (ascending)")
        if record["valid"] is not any(value != 0 for value in bands):
            raise ContractError("S1 valid must equal any(band != 0)")
        normalized["orbit"] = orbit
    normalized["valid"] = record["valid"]
    return normalized


def _sort_key(record: Mapping[str, object], modality: Modality) -> tuple[object, ...]:
    sensor_value = record["scl"] if modality == "s2" else record["orbit"]
    return (
        record["country"],
        record["year"],
        record["sample_id"],
        record["epoch_day"],
        record["source_item_id"],
        record["source_item_sha256"],
        record["catalog_query_sha256"],
        sensor_value,
        tuple(record["bands"]),  # type: ignore[arg-type]
        record["valid"],
    )


def _normalize_records(
    records: Iterable[Mapping[str, object]], modality: Modality
) -> list[dict[str, object]]:
    normalized = [_normalize_record(record, modality) for record in records]
    seen: set[tuple[object, ...]] = set()
    for record in normalized:
        identity = (
            (record["sample_id"], record["epoch_day"])
            if modality == "s2"
            else (record["sample_id"], record["epoch_day"], record["orbit"])
        )
        if identity in seen:
            raise ContractError(f"duplicate {modality} observation identity: {identity}")
        seen.add(identity)
    normalized.sort(key=lambda record: _sort_key(record, modality))
    return normalized


def _validate_schema(actual: Any, modality: Modality) -> None:
    expected = point_store_schema(modality)
    if not actual.equals(expected, check_metadata=True):
        raise ContractError(f"{modality} point-store Arrow schema does not match the contract")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_point_store(
    path: str | Path,
    modality: Modality,
    records: Iterable[Mapping[str, object]],
) -> PointStoreWriteResult:
    """Validate and atomically write a canonical Parquet/Zstd observation shard."""
    pa, pq = _require_pyarrow()
    schema = point_store_schema(modality)
    rows = _normalize_records(records, modality)
    if not rows:
        raise ContractError("point-store shards must contain at least one observation")
    table = pa.Table.from_pylist(rows, schema=schema)
    _validate_schema(table.schema, modality)

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.{uuid.uuid4().hex}.part")
    try:
        pq.write_table(
            table,
            temporary,
            compression="zstd",
            use_dictionary=[
                "sample_id",
                "country",
                "source_item_id",
                "source_item_sha256",
                "catalog_query_sha256",
            ],
            write_statistics=True,
        )
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        parquet_file = pq.ParquetFile(temporary)
        _validate_schema(parquet_file.schema_arrow, modality)
        if parquet_file.metadata.num_rows != len(rows):
            raise ContractError("point-store Parquet row count changed during serialization")
        checksum = sha256_file(temporary)
        size_bytes = temporary.stat().st_size
        try:
            os.link(temporary, destination)
        except FileExistsError as error:
            read_point_store(destination, modality)
            existing_checksum = sha256_file(destination)
            if existing_checksum != checksum:
                raise ContractError(
                    f"refusing to replace immutable point-store shard: {destination}"
                ) from error
            size_bytes = destination.stat().st_size
        _fsync_directory(destination.parent)
    finally:
        temporary.unlink(missing_ok=True)

    return PointStoreWriteResult(
        path=destination,
        modality=modality,
        row_count=len(rows),
        size_bytes=size_bytes,
        sha256=checksum,
    )


def read_point_store(
    path: str | Path, modality: Modality, expected_sha256: str | None = None
) -> Any:
    """Read a shard only after validating its schema, values, and canonical order."""
    _, pq = _require_pyarrow()
    source = Path(path)
    if expected_sha256 is not None:
        actual_sha256 = sha256_file(source)
        if actual_sha256 != _sha256(expected_sha256, "expected_sha256"):
            raise ContractError(
                f"point-store SHA-256 mismatch: {actual_sha256} != {expected_sha256.lower()}"
            )
    parquet_file = pq.ParquetFile(source)
    _validate_schema(parquet_file.schema_arrow, modality)
    table = parquet_file.read()
    _validate_schema(table.schema, modality)
    records = table.to_pylist()
    normalized = _normalize_records(records, modality)
    if records != normalized:
        raise ContractError("point-store rows are not in canonical order")
    return table
