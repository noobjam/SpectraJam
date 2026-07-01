from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date

import pytest

from spectrajam.contracts import ContractError, sha256_file
from spectrajam.pointstore import (
    S1_ORBIT_ASCENDING,
    S1_ORBIT_DESCENDING,
    point_store_schema,
    read_point_store,
    s1_schema,
    s2_schema,
    write_point_store,
)

pa = pytest.importorskip("pyarrow")

_EPOCH = date(1970, 1, 1)


def _epoch_day(value: str) -> int:
    return (date.fromisoformat(value) - _EPOCH).days


def _s2_record(sample_id: str = "sample-b", day: str = "2024-01-02") -> dict[str, object]:
    absolute_day = _epoch_day(day)
    return {
        "sample_id": sample_id,
        "country": "RWA",
        "year": 2024,
        "epoch_day": absolute_day,
        "day_of_year": absolute_day - _epoch_day("2024-01-01") + 1,
        "source_item_id": f"s2-{sample_id}",
        "source_item_sha256": "a" * 64,
        "catalog_query_sha256": "b" * 64,
        "bands": [0, 1, 2, 3, 4, 5, 10_000, 32_768, 65_534, 65_535],
        "scl": 4,
        "valid": True,
    }


def _s1_record() -> dict[str, object]:
    return {
        "sample_id": "sample-a",
        "country": "ISR",
        "year": 2023,
        "epoch_day": _epoch_day("2023-12-31"),
        "day_of_year": 365,
        "source_item_id": "s1-item-a",
        "source_item_sha256": "c" * 64,
        "catalog_query_sha256": "d" * 64,
        "bands": [0, 0],
        "orbit": S1_ORBIT_DESCENDING,
        "valid": False,
    }


def test_canonical_schemas_preserve_exact_integer_contract() -> None:
    s2 = s2_schema()
    assert s2.names == [
        "sample_id",
        "country",
        "year",
        "epoch_day",
        "day_of_year",
        "source_item_id",
        "source_item_sha256",
        "catalog_query_sha256",
        "bands",
        "scl",
        "valid",
    ]
    assert pa.types.is_fixed_size_list(s2.field("bands").type)
    assert s2.field("bands").type.list_size == 10
    assert s2.field("bands").type.value_type == pa.uint16()
    assert s2.field("scl").type == pa.uint8()
    assert s2.field("valid").type == pa.bool_()
    assert all(not field.nullable for field in s2)
    assert s2.metadata[b"spectrajam.pointstore.modality"] == b"s2"
    assert s2.metadata[b"spectrajam.pointstore.provider_profile"] == b"mpc-v1.1"
    assert s2.metadata[b"spectrajam.pointstore.preprocessing"].startswith(
        b"tessera-v1.1-mpc"
    )

    s1 = s1_schema()
    assert pa.types.is_fixed_size_list(s1.field("bands").type)
    assert s1.field("bands").type.list_size == 2
    assert s1.field("bands").type.value_type == pa.int16()
    assert s1.field("orbit").type == pa.int8()
    assert s1.metadata[b"spectrajam.pointstore.band_order"] == b"VV,VH"
    assert S1_ORBIT_ASCENDING == 1
    assert S1_ORBIT_DESCENDING == -1
    assert point_store_schema("s1").equals(s1, check_metadata=True)


@pytest.mark.parametrize(
    ("modality", "records"),
    [
        ("s2", [_s2_record(), _s2_record("sample-a", "2024-01-01")]),
        ("s1", [_s1_record()]),
    ],
)
def test_atomic_parquet_roundtrip_is_sorted_and_checksummed(
    tmp_path, modality: str, records: list[dict[str, object]]
) -> None:
    destination = tmp_path / f"{modality}.parquet"
    result = write_point_store(destination, modality, reversed(records))

    assert result.path == destination
    assert result.row_count == len(records)
    assert result.size_bytes == destination.stat().st_size
    assert result.sha256 == sha256_file(destination)
    assert not list(tmp_path.glob("*.part"))

    table = read_point_store(destination, modality)
    assert table.schema.equals(point_store_schema(modality), check_metadata=True)
    assert table.num_rows == len(records)
    if modality == "s2":
        assert table.column("sample_id").to_pylist() == ["sample-a", "sample-b"]
        assert table.column("bands").type.value_type == pa.uint16()
    else:
        assert table.column("bands").type.value_type == pa.int16()


def test_row_order_produces_deterministic_parquet(tmp_path) -> None:
    records = [_s2_record(), _s2_record("sample-a", "2024-01-01")]
    first = write_point_store(tmp_path / "first.parquet", "s2", records)
    second = write_point_store(tmp_path / "second.parquet", "s2", reversed(records))
    assert first.sha256 == second.sha256


@pytest.mark.parametrize(
    ("modality", "field", "value", "message"),
    [
        ("s2", "bands", [1] * 9, "exactly 10"),
        ("s2", "bands", [1] * 9 + [65_536], r"\[0, 65535\]"),
        ("s2", "scl", 12, r"\[0, 11\]"),
        ("s2", "day_of_year", 20, "do not match"),
        ("s2", "valid", 1, "must be bool"),
        ("s2", "valid", False, "S2 valid"),
        ("s1", "bands", [-1, 1], r"\[0, 32767\]"),
        ("s1", "orbit", 0, "descending.*ascending"),
        ("s1", "orbit", 128, r"\[-128, 127\]"),
        ("s1", "valid", True, r"any\(band != 0\)"),
        ("s2", "source_item_sha256", "not-a-digest", "SHA-256"),
    ],
)
def test_invalid_values_are_rejected_without_partial_files(
    tmp_path, modality: str, field: str, value: object, message: str
) -> None:
    record = deepcopy(_s2_record() if modality == "s2" else _s1_record())
    record[field] = value
    destination = tmp_path / "invalid.parquet"

    with pytest.raises(ContractError, match=message):
        write_point_store(destination, modality, [record])

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_reader_rejects_the_wrong_sensor_schema(tmp_path) -> None:
    destination = tmp_path / "s1.parquet"
    write_point_store(destination, "s1", [_s1_record()])
    with pytest.raises(ContractError, match="schema"):
        read_point_store(destination, "s2")


def test_reader_can_enforce_the_committed_checksum(tmp_path) -> None:
    destination = tmp_path / "s2.parquet"
    result = write_point_store(destination, "s2", [_s2_record()])
    assert read_point_store(destination, "s2", result.sha256).num_rows == 1
    with pytest.raises(ContractError, match="SHA-256 mismatch"):
        read_point_store(destination, "s2", "f" * 64)


def test_empty_shards_are_rejected(tmp_path) -> None:
    destination = tmp_path / "empty.parquet"
    with pytest.raises(ContractError, match="at least one observation"):
        write_point_store(destination, "s2", [])
    assert not destination.exists()


def test_existing_shard_is_reused_or_refused_but_never_replaced(tmp_path) -> None:
    destination = tmp_path / "s2.parquet"
    original = write_point_store(destination, "s2", [_s2_record()])
    reused = write_point_store(destination, "s2", [_s2_record()])
    assert reused.sha256 == original.sha256

    with pytest.raises(ContractError, match="immutable"):
        write_point_store(destination, "s2", [_s2_record("different")])
    assert sha256_file(destination) == original.sha256


def test_duplicate_observation_identity_is_rejected(tmp_path) -> None:
    first = _s2_record()
    duplicate = deepcopy(first)
    duplicate["source_item_id"] = "another-overlap-item"
    duplicate["source_item_sha256"] = "e" * 64
    with pytest.raises(ContractError, match="duplicate s2 observation"):
        write_point_store(tmp_path / "duplicate.parquet", "s2", [first, duplicate])


def test_concurrent_writers_cannot_replace_each_other(tmp_path) -> None:
    destination = tmp_path / "race.parquet"

    def write(sample_id: str):
        try:
            return write_point_store(destination, "s2", [_s2_record(sample_id)])
        except ContractError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(write, ("first", "second")))

    assert sum(isinstance(value, ContractError) for value in outcomes) == 1
    assert read_point_store(destination, "s2").column("sample_id").to_pylist() in [
        ["first"],
        ["second"],
    ]
