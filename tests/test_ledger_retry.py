import sqlite3
from pathlib import Path

import pytest
import requests

from spectrajam.contracts import ContractError, PointYear, stable_sample_id
from spectrajam.ledger import AcquisitionLedger, Artifact, IncompleteAcquisitionError
from spectrajam.retry import retry_after_seconds, run_due_tasks


def _record() -> PointYear:
    return PointYear(
        sample_id=stable_sample_id("RWA", "candidate", 2023),
        candidate_id="candidate",
        country="RWA",
        longitude=30.0,
        latitude=-2.0,
        spatial_block="block-1",
        stratum="forest",
        inclusion_probability=0.5,
        spatial_split="train",
        year_split="train",
        year=2023,
    )


def test_transient_failures_resume_without_losing_tasks(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    kwargs = {"manifest_sha256": "1" * 64, "config_sha256": "2" * 64}
    assert ledger.bootstrap([_record()], ["s1", "s2"], max_attempts=3, **kwargs) == 2
    assert ledger.bootstrap([_record()], ["s1", "s2"], max_attempts=3, **kwargs) == 0
    calls: dict[str, int] = {}

    def worker(task):
        calls[task.task_id] = calls.get(task.task_id, 0) + 1
        if calls[task.task_id] < 3:
            raise requests.Timeout("temporary")
        return Artifact("memory://artifact", "0" * 64, 12, {"ok": True})

    for _ in range(3):
        run_due_tasks(
            ledger,
            worker,
            worker_id="worker-a",
            limit=10,
            lease_seconds=60,
            base_delay_seconds=0,
            max_delay_seconds=0,
        )
    ledger.assert_complete()
    assert ledger.summary()["succeeded"] == 2
    assert set(calls.values()) == {3}


def test_terminal_failure_blocks_training_gate(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [_record()], ["s2"], max_attempts=2,
        manifest_sha256="1" * 64, config_sha256="2" * 64
    )

    def broken(_task):
        raise ValueError("schema mismatch")

    result = run_due_tasks(
        ledger,
        broken,
        worker_id="worker-a",
        limit=1,
        lease_seconds=60,
        base_delay_seconds=0,
        max_delay_seconds=0,
    )
    assert result.failed == 1
    with pytest.raises(IncompleteAcquisitionError):
        ledger.assert_complete()


def test_expired_lease_is_recovered(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [_record()], ["s2"], max_attempts=2,
        manifest_sha256="1" * 64, config_sha256="2" * 64
    )
    assert len(ledger.claim("dead-worker", 1, lease_seconds=1, now=10)) == 1
    assert ledger.recover_expired_leases(now=12) == 1
    assert len(ledger.claim("new-worker", 1, lease_seconds=10, now=12)) == 1


def test_retry_after_is_honored() -> None:
    response = requests.Response()
    response.status_code = 429
    response.headers["Retry-After"] = "17"
    error = requests.HTTPError(response=response)
    assert retry_after_seconds(error) == 17


def test_final_expired_lease_becomes_terminal_failure(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [_record()], ["s2"], max_attempts=1,
        manifest_sha256="1" * 64, config_sha256="2" * 64
    )
    assert ledger.claim("dead-worker", 1, lease_seconds=1, now=10)
    assert ledger.recover_expired_leases(now=12) == 1
    assert ledger.summary()["failed"] == 1


def test_empty_ledger_never_passes_completion_gate(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    with pytest.raises(IncompleteAcquisitionError, match="empty"):
        ledger.assert_complete()


def test_ledger_rejects_a_different_manifest_binding(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [_record()], ["s2"], max_attempts=2,
        manifest_sha256="1" * 64, config_sha256="2" * 64
    )
    with pytest.raises(ContractError, match="different manifest_sha256"):
        ledger.bootstrap(
            [_record()], ["s2"], max_attempts=2,
            manifest_sha256="3" * 64, config_sha256="2" * 64
        )


def test_ledger_binds_sampling_receipt(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    kwargs = {
        "manifest_sha256": "1" * 64,
        "config_sha256": "2" * 64,
        "sampling_receipt_sha256": "3" * 64,
    }
    assert ledger.bootstrap([_record()], ["s2"], max_attempts=2, **kwargs) == 1
    with pytest.raises(ContractError, match="sampling_receipt_sha256"):
        ledger.bootstrap(
            [_record()],
            ["s2"],
            max_attempts=2,
            manifest_sha256="1" * 64,
            config_sha256="2" * 64,
            sampling_receipt_sha256="4" * 64,
        )


def test_deleted_task_breaks_bound_completion_invariant(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    ledger = AcquisitionLedger(path)
    ledger.bootstrap(
        [_record()], ["s2"], max_attempts=2,
        manifest_sha256="1" * 64, config_sha256="2" * 64
    )
    with sqlite3.connect(path) as connection:
        connection.execute("DELETE FROM tasks")
    with pytest.raises(IncompleteAcquisitionError):
        ledger.assert_complete()
