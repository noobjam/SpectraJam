import json
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


def _record_for(candidate_id: str, year: int = 2023) -> PointYear:
    return PointYear(
        sample_id=stable_sample_id("RWA", candidate_id, year),
        candidate_id=candidate_id,
        country="RWA",
        longitude=30.0,
        latitude=-2.0,
        spatial_block="block-1",
        stratum="forest",
        inclusion_probability=0.5,
        spatial_split="train",
        year_split="train",
        year=year,
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


def test_claim_group_uses_the_first_due_country_year_modality(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    ledger = AcquisitionLedger(path)
    records = [
        _record_for("a"),
        _record_for("b"),
        _record_for("c"),
        _record_for("d", 2024),
    ]
    ledger.bootstrap(
        records,
        ["s1", "s2"],
        max_attempts=2,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    with sqlite3.connect(path) as connection:
        expected = connection.execute(
            "SELECT country, year, modality FROM tasks ORDER BY next_attempt_at, task_id LIMIT 1"
        ).fetchone()
        group_size = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE country=? AND year=? AND modality=?",
            expected,
        ).fetchone()[0]

    claimed = ledger.claim_group("worker-a", limit=2, lease_seconds=30, now=10)

    assert len(claimed) == min(2, group_size)
    assert {(task.country, task.year, task.modality) for task in claimed} == {expected}
    assert all(task.attempts == 1 for task in claimed)


def test_renew_leases_updates_a_group_atomically(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    ledger = AcquisitionLedger(path)
    ledger.bootstrap(
        [_record_for("a"), _record_for("b")],
        ["s2"],
        max_attempts=2,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    tasks = ledger.claim_group("worker-a", limit=2, lease_seconds=10, now=10)
    ledger.renew_leases(tasks, "worker-a", lease_seconds=30, now=12)

    with sqlite3.connect(path) as connection:
        renewed = connection.execute(
            "SELECT DISTINCT lease_until FROM tasks WHERE status='running'"
        ).fetchall()
    assert renewed == [(42.0,)]

    with pytest.raises(RuntimeError, match="not every task"):
        ledger.renew_leases(tasks, "other-worker", lease_seconds=50, now=13)
    with sqlite3.connect(path) as connection:
        unchanged = connection.execute(
            "SELECT DISTINCT lease_until FROM tasks WHERE status='running'"
        ).fetchall()
    assert unchanged == [(42.0,)]


def test_no_artifact_resolution_and_terminal_outcome_counts(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite"
    ledger = AcquisitionLedger(path)
    ledger.bootstrap(
        [_record_for("complete"), _record_for("empty"), _record_for("failed")],
        ["s2"],
        max_attempts=2,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    tasks = {task.sample_id: task for task in ledger.claim_group("worker-a", 3, 60, now=10)}
    complete = tasks[stable_sample_id("RWA", "complete", 2023)]
    empty = tasks[stable_sample_id("RWA", "empty", 2023)]
    failed = tasks[stable_sample_id("RWA", "failed", 2023)]
    ledger.succeed(
        complete,
        Artifact("memory://complete", "0" * 64, 3, {"ok": True}),
        "worker-a",
    )
    ledger.resolve_without_artifact(
        empty,
        {"catalog_query_sha256": "a" * 64},
        "worker-a",
    )
    ledger.fail(
        failed,
        "bad data",
        "worker-a",
        retryable=False,
        base_delay_seconds=0,
        max_delay_seconds=0,
        now=10,
    )

    assert ledger.outcomes() == {
        "complete": 1,
        "insufficient_valid_observations": 0,
        "no_source_observation": 1,
        "terminal_data_error": 1,
    }
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            "SELECT status, artifact_uri, artifact_sha256, observation_count, artifact_metadata "
            "FROM tasks WHERE task_id=?",
            (empty.task_id,),
        ).fetchone()
    assert row[:4] == ("succeeded", None, None, 0)
    assert json.loads(row[4]) == {
        "catalog_query_sha256": "a" * 64,
        "outcome": "no_source_observation",
    }


def test_seconds_until_due_tracks_retries_and_running_lease_expiry(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [_record()],
        ["s2"],
        max_attempts=2,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
    )
    assert ledger.seconds_until_due(now=10) == 0
    task = ledger.claim_group("worker-a", 1, lease_seconds=5, now=10)[0]
    assert ledger.seconds_until_due(now=12) == 3

    assert (
        ledger.fail(
            task,
            "temporary",
            "worker-a",
            retryable=True,
            base_delay_seconds=0,
            max_delay_seconds=0,
            retry_after_seconds=10,
            now=12,
        )
        == "retry"
    )
    assert ledger.seconds_until_due(now=15) == 7

    retried = ledger.claim_group("worker-a", 1, lease_seconds=5, now=22)[0]
    ledger.resolve_without_artifact(retried, {}, "worker-a")
    assert ledger.seconds_until_due(now=23) is None


def test_ledger_binds_catalog_and_materializer_contracts(tmp_path: Path) -> None:
    ledger = AcquisitionLedger(tmp_path / "state.sqlite")
    ledger.bootstrap(
        [_record()],
        ["s2"],
        max_attempts=2,
        manifest_sha256="1" * 64,
        config_sha256="2" * 64,
        sampling_receipt_sha256="3" * 64,
    )
    ledger.require_binding("1" * 64, "2" * 64, "3" * 64)
    assert ledger.modalities() == ("s2",)

    ledger.bind_metadata("catalog_inventory_sha256", "4" * 64)
    ledger.bind_metadata("materializer_contract_sha256", "5" * 64)
    ledger.bind_metadata("catalog_inventory_sha256", "4" * 64)
    with pytest.raises(ContractError, match="different catalog_inventory_sha256"):
        ledger.bind_metadata("catalog_inventory_sha256", "6" * 64)
