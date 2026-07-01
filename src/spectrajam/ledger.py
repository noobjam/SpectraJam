from __future__ import annotations

import hashlib
import json
import random
import sqlite3
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .contracts import ContractError, PointYear, sha256_file, stable_sample_id

TERMINAL_STATES = {"succeeded", "failed"}
ACTIVE_STATES = {"pending", "retry", "running"}


class IncompleteAcquisitionError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class Task:
    task_id: str
    sample_id: str
    country: str
    longitude: float
    latitude: float
    spatial_block: str
    year: int
    modality: str
    attempts: int
    max_attempts: int


@dataclass(frozen=True, slots=True)
class Artifact:
    uri: str
    sha256: str
    observation_count: int
    metadata: dict[str, object]


def task_id(sample_id: str, modality: str) -> str:
    digest = hashlib.sha256(f"spectrajam:task:v1:{sample_id}:{modality}".encode())
    return digest.hexdigest()[:28]


class AcquisitionLedger:
    """Durable, concurrency-safe accounting for every expected point-year-modality."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    sample_id TEXT NOT NULL,
                    candidate_id TEXT NOT NULL,
                    country TEXT NOT NULL,
                    longitude REAL NOT NULL,
                    latitude REAL NOT NULL,
                    spatial_block TEXT NOT NULL,
                    inclusion_probability REAL NOT NULL,
                    spatial_split TEXT NOT NULL,
                    year_split TEXT NOT NULL,
                    year INTEGER NOT NULL,
                    modality TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending'
                        CHECK(status IN ('pending', 'retry', 'running', 'succeeded', 'failed')),
                    attempts INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    next_attempt_at REAL NOT NULL DEFAULT 0,
                    lease_owner TEXT,
                    lease_until REAL,
                    last_error TEXT,
                    artifact_uri TEXT,
                    artifact_sha256 TEXT,
                    observation_count INTEGER,
                    artifact_metadata TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(sample_id, modality)
                );
                CREATE INDEX IF NOT EXISTS tasks_claim_idx
                    ON tasks(status, next_attempt_at, lease_until);
                CREATE INDEX IF NOT EXISTS tasks_sample_idx
                    ON tasks(country, year, modality, status);
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )

    def bootstrap(
        self,
        records: Iterable[PointYear],
        modalities: Iterable[str],
        max_attempts: int,
        manifest_sha256: str,
        config_sha256: str,
        chunk_size: int = 5000,
    ) -> int:
        if max_attempts < 1:
            raise ContractError("max_attempts must be at least 1")
        modalities_tuple = tuple(sorted(set(modalities)))
        if not modalities_tuple or any(item not in {"s1", "s2"} for item in modalities_tuple):
            raise ContractError("modalities must contain s1 and/or s2")

        if len(manifest_sha256) != 64 or len(config_sha256) != 64:
            raise ContractError("manifest and config SHA-256 digests are required")
        if chunk_size < 1:
            raise ContractError("chunk_size must be positive")

        signature = {
            "manifest_sha256": manifest_sha256.lower(),
            "config_sha256": config_sha256.lower(),
            "modalities": json.dumps(modalities_tuple),
            "max_attempts": str(max_attempts),
        }
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            existing = {
                row["key"]: row["value"]
                for row in connection.execute("SELECT key, value FROM metadata")
            }
            if existing:
                for key, value in signature.items():
                    if existing.get(key) != value:
                        raise ContractError(
                            f"ledger is bound to different {key}: "
                            f"{existing.get(key)!r} != {value!r}"
                        )
                expected = int(existing.get("expected_task_count", "0"))
                actual = int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
                if expected < 1 or actual != expected:
                    raise ContractError(
                        "ledger task count does not match its manifest binding: "
                        f"{actual} != {expected}"
                    )
                connection.execute("COMMIT")
                return 0
            if connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]:
                raise ContractError("non-empty ledger is missing manifest metadata")

            connection.execute("CREATE TEMP TABLE input_ids (sample_id TEXT PRIMARY KEY)")
            now = time.time()
            inserted = 0
            rows: list[tuple[object, ...]] = []
            for record in records:
                expected_id = stable_sample_id(record.country, record.candidate_id, record.year)
                if record.sample_id != expected_id:
                    raise ContractError(
                        f"sample_id mismatch for {record.candidate_id}/{record.year}"
                    )
                try:
                    connection.execute(
                        "INSERT INTO input_ids(sample_id) VALUES (?)", (record.sample_id,)
                    )
                except sqlite3.IntegrityError as error:
                    raise ContractError(
                        f"duplicate sample_id in manifest: {record.sample_id}"
                    ) from error
                for modality in modalities_tuple:
                    rows.append(
                        (
                            task_id(record.sample_id, modality),
                            record.sample_id,
                            record.candidate_id,
                            record.country,
                            record.longitude,
                            record.latitude,
                            record.spatial_block,
                            record.inclusion_probability,
                            record.spatial_split,
                            record.year_split,
                            record.year,
                            modality,
                            max_attempts,
                            now,
                            now,
                        )
                    )
                if len(rows) >= chunk_size:
                    self._insert_rows(connection, rows)
                    inserted += len(rows)
                    rows.clear()
            if rows:
                self._insert_rows(connection, rows)
                inserted += len(rows)
            if inserted < 1:
                raise ContractError(
                    "cannot initialize an acquisition ledger from an empty manifest"
                )
            for key, value in signature.items():
                connection.execute("INSERT INTO metadata(key, value) VALUES (?, ?)", (key, value))
            connection.execute(
                "INSERT INTO metadata(key, value) VALUES ('expected_task_count', ?)",
                (str(inserted),),
            )
            connection.execute("COMMIT")
            return inserted
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _insert_rows(connection: sqlite3.Connection, rows: list[tuple[object, ...]]) -> None:
        connection.executemany(
            """
            INSERT INTO tasks (
                task_id, sample_id, candidate_id, country, longitude, latitude,
                spatial_block, inclusion_probability, spatial_split, year_split, year, modality,
                max_attempts, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    def recover_expired_leases(self, now: float | None = None) -> int:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            failed = connection.execute(
                """
                UPDATE tasks
                SET status='failed', lease_owner=NULL, lease_until=NULL,
                    last_error='worker lease expired after final attempt', updated_at=?
                WHERE status='running' AND lease_until < ? AND attempts >= max_attempts
                """,
                (timestamp, timestamp),
            ).rowcount
            retried = connection.execute(
                """
                UPDATE tasks
                SET status='retry', lease_owner=NULL, lease_until=NULL,
                    last_error=COALESCE(last_error, 'worker lease expired'), updated_at=?
                WHERE status='running' AND lease_until < ? AND attempts < max_attempts
                """,
                (timestamp, timestamp),
            ).rowcount
            return failed + retried

    def claim(
        self,
        worker_id: str,
        limit: int,
        lease_seconds: int,
        now: float | None = None,
    ) -> list[Task]:
        if not worker_id:
            raise ContractError("worker_id is required")
        if limit < 1 or lease_seconds < 1:
            raise ContractError("limit and lease_seconds must be positive")
        timestamp = time.time() if now is None else now
        lease_until = timestamp + lease_seconds

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE tasks
                SET status='failed', lease_owner=NULL, lease_until=NULL,
                    last_error='worker lease expired after final attempt', updated_at=?
                WHERE status='running' AND lease_until < ? AND attempts >= max_attempts
                """,
                (timestamp, timestamp),
            )
            connection.execute(
                """
                UPDATE tasks
                SET status='retry', lease_owner=NULL, lease_until=NULL,
                    last_error=COALESCE(last_error, 'worker lease expired'), updated_at=?
                WHERE status='running' AND lease_until < ? AND attempts < max_attempts
                """,
                (timestamp, timestamp),
            )
            rows = connection.execute(
                """
                SELECT * FROM tasks
                WHERE status IN ('pending', 'retry')
                  AND next_attempt_at <= ?
                  AND attempts < max_attempts
                ORDER BY next_attempt_at, task_id
                LIMIT ?
                """,
                (timestamp, limit),
            ).fetchall()
            ids = [row["task_id"] for row in rows]
            if ids:
                placeholders = ",".join("?" for _ in ids)
                connection.execute(
                    f"""
                    UPDATE tasks
                    SET status='running', attempts=attempts+1, lease_owner=?,
                        lease_until=?, updated_at=?
                    WHERE task_id IN ({placeholders})
                    """,
                    (worker_id, lease_until, timestamp, *ids),
                )
                rows = connection.execute(
                    f"SELECT * FROM tasks WHERE task_id IN ({placeholders}) ORDER BY task_id",
                    ids,
                ).fetchall()
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

        return [self._task_from_row(row) for row in rows]

    def renew_lease(
        self, task_id_value: str, worker_id: str, lease_seconds: int, now: float | None = None
    ) -> None:
        timestamp = time.time() if now is None else now
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET lease_until=?, updated_at=?
                WHERE task_id=? AND status='running' AND lease_owner=?
                """,
                (timestamp + lease_seconds, timestamp, task_id_value, worker_id),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"task {task_id_value} is not leased by {worker_id}")

    @staticmethod
    def _task_from_row(row: sqlite3.Row) -> Task:
        return Task(
            task_id=row["task_id"],
            sample_id=row["sample_id"],
            country=row["country"],
            longitude=row["longitude"],
            latitude=row["latitude"],
            spatial_block=row["spatial_block"],
            year=row["year"],
            modality=row["modality"],
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
        )

    def succeed(self, task: Task, artifact: Artifact, worker_id: str) -> None:
        if artifact.observation_count < 1:
            raise ContractError("a successful artifact must contain at least one observation")
        if not artifact.uri:
            raise ContractError("a successful artifact URI is required")
        if len(artifact.sha256) != 64 or any(
            value not in "0123456789abcdef" for value in artifact.sha256.lower()
        ):
            raise ContractError("artifact sha256 must be a hexadecimal digest")
        if "://" not in artifact.uri or artifact.uri.startswith("file://"):
            local_path = Path(
                artifact.uri.removeprefix("file://")
                if artifact.uri.startswith("file://")
                else artifact.uri
            )
            if not local_path.is_file():
                raise ContractError(f"artifact file does not exist: {local_path}")
            actual = sha256_file(local_path)
            if actual != artifact.sha256.lower():
                raise ContractError(
                    f"artifact checksum mismatch: expected {artifact.sha256}, got {actual}"
                )
        now = time.time()
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status='succeeded', artifact_uri=?, artifact_sha256=?,
                    observation_count=?, artifact_metadata=?, lease_owner=NULL,
                    lease_until=NULL, last_error=NULL, updated_at=?
                WHERE task_id=? AND status='running' AND lease_owner=?
                """,
                (
                    artifact.uri,
                    artifact.sha256,
                    artifact.observation_count,
                    json.dumps(artifact.metadata, sort_keys=True),
                    now,
                    task.task_id,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"task {task.task_id} is not leased by {worker_id}")

    def fail(
        self,
        task: Task,
        error: BaseException | str,
        worker_id: str,
        retryable: bool,
        base_delay_seconds: float,
        max_delay_seconds: float,
        retry_after_seconds: float | None = None,
        now: float | None = None,
    ) -> str:
        timestamp = time.time() if now is None else now
        message = str(error)[:4000]
        exhausted = task.attempts >= task.max_attempts
        status = "retry" if retryable and not exhausted else "failed"
        delay = 0.0
        if status == "retry":
            cap = min(max_delay_seconds, base_delay_seconds * (2 ** (task.attempts - 1)))
            delay = random.uniform(0.0, cap)
            if retry_after_seconds is not None:
                delay = max(delay, retry_after_seconds)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status=?, next_attempt_at=?, last_error=?, lease_owner=NULL,
                    lease_until=NULL, updated_at=?
                WHERE task_id=? AND status='running' AND lease_owner=?
                """,
                (
                    status,
                    timestamp + delay,
                    message,
                    timestamp,
                    task.task_id,
                    worker_id,
                ),
            )
            if cursor.rowcount != 1:
                raise RuntimeError(f"task {task.task_id} is not leased by {worker_id}")
        return status

    def summary(self) -> dict[str, int]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks GROUP BY status"
            ).fetchall()
        result = {state: 0 for state in (*ACTIVE_STATES, *TERMINAL_STATES)}
        result.update({row["status"]: row["count"] for row in rows})
        result["total"] = sum(row["count"] for row in rows)
        return result

    def failures(self, limit: int = 20) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT task_id, sample_id, country, year, modality, attempts, last_error
                FROM tasks WHERE status='failed' ORDER BY updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def require_binding(self, manifest_sha256: str, config_sha256: str) -> None:
        with self._connect() as connection:
            values = {
                row["key"]: row["value"]
                for row in connection.execute(
                    "SELECT key, value FROM metadata "
                    "WHERE key IN ('manifest_sha256', 'config_sha256')"
                )
            }
        expected = {
            "manifest_sha256": manifest_sha256.lower(),
            "config_sha256": config_sha256.lower(),
        }
        if values != expected:
            raise ContractError(f"ledger binding mismatch: expected {expected}, got {values}")

    def assert_complete(self) -> None:
        summary = self.summary()
        if summary["total"] == 0:
            raise IncompleteAcquisitionError("acquisition ledger is empty")
        with self._connect() as connection:
            row = connection.execute(
                "SELECT value FROM metadata WHERE key='expected_task_count'"
            ).fetchone()
        if row is None or summary["total"] != int(row["value"]):
            raise IncompleteAcquisitionError(
                f"ledger task count {summary['total']} does not match its bound expected count"
            )
        incomplete = summary["total"] - summary["succeeded"]
        if incomplete:
            raise IncompleteAcquisitionError(
                f"acquisition is incomplete: {summary}; recent failures={self.failures(5)}"
            )

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0])
