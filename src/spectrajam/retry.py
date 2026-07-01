from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Protocol

import requests

from .ledger import AcquisitionLedger, Artifact, Task

RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class Worker(Protocol):
    def __call__(self, task: Task) -> Artifact: ...


@dataclass(frozen=True, slots=True)
class RunResult:
    claimed: int
    succeeded: int
    requeued: int
    failed: int


def is_retryable(error: BaseException) -> bool:
    if isinstance(
        error,
        (
            requests.Timeout,
            requests.ConnectionError,
            requests.exceptions.ChunkedEncodingError,
        ),
    ):
        return True
    if isinstance(error, requests.HTTPError) and error.response is not None:
        return error.response.status_code in RETRYABLE_STATUS_CODES
    return False


def retry_after_seconds(error: BaseException) -> float | None:
    if not isinstance(error, requests.HTTPError) or error.response is None:
        return None
    value = error.response.headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - time.time())
        except (TypeError, ValueError, OverflowError):
            return None


def run_due_tasks(
    ledger: AcquisitionLedger,
    worker: Worker,
    worker_id: str,
    limit: int,
    lease_seconds: int,
    base_delay_seconds: float,
    max_delay_seconds: float,
    classify: Callable[[BaseException], bool] = is_retryable,
) -> RunResult:
    """Run only currently due tasks; future retries remain durable in SQLite."""
    tasks = ledger.claim(worker_id, limit=limit, lease_seconds=lease_seconds)
    succeeded = requeued = failed = 0
    for task in tasks:
        try:
            artifact = worker(task)
            ledger.succeed(task, artifact, worker_id)
            succeeded += 1
        except Exception as error:
            status = ledger.fail(
                task,
                error,
                worker_id,
                retryable=classify(error),
                base_delay_seconds=base_delay_seconds,
                max_delay_seconds=max_delay_seconds,
                retry_after_seconds=retry_after_seconds(error),
            )
            if status == "retry":
                requeued += 1
            else:
                failed += 1
    return RunResult(len(tasks), succeeded, requeued, failed)
