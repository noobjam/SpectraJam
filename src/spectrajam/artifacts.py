from __future__ import annotations

import hashlib
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

_RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}
_CONTENT_RANGE = re.compile(r"bytes (\d+)-(\d+)/(\d+|\*)$")


class ArtifactFetchError(RuntimeError):
    """A pinned artifact could not be downloaded safely."""


class ArtifactIntegrityError(ArtifactFetchError):
    """An artifact failed its expected byte count or SHA-256 check."""


class InvalidDestinationError(ArtifactIntegrityError):
    """The final destination exists but is not the requested artifact."""


class _IncompleteDownloadError(ArtifactFetchError):
    pass


@dataclass(frozen=True, slots=True)
class PinnedArtifact:
    filename: str
    url: str
    revision: str
    expected_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if not self.filename or Path(self.filename).name != self.filename:
            raise ValueError("artifact filename must be a basename")
        if not self.url.startswith("https://"):
            raise ValueError("artifact URL must use HTTPS")
        if self.expected_bytes < 1:
            raise ValueError("artifact expected_bytes must be positive")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", self.sha256):
            raise ValueError("artifact sha256 must contain 64 hexadecimal characters")


TESSERA_V11_MPC_ENCODER = PinnedArtifact(
    filename="tessera_v1_1_mpc_encoder.pt",
    url=(
        "https://huggingface.co/geotessera/TESSERA-V-1.1/resolve/"
        "e037fc62cd196f9e05dde4c4104e1383541b41c5/tessera_v1_1_mpc_encoder.pt"
    ),
    revision="e037fc62cd196f9e05dde4c4104e1383541b41c5",
    expected_bytes=230_891_229,
    sha256="5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3",
)


def _sha256_file(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _integrity_issue(path: Path, artifact: PinnedArtifact, chunk_size: int) -> str | None:
    if not path.is_file():
        return "path is not a regular file"
    actual_bytes = path.stat().st_size
    if actual_bytes != artifact.expected_bytes:
        return f"byte count {actual_bytes} != {artifact.expected_bytes}"
    actual_sha256 = _sha256_file(path, chunk_size)
    if actual_sha256.lower() != artifact.sha256.lower():
        return f"SHA-256 {actual_sha256} != {artifact.sha256.lower()}"
    return None


def _path_exists(path: Path) -> bool:
    return os.path.lexists(path)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as source:
        os.fsync(source.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _install_part(
    part: Path,
    destination: Path,
    artifact: PinnedArtifact,
    chunk_size: int,
) -> Path:
    if _path_exists(destination):
        issue = _integrity_issue(destination, artifact, chunk_size)
        if issue is not None:
            raise InvalidDestinationError(
                f"refusing to replace invalid destination {destination}: {issue}"
            )
        part.unlink(missing_ok=True)
        return destination

    _fsync_file(part)
    os.replace(part, destination)
    _fsync_directory(destination.parent)
    return destination


def _validate_content_range(
    response: requests.Response,
    offset: int,
    artifact: PinnedArtifact,
) -> None:
    value = response.headers.get("Content-Range", "")
    match = _CONTENT_RANGE.fullmatch(value.strip())
    if match is None:
        raise ArtifactFetchError(f"invalid Content-Range for resumed artifact: {value!r}")
    start, end, total = match.groups()
    if int(start) != offset or int(end) < int(start):
        raise ArtifactFetchError(
            f"Content-Range does not resume byte {offset}: {value!r}"
        )
    if total != "*" and int(total) != artifact.expected_bytes:
        raise ArtifactFetchError(
            f"Content-Range total does not match {artifact.expected_bytes}: {value!r}"
        )


def _download_once(
    session: requests.Session,
    artifact: PinnedArtifact,
    part: Path,
    timeout: float | tuple[float, float],
    chunk_size: int,
) -> int:
    offset = part.stat().st_size if part.exists() else 0
    headers = {"Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"

    response = session.get(
        artifact.url,
        headers=headers,
        stream=True,
        timeout=timeout,
        allow_redirects=True,
    )
    try:
        response.raise_for_status()
        if response.status_code not in {200, 206}:
            raise ArtifactFetchError(
                f"unexpected HTTP {response.status_code} for {artifact.filename}"
            )

        if response.status_code == 206:
            _validate_content_range(response, offset, artifact)
            mode = "ab" if offset else "wb"
            initial_bytes = offset
        else:
            # A server may ignore Range and return the whole object. Starting over
            # is safe; appending that response would corrupt the artifact.
            mode = "wb"
            initial_bytes = 0

        with part.open(mode) as output:
            try:
                written = initial_bytes
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    output.write(chunk)
                    written += len(chunk)
                    if written > artifact.expected_bytes:
                        raise ArtifactIntegrityError(
                            f"downloaded more than {artifact.expected_bytes} bytes for "
                            f"{artifact.filename}"
                        )
            finally:
                output.flush()
                os.fsync(output.fileno())
        return offset if response.status_code == 206 else 0
    finally:
        response.close()


def _is_retryable(error: BaseException) -> bool:
    if isinstance(error, _IncompleteDownloadError):
        return True
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
        return error.response.status_code in _RETRYABLE_STATUS_CODES
    return False


def _retry_delay(error: BaseException, attempt: int, base_delay_seconds: float) -> float:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        retry_after = error.response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(0.0, float(retry_after))
            except ValueError:
                pass
    return base_delay_seconds * (2**attempt)


def fetch_verified_artifact(
    artifact: PinnedArtifact,
    destination: str | Path,
    *,
    session: requests.Session | None = None,
    max_attempts: int = 5,
    timeout: float | tuple[float, float] = (10.0, 120.0),
    chunk_size: int = 1024 * 1024,
    base_delay_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Path:
    """Fetch an immutable artifact, resuming a verified ``.part`` when possible.

    A valid final destination is reused without network access. An invalid final
    destination is never replaced; callers must remove it explicitly after
    investigating the mismatch.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if base_delay_seconds < 0:
        raise ValueError("base_delay_seconds cannot be negative")

    target = Path(destination)
    if _path_exists(target):
        issue = _integrity_issue(target, artifact, chunk_size)
        if issue is not None:
            raise InvalidDestinationError(
                f"refusing to replace invalid destination {target}: {issue}"
            )
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(f"{target.name}.part")
    if _path_exists(part):
        if not part.is_file():
            raise ArtifactFetchError(f"partial artifact is not a regular file: {part}")
        if part.stat().st_size == artifact.expected_bytes:
            issue = _integrity_issue(part, artifact, chunk_size)
            if issue is None:
                return _install_part(part, target, artifact, chunk_size)
            part.unlink()
        elif part.stat().st_size > artifact.expected_bytes:
            part.unlink()

    owned_session = session is None
    client = requests.Session() if session is None else session
    last_error: BaseException | None = None
    try:
        for attempt in range(max_attempts):
            resumed_from = part.stat().st_size if part.exists() else 0
            try:
                used_offset = _download_once(
                    client,
                    artifact,
                    part,
                    timeout=timeout,
                    chunk_size=chunk_size,
                )
                issue = _integrity_issue(part, artifact, chunk_size)
                if issue is None:
                    return _install_part(part, target, artifact, chunk_size)

                actual_bytes = part.stat().st_size
                if actual_bytes < artifact.expected_bytes:
                    raise _IncompleteDownloadError(issue)
                if used_offset or resumed_from:
                    # A stale/corrupt prefix can only be detected after hashing the
                    # completed object. Retry once from byte zero rather than keeping
                    # a poisoned resumable file.
                    part.unlink()
                    raise _IncompleteDownloadError(f"resumed artifact failed validation: {issue}")
                raise ArtifactIntegrityError(
                    f"downloaded artifact {artifact.filename} failed validation: {issue}"
                )
            except Exception as error:
                if not _is_retryable(error):
                    if isinstance(error, ArtifactFetchError):
                        raise
                    raise ArtifactFetchError(
                        f"failed to fetch {artifact.filename}: {error}"
                    ) from error
                last_error = error
                if attempt + 1 >= max_attempts:
                    break
                sleep(_retry_delay(error, attempt, base_delay_seconds))
    finally:
        if owned_session:
            client.close()

    raise ArtifactFetchError(
        f"failed to fetch {artifact.filename} after {max_attempts} attempts"
    ) from last_error
