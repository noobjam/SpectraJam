import hashlib
from pathlib import Path

import pytest
import requests

from spectrajam.artifacts import (
    TESSERA_V11_MPC_ENCODER,
    ArtifactFetchError,
    ArtifactIntegrityError,
    InvalidDestinationError,
    PinnedArtifact,
    fetch_verified_artifact,
)


class FakeResponse:
    def __init__(self, status_code, chunks=(), headers=None):
        self.status_code = status_code
        self.chunks = chunks
        self.headers = headers or {}
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)

    def iter_content(self, chunk_size):
        del chunk_size
        for chunk in self.chunks:
            if isinstance(chunk, BaseException):
                raise chunk
            yield chunk

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self, outcomes=()):
        self.outcomes = list(outcomes)
        self.calls = []
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def close(self):
        self.closed = True


def _artifact(payload: bytes) -> PinnedArtifact:
    return PinnedArtifact(
        filename="fixture.bin",
        url="https://example.test/fixture.bin",
        revision="test-revision",
        expected_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
    )


def test_tessera_mpc_encoder_is_revision_and_digest_pinned() -> None:
    artifact = TESSERA_V11_MPC_ENCODER
    assert artifact.filename == "tessera_v1_1_mpc_encoder.pt"
    assert artifact.revision == "e037fc62cd196f9e05dde4c4104e1383541b41c5"
    assert artifact.revision in artifact.url
    assert artifact.expected_bytes == 230_891_229
    assert artifact.sha256 == "5dab0f070d5711034f7c241e841eaeedb49fef90b9355f68c8f20b9507839ec3"


def test_streams_valid_artifact_then_atomically_installs_it(tmp_path: Path) -> None:
    payload = b"abcdefgh"
    response = FakeResponse(200, [b"ab", b"cdef", b"gh"])
    session = FakeSession([response])
    destination = tmp_path / "nested" / "fixture.bin"

    result = fetch_verified_artifact(
        _artifact(payload), destination, session=session, chunk_size=2
    )

    assert result == destination
    assert destination.read_bytes() == payload
    assert not (tmp_path / "nested" / "fixture.bin.part").exists()
    assert response.closed
    assert session.calls[0][1]["stream"] is True
    assert session.calls[0][1]["headers"] == {"Accept-Encoding": "identity"}


def test_valid_destination_is_reused_without_network_access(tmp_path: Path) -> None:
    payload = b"already complete"
    destination = tmp_path / "fixture.bin"
    destination.write_bytes(payload)
    session = FakeSession()

    assert fetch_verified_artifact(_artifact(payload), destination, session=session) == destination
    assert session.calls == []


def test_invalid_destination_is_never_silently_replaced(tmp_path: Path) -> None:
    payload = b"expected"
    destination = tmp_path / "fixture.bin"
    destination.write_bytes(b"wrong")
    session = FakeSession([FakeResponse(200, [payload])])

    with pytest.raises(InvalidDestinationError, match="refusing to replace"):
        fetch_verified_artifact(_artifact(payload), destination, session=session)

    assert destination.read_bytes() == b"wrong"
    assert session.calls == []


def test_partial_file_resumes_when_server_honors_range(tmp_path: Path) -> None:
    payload = b"0123456789"
    destination = tmp_path / "fixture.bin"
    destination.with_name("fixture.bin.part").write_bytes(payload[:4])
    response = FakeResponse(
        206,
        [payload[4:7], payload[7:]],
        {"Content-Range": f"bytes 4-9/{len(payload)}"},
    )
    session = FakeSession([response])

    fetch_verified_artifact(_artifact(payload), destination, session=session)

    assert destination.read_bytes() == payload
    assert session.calls[0][1]["headers"]["Range"] == "bytes=4-"


def test_partial_file_restarts_when_server_ignores_range(tmp_path: Path) -> None:
    payload = b"complete payload"
    destination = tmp_path / "fixture.bin"
    destination.with_name("fixture.bin.part").write_bytes(payload[:5])
    session = FakeSession([FakeResponse(200, [payload])])

    fetch_verified_artifact(_artifact(payload), destination, session=session)

    assert destination.read_bytes() == payload
    assert session.calls[0][1]["headers"]["Range"] == "bytes=5-"


def test_interrupted_stream_is_fsynced_and_resumed_on_retry(tmp_path: Path) -> None:
    payload = b"abcdefghij"
    destination = tmp_path / "fixture.bin"
    first = FakeResponse(200, [payload[:4], requests.ConnectionError("dropped")])
    second = FakeResponse(
        206,
        [payload[4:]],
        {"Content-Range": f"bytes 4-9/{len(payload)}"},
    )
    session = FakeSession([first, second])
    delays = []

    fetch_verified_artifact(
        _artifact(payload),
        destination,
        session=session,
        max_attempts=2,
        base_delay_seconds=0.25,
        sleep=delays.append,
    )

    assert destination.read_bytes() == payload
    assert delays == [0.25]
    assert session.calls[1][1]["headers"]["Range"] == "bytes=4-"
    assert first.closed and second.closed


def test_retryable_http_status_honors_retry_after(tmp_path: Path) -> None:
    payload = b"ok"
    unavailable = FakeResponse(503, headers={"Retry-After": "3"})
    success = FakeResponse(200, [payload])
    session = FakeSession([unavailable, success])
    delays = []

    fetch_verified_artifact(
        _artifact(payload),
        tmp_path / "fixture.bin",
        session=session,
        max_attempts=2,
        sleep=delays.append,
    )

    assert delays == [3.0]
    assert unavailable.closed and success.closed


def test_non_retryable_http_status_fails_once(tmp_path: Path) -> None:
    response = FakeResponse(404)
    session = FakeSession([response])

    with pytest.raises(ArtifactFetchError):
        fetch_verified_artifact(
            _artifact(b"payload"),
            tmp_path / "fixture.bin",
            session=session,
            max_attempts=3,
            sleep=lambda _delay: pytest.fail("must not retry a 404"),
        )

    assert len(session.calls) == 1


def test_wrong_hash_never_promotes_part_file(tmp_path: Path) -> None:
    payload = b"expected"
    destination = tmp_path / "fixture.bin"
    session = FakeSession([FakeResponse(200, [b"corrupt!"])])

    with pytest.raises(ArtifactIntegrityError, match="SHA-256"):
        fetch_verified_artifact(_artifact(payload), destination, session=session)

    assert not destination.exists()
    assert destination.with_name("fixture.bin.part").read_bytes() == b"corrupt!"


def test_mismatched_content_range_is_rejected(tmp_path: Path) -> None:
    payload = b"0123456789"
    destination = tmp_path / "fixture.bin"
    destination.with_name("fixture.bin.part").write_bytes(payload[:4])
    session = FakeSession(
        [FakeResponse(206, [payload[4:]], {"Content-Range": "bytes 3-9/10"})]
    )

    with pytest.raises(ArtifactFetchError, match="does not resume byte 4"):
        fetch_verified_artifact(_artifact(payload), destination, session=session)

    assert not destination.exists()
    assert destination.with_name("fixture.bin.part").read_bytes() == payload[:4]
