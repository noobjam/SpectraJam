from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

# Checkpoint-critical order used by the official TESSERA preprocessing stack.
CANONICAL_S2_BANDS = (
    "B04",
    "B02",
    "B03",
    "B08",
    "B8A",
    "B05",
    "B06",
    "B07",
    "B11",
    "B12",
)
CANONICAL_S1_BANDS = ("VV", "VH")

TESSERA_UPSTREAM_REPOSITORY = "https://github.com/ucam-eo/tessera.git"
TESSERA_UPSTREAM_COMMIT = "d06ee44a053246db3e73f104403f6eaf642e1abf"


class ContractError(ValueError):
    """Raised when data or model metadata can silently corrupt an experiment."""


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_sha256(path: str | Path, expected: str) -> None:
    if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected.lower()):
        raise ContractError("checkpoint sha256 must be a 64-character hexadecimal digest")
    actual = sha256_file(path)
    if actual.lower() != expected.lower():
        raise ContractError(f"sha256 mismatch for {path}: expected {expected}, got {actual}")


def require_band_order(bands: Iterable[str]) -> None:
    actual = tuple(bands)
    if actual != CANONICAL_S2_BANDS:
        raise ContractError(
            "Sentinel-2 band order is checkpoint-critical; "
            f"expected {CANONICAL_S2_BANDS}, got {actual}"
        )


@dataclass(frozen=True, slots=True)
class PointYear:
    sample_id: str
    candidate_id: str
    country: str
    longitude: float
    latitude: float
    spatial_block: str
    stratum: str
    inclusion_probability: float
    spatial_split: str
    year_split: str
    year: int

    def __post_init__(self) -> None:
        if not self.sample_id:
            raise ContractError("sample_id is required")
        if self.country not in {"RWA", "ISR"}:
            raise ContractError(f"unsupported country code: {self.country}")
        if not math.isfinite(self.longitude) or not -180 <= self.longitude <= 180:
            raise ContractError(f"invalid longitude: {self.longitude}")
        if not math.isfinite(self.latitude) or not -90 <= self.latitude <= 90:
            raise ContractError(f"invalid latitude: {self.latitude}")
        if self.spatial_split not in {"train", "validation", "test"}:
            raise ContractError(f"invalid spatial split: {self.spatial_split}")
        if self.year_split not in {"train", "validation", "test"}:
            raise ContractError(f"invalid year split: {self.year_split}")
        if not 2017 <= self.year <= 2100:
            raise ContractError(f"invalid Sentinel-era year: {self.year}")
        if not 0 < self.inclusion_probability <= 1:
            raise ContractError("inclusion_probability must be in (0, 1]")


def stable_sample_id(country: str, candidate_id: str, year: int) -> str:
    payload = f"spectrajam:v1:{country}:{candidate_id}:{year}".encode()
    return hashlib.sha256(payload).hexdigest()[:24]
