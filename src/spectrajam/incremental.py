from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import nn

from .windows import TemporalSeries, TemporalWindow, slice_series


class HashWriter(Protocol):
    def update(self, value: bytes) -> None: ...


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{name} must be a 64-character hexadecimal SHA-256")


@dataclass(frozen=True, slots=True)
class EncoderIdentity:
    """Everything outside the input tensors that can change an embedding."""

    model_sha256: str
    preprocessing_sha256: str
    provider_profile: str
    adapter_sha256: str | None = None
    runtime_policy: str = "eval-fp32-deterministic-v1"

    def __post_init__(self) -> None:
        _require_digest(self.model_sha256, "model_sha256")
        _require_digest(self.preprocessing_sha256, "preprocessing_sha256")
        if self.adapter_sha256 is not None:
            _require_digest(self.adapter_sha256, "adapter_sha256")
        if not self.provider_profile or not self.runtime_policy:
            raise ValueError("provider_profile and runtime_policy are required")

    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "adapter_sha256": self.adapter_sha256,
                "model_sha256": self.model_sha256,
                "preprocessing_sha256": self.preprocessing_sha256,
                "provider_profile": self.provider_profile,
                "runtime_policy": self.runtime_policy,
                "schema": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _update_tensor(digest: HashWriter, name: str, value: torch.Tensor) -> None:
    cpu = value.detach().contiguous().cpu()
    digest.update(name.encode())
    digest.update(str(cpu.dtype).encode())
    digest.update(json.dumps(list(cpu.shape)).encode())
    digest.update(cpu.view(torch.uint8).numpy().tobytes())


def _update_series(digest: HashWriter, name: str, series: TemporalSeries) -> None:
    digest.update(name.encode())
    _update_tensor(digest, "bands", series.bands)
    _update_tensor(digest, "observation_day", series.observation_day)
    _update_tensor(digest, "day_of_year", series.day_of_year)
    _update_tensor(digest, "valid", series.valid)
    digest.update(json.dumps(series.observation_ids, separators=(",", ":")).encode())


@dataclass(frozen=True, slots=True)
class PreparedWindow:
    window: TemporalWindow
    s2: TemporalSeries
    s1: TemporalSeries

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"spectrajam-prepared-window-v1")
        digest.update(
            json.dumps(
                {
                    "start_day": self.window.start_day,
                    "end_day": self.window.end_day,
                    "mode": self.window.mode,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        )
        _update_series(digest, "s2", self.s2)
        _update_series(digest, "s1", self.s1)
        return digest.hexdigest()


class WindowEncoder(Protocol):
    def encode(self, prepared: PreparedWindow) -> torch.Tensor: ...


class EmbeddingCache(Protocol):
    def get(self, key: str) -> torch.Tensor | None: ...

    def put(self, key: str, value: torch.Tensor) -> None: ...


class MemoryEmbeddingCache:
    """Small-process cache; the cache interface allows a later durable backend."""

    def __init__(self):
        self._values: dict[str, torch.Tensor] = {}

    def get(self, key: str) -> torch.Tensor | None:
        value = self._values.get(key)
        return None if value is None else value.clone()

    def put(self, key: str, value: torch.Tensor) -> None:
        self._values[key] = value.detach().cpu().clone()

    def __len__(self) -> int:
        return len(self._values)


class TorchWindowEncoder:
    """Run a student or mask-aware TESSERA wrapper on one prepared window."""

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.requires_grad_(False).eval()
        self._parameter_versions = {
            name: parameter._version for name, parameter in self.model.named_parameters()
        }

    def assert_immutable(self) -> None:
        current = {
            name: parameter._version for name, parameter in self.model.named_parameters()
        }
        if current != self._parameter_versions:
            raise RuntimeError(
                "encoder weights changed; create a new encoder identity and builder"
            )

    @staticmethod
    def _batch(
        series: TemporalSeries,
        channels: int,
        window: TemporalWindow,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if series.bands.shape[1] != channels:
            raise ValueError(f"expected {channels} channels, got {series.bands.shape[1]}")
        length = max(1, series.bands.shape[0])
        bands = torch.zeros((1, length, channels), device=device, dtype=dtype)
        doy = series.day_of_year.new_zeros((1, length), device=device)
        valid = torch.zeros((1, length), device=device, dtype=torch.bool)
        relative = series.observation_day.new_zeros((1, length), device=device)
        if series.bands.shape[0]:
            count = series.bands.shape[0]
            bands[0, :count] = series.bands.to(device=device, dtype=dtype)
            doy[0, :count] = series.day_of_year.to(device)
            valid[0, :count] = True
            relative[0, :count] = series.observation_day.to(device) - window.start_day
        return bands, doy, valid, relative

    def encode(self, prepared: PreparedWindow) -> torch.Tensor:
        self.assert_immutable()
        try:
            first_parameter = next(self.model.parameters())
            device = first_parameter.device
            dtype = first_parameter.dtype
        except StopIteration:
            device = prepared.s2.bands.device
            dtype = prepared.s2.bands.dtype
        self.model.eval()
        s2_bands, s2_doy, s2_valid, s2_relative = self._batch(
            prepared.s2, 10, prepared.window, device, dtype
        )
        s1_bands, s1_doy, s1_valid, s1_relative = self._batch(
            prepared.s1, 2, prepared.window, device, dtype
        )
        with torch.inference_mode():
            embedding = self.model(
                s2_bands,
                s2_doy,
                s2_valid,
                s1_bands,
                s1_doy,
                s1_valid,
                s2_relative_day=s2_relative,
                s1_relative_day=s1_relative,
                window_duration_days=torch.tensor(
                    [prepared.window.duration_days], device=device
                ),
                s2_observation_count=torch.tensor(
                    [len(prepared.s2.observation_ids)], device=device
                ),
                s1_observation_count=torch.tensor(
                    [len(prepared.s1.observation_ids)], device=device
                ),
            )
        if embedding.ndim != 2 or embedding.shape[0] != 1:
            raise ValueError("window encoder must return shape [1, embedding]")
        result = embedding[0].detach().float().cpu()
        if not torch.isfinite(result).all():
            raise ValueError("window encoder produced a non-finite embedding")
        return result


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    embedding: torch.Tensor | None
    cache_hit: bool
    cache_key: str
    prepared_input_sha256: str
    window: TemporalWindow
    s2_observation_count: int
    s1_observation_count: int
    empty_window: bool = False


def merge_series(base: TemporalSeries, update: TemporalSeries) -> TemporalSeries:
    """Upsert observations by stable ID and sort by absolute day then ID."""
    if base.bands.shape[1] != update.bands.shape[1]:
        raise ValueError("cannot merge timelines with different channel counts")
    if base.bands.dtype != update.bands.dtype or base.bands.device != update.bands.device:
        raise ValueError("timeline updates must use the same band dtype and device")
    records: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for series in (base, update):
        for index, identifier in enumerate(series.observation_ids):
            records[identifier] = (
                series.bands[index],
                series.observation_day[index],
                series.day_of_year[index],
                series.valid[index],
            )
    ordered = sorted(
        records,
        key=lambda identifier: (int(records[identifier][1]), identifier),
    )
    if not ordered:
        return base
    return TemporalSeries(
        bands=torch.stack([records[key][0] for key in ordered]),
        observation_day=torch.stack([records[key][1] for key in ordered]),
        day_of_year=torch.stack([records[key][2] for key in ordered]),
        valid=torch.stack([records[key][3] for key in ordered]).bool(),
        observation_ids=tuple(ordered),
    )


class ExactIncrementalEmbeddingBuilder:
    """Cache exact window recomputations; never reuse invalid Transformer state."""

    def __init__(
        self,
        encoder: WindowEncoder,
        identity: EncoderIdentity,
        s2: TemporalSeries,
        s1: TemporalSeries,
        cache: EmbeddingCache | None = None,
    ):
        if s2.bands.shape[1] != 10 or s1.bands.shape[1] != 2:
            raise ValueError("incremental builder requires canonical S2=10 and S1=2 channels")
        self.encoder = encoder
        self.identity = identity
        self.s2 = s2
        self.s1 = s1
        self.cache = cache if cache is not None else MemoryEmbeddingCache()

    def upsert(self, modality: str, observations: TemporalSeries) -> None:
        if modality == "s2":
            self.s2 = merge_series(self.s2, observations)
        elif modality == "s1":
            self.s1 = merge_series(self.s1, observations)
        else:
            raise ValueError("modality must be s1 or s2")

    def prepare(self, window: TemporalWindow) -> PreparedWindow:
        return PreparedWindow(
            window=window,
            s2=slice_series(self.s2, window),
            s1=slice_series(self.s1, window),
        )

    def embed(self, window: TemporalWindow) -> EmbeddingResult:
        immutability_check = getattr(self.encoder, "assert_immutable", None)
        if immutability_check is not None:
            immutability_check()
        prepared = self.prepare(window)
        prepared_sha256 = prepared.fingerprint()
        key = hashlib.sha256(
            f"{self.identity.fingerprint()}:{prepared_sha256}".encode()
        ).hexdigest()
        if not prepared.s2.observation_ids and not prepared.s1.observation_ids:
            return EmbeddingResult(None, False, key, prepared_sha256, window, 0, 0, True)
        s2_count = len(prepared.s2.observation_ids)
        s1_count = len(prepared.s1.observation_ids)
        cached = self.cache.get(key)
        if cached is not None:
            if cached.ndim != 1 or not torch.isfinite(cached).all():
                raise ValueError("embedding cache returned a corrupt value")
            return EmbeddingResult(
                cached, True, key, prepared_sha256, window, s2_count, s1_count
            )
        embedding = self.encoder.encode(prepared)
        if embedding.ndim != 1 or not torch.isfinite(embedding).all():
            raise ValueError("window encoder must return one finite embedding vector")
        embedding = embedding.detach().float().cpu().clone()
        self.cache.put(key, embedding)
        return EmbeddingResult(
            embedding, False, key, prepared_sha256, window, s2_count, s1_count
        )

    def embed_many(
        self, windows: tuple[TemporalWindow, ...] | list[TemporalWindow]
    ) -> tuple[EmbeddingResult, ...]:
        return tuple(self.embed(window) for window in windows)
