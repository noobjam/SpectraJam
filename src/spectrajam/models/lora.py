from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import parametrize

from ..contracts import (
    CANONICAL_S1_BANDS,
    CANONICAL_S2_BANDS,
    TESSERA_UPSTREAM_COMMIT,
    ContractError,
)


@dataclass(frozen=True, slots=True)
class LoRAConfig:
    rank: int = 8
    alpha: float = 8.0
    target: str = "attention"
    adapt_input_projection: bool = False

    def validate(self) -> None:
        if self.rank < 1:
            raise ValueError("LoRA rank must be positive")
        if self.alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        if self.target not in {"attention", "attention_ffn"}:
            raise ValueError("LoRA target must be attention or attention_ffn")


class LinearLoRA(nn.Module):
    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.lora_A = nn.Parameter(
            torch.empty(rank, in_features, device=device, dtype=dtype)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_features, rank, device=device, dtype=dtype)
        )
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        return original + (self.lora_B @ self.lora_A) * self.scale


class PackedQKVLoRA(nn.Module):
    """Apply LoRA to Q and V rows of PyTorch's packed in-projection weight."""

    def __init__(
        self,
        embed_dim: int,
        rank: int,
        alpha: float,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.q_A = nn.Parameter(torch.empty(rank, embed_dim, device=device, dtype=dtype))
        self.q_B = nn.Parameter(torch.zeros(embed_dim, rank, device=device, dtype=dtype))
        self.v_A = nn.Parameter(torch.empty(rank, embed_dim, device=device, dtype=dtype))
        self.v_B = nn.Parameter(torch.zeros(embed_dim, rank, device=device, dtype=dtype))
        self.scale = alpha / rank
        nn.init.kaiming_uniform_(self.q_A, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.v_A, a=math.sqrt(5))

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        if original.shape[0] != original.shape[1] * 3:
            raise ValueError(f"expected packed QKV weight, got {tuple(original.shape)}")
        delta_q = (self.q_B @ self.q_A) * self.scale
        delta_v = (self.v_B @ self.v_A) * self.scale
        return original + torch.cat([delta_q, torch.zeros_like(delta_q), delta_v], dim=0)


def _register_linear(module: nn.Linear, config: LoRAConfig) -> None:
    if parametrize.is_parametrized(module, "weight"):
        return
    parametrize.register_parametrization(
        module,
        "weight",
        LinearLoRA(
            module.out_features,
            module.in_features,
            config.rank,
            config.alpha,
            device=module.weight.device,
            dtype=module.weight.dtype,
        ),
    )


def inject_lora(model: nn.Module, config: LoRAConfig) -> list[str]:
    """Freeze the base and install zero-output LoRA adapters in-place."""
    config.validate()
    model.requires_grad_(False)
    installed: list[str] = []
    modules = list(model.named_modules())

    for name, module in modules:
        if isinstance(module, nn.MultiheadAttention):
            if module.in_proj_weight is None:
                raise ValueError(f"{name} uses separate Q/K/V weights; unsupported by this adapter")
            if not parametrize.is_parametrized(module, "in_proj_weight"):
                parametrize.register_parametrization(
                    module,
                    "in_proj_weight",
                    PackedQKVLoRA(
                        module.embed_dim,
                        config.rank,
                        config.alpha,
                        device=module.in_proj_weight.device,
                        dtype=module.in_proj_weight.dtype,
                    ),
                )
                installed.append(f"{name}.in_proj_weight[q,v]")

    for name, module in modules:
        if not isinstance(module, nn.Linear):
            continue
        is_attention_output = name.endswith("self_attn.out_proj")
        is_ffn = name.endswith("linear1") or name.endswith("linear2")
        is_input = ".embedding.0" in name or ".embedding.2" in name
        is_reducer = name.endswith("dim_reducer.0") or name.endswith("dim_reducer.4")
        should_adapt = is_attention_output
        should_adapt |= config.target == "attention_ffn" and (is_ffn or is_reducer)
        should_adapt |= config.adapt_input_projection and is_input
        if should_adapt:
            _register_linear(module, config)
            installed.append(f"{name}.weight")

    if not installed:
        raise ValueError("no LoRA targets matched the model")
    for name, parameter in model.named_parameters():
        if "window_conditioner." in name:
            parameter.requires_grad_(True)
    return installed


def trainable_parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def installed_lora_config(model: nn.Module) -> LoRAConfig:
    adapters = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, (LinearLoRA, PackedQKVLoRA))
    ]
    if not adapters:
        raise ContractError("model has no installed LoRA parametrizations")
    ranks = {
        int(module.lora_A.shape[0])
        if isinstance(module, LinearLoRA)
        else int(module.q_A.shape[0])
        for _, module in adapters
    }
    scales = {float(module.scale) for _, module in adapters}
    if len(ranks) != 1 or len(scales) != 1:
        raise ContractError("installed LoRA parametrizations use inconsistent rank or scale")
    rank = ranks.pop()
    alpha = scales.pop() * rank
    names = tuple(name for name, _ in adapters)
    target = (
        "attention_ffn"
        if any(
            fragment in name
            for name in names
            for fragment in ("linear1", "linear2", "dim_reducer")
        )
        else "attention"
    )
    adapt_input = any(".embedding." in name for name in names)
    return LoRAConfig(rank, alpha, target, adapt_input)


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if (
            "parametrizations." in key and not key.endswith(".original")
        )
        or "window_conditioner." in key
    }


def merge_lora(model: nn.Module) -> None:
    for _, module in list(model.named_modules()):
        for parameter_name in ("in_proj_weight", "weight"):
            if parametrize.is_parametrized(module, parameter_name):
                parametrize.remove_parametrizations(
                    module, parameter_name, leave_parametrized=True
                )


def _remove_lora_unmerged(model: nn.Module) -> None:
    for _, module in list(model.named_modules()):
        for parameter_name in ("in_proj_weight", "weight"):
            if parametrize.is_parametrized(module, parameter_name):
                parametrize.remove_parametrizations(
                    module, parameter_name, leave_parametrized=False
                )


def save_adapter(
    path: str | Path,
    model: nn.Module,
    config: LoRAConfig,
    base_checkpoint_sha256: str,
    provider_profile: str,
    country: str,
) -> None:
    config.validate()
    installed = installed_lora_config(model)
    if (
        installed.rank != config.rank
        or not math.isclose(installed.alpha, config.alpha, rel_tol=0, abs_tol=1e-12)
        or installed.target != config.target
        or installed.adapt_input_projection != config.adapt_input_projection
    ):
        raise ContractError(
            f"adapter metadata config {config} does not match installed {installed}"
        )
    if len(base_checkpoint_sha256) != 64:
        raise ContractError("adapter requires the base checkpoint SHA-256")
    if provider_profile != "mpc-v1.1":
        raise ContractError("the initial adapters require provider_profile=mpc-v1.1")
    if country not in {"RWA", "ISR", "RWA+ISR"}:
        raise ContractError(f"invalid adapter country scope: {country}")
    state = adapter_state_dict(model)
    window_conditioned = any("window_conditioner." in key for key in state)
    payload = {
        "schema_version": 2,
        "metadata": {
            "rank": config.rank,
            "alpha": config.alpha,
            "target": config.target,
            "adapt_input_projection": config.adapt_input_projection,
            "base_checkpoint_sha256": base_checkpoint_sha256.lower(),
            "provider_profile": provider_profile,
            "country": country,
            "upstream_commit": TESSERA_UPSTREAM_COMMIT,
            "s2_band_order": CANONICAL_S2_BANDS,
            "s1_band_order": CANONICAL_S1_BANDS,
            "window_conditioned": window_conditioned,
            "window_conditioner_schema": (
                "duration-counts-missing-v1" if window_conditioned else None
            ),
        },
        "adapter_state": state,
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    torch.save(payload, temporary)
    temporary.replace(destination)


def load_adapter(
    path: str | Path,
    model: nn.Module,
    expected_base_checkpoint_sha256: str,
    expected_provider_profile: str,
    expected_country: str,
) -> LoRAConfig:
    for name, module in model.named_modules():
        if parametrize.is_parametrized(module):
            raise ContractError(
                f"adapter loading requires a pristine base model; {name or '<root>'} "
                "is already parametrized"
            )
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") not in {1, 2}:
        raise ContractError("unsupported LoRA adapter schema")
    metadata = payload.get("metadata")
    state = payload.get("adapter_state")
    if not isinstance(metadata, dict) or not isinstance(state, dict):
        raise ContractError("adapter is missing metadata or weights")
    expected_metadata = {
        "base_checkpoint_sha256": expected_base_checkpoint_sha256.lower(),
        "provider_profile": expected_provider_profile,
        "country": expected_country,
        "upstream_commit": TESSERA_UPSTREAM_COMMIT,
        "s2_band_order": tuple(CANONICAL_S2_BANDS),
        "s1_band_order": tuple(CANONICAL_S1_BANDS),
    }
    for key, expected in expected_metadata.items():
        actual = metadata.get(key)
        if isinstance(actual, list):
            actual = tuple(actual)
        if actual != expected:
            raise ContractError(f"adapter {key} mismatch: expected {expected!r}, got {actual!r}")
    target_window_conditioned = any(
        "window_conditioner" in name for name, _ in model.named_modules()
    )
    saved_window_conditioned = bool(metadata.get("window_conditioned", False))
    if target_window_conditioned != saved_window_conditioned:
        raise ContractError(
            "adapter window-conditioning mismatch: "
            f"expected {target_window_conditioned}, got {saved_window_conditioned}"
        )
    expected_conditioner_schema = (
        "duration-counts-missing-v1" if target_window_conditioned else None
    )
    if metadata.get("window_conditioner_schema") != expected_conditioner_schema:
        raise ContractError("adapter window-conditioner schema mismatch")
    config = LoRAConfig(
        rank=int(metadata["rank"]),
        alpha=float(metadata["alpha"]),
        target=str(metadata["target"]),
        adapt_input_projection=bool(metadata["adapt_input_projection"]),
    )
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    try:
        inject_lora(model, config)
        expected_keys = set(adapter_state_dict(model))
        if set(state) != expected_keys:
            raise ContractError(
                f"adapter parameter mismatch: missing={sorted(expected_keys - set(state))}, "
                f"unexpected={sorted(set(state) - expected_keys)}"
            )
        current = model.state_dict()
        for key, value in state.items():
            if current[key].shape != value.shape:
                raise ContractError(
                    f"adapter tensor shape mismatch for {key}: "
                    f"{value.shape} != {current[key].shape}"
                )
            if not torch.isfinite(value).all():
                raise ContractError(f"adapter tensor contains non-finite values: {key}")
        model.load_state_dict(state, strict=False)
    except Exception:
        _remove_lora_unmerged(model)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad.get(name, False))
        raise
    return config
