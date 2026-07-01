"""Checkpoint-faithful TESSERA v1.1 encoder assembly.

The model structure is derived from ucam-eo/tessera at commit d06ee44 under
the MIT license. SpectraJam deliberately fails on missing or unexpected keys;
the official inference helper's permissive loading is unsafe for adaptation.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from ..contracts import ContractError, require_sha256


class TemporalPositionalEncoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, day_of_year: torch.Tensor) -> torch.Tensor:
        position = day_of_year.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32, device=day_of_year.device)
            * -(math.log(10000.0) / self.d_model)
        )
        encoding = torch.zeros(
            day_of_year.shape[0],
            day_of_year.shape[1],
            self.d_model,
            device=day_of_year.device,
            dtype=torch.float32,
        )
        encoding[:, :, 0::2] = torch.sin(position * div_term)
        encoding[:, :, 1::2] = torch.cos(position * div_term)
        dtype = day_of_year.dtype if day_of_year.is_floating_point() else torch.float32
        return encoding.to(dtype=dtype)


class CustomGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.W_ir = nn.Linear(input_size, hidden_size, bias=False)
        self.W_iz = nn.Linear(input_size, hidden_size, bias=False)
        self.W_ih = nn.Linear(input_size, hidden_size, bias=False)
        self.W_hr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hz = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_hh = nn.Linear(hidden_size, hidden_size, bias=False)
        self.b_r = nn.Parameter(torch.zeros(hidden_size))
        self.b_z = nn.Parameter(torch.zeros(hidden_size))
        self.b_h = nn.Parameter(torch.zeros(hidden_size))
        for module in (self.W_ir, self.W_iz, self.W_ih, self.W_hr, self.W_hz, self.W_hh):
            nn.init.xavier_uniform_(module.weight)

    def forward(self, value: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        reset = torch.sigmoid(self.W_ir(value) + self.W_hr(previous) + self.b_r)
        update = torch.sigmoid(self.W_iz(value) + self.W_hz(previous) + self.b_z)
        candidate = torch.tanh(self.W_ih(value) + self.W_hh(reset * previous) + self.b_h)
        return (1 - update) * previous + update * candidate


class CustomGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, batch_first: bool = True):
        super().__init__()
        if not batch_first:
            raise ValueError("TESSERA v1.1 requires batch_first=True")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.gru_cell = CustomGRUCell(input_size, hidden_size)

    def forward(
        self, value: torch.Tensor, initial: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, sequence, _ = value.shape
        hidden = initial
        if hidden is None:
            hidden = torch.zeros(batch, self.hidden_size, device=value.device, dtype=value.dtype)
        outputs = []
        for index in range(sequence):
            hidden = self.gru_cell(value[:, index, :], hidden)
            outputs.append(hidden)
        return torch.stack(outputs, dim=1), hidden


class CustomTemporalAwarePooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.temporal_context = CustomGRU(input_dim, input_dim, batch_first=True)
        self.query = nn.Linear(input_dim, 1)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.shape[1] == 0:
            return torch.zeros(
                value.shape[0],
                value.shape[2],
                device=value.device,
                dtype=value.dtype,
            )
        if value.shape[1] == 1:
            return value[:, 0, :]
        context, _ = self.temporal_context(value)
        context = self.layer_norm(context)
        weights = torch.softmax(self.query(context), dim=1)
        return (weights * value).sum(dim=1)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        band_num: int,
        latent_dim: int,
        nhead: int = 4,
        num_encoder_layers: int = 4,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        width = latent_dim * 4
        self.embedding = nn.Sequential(
            nn.Linear(band_num, width),
            nn.ReLU(),
            nn.Linear(width, width),
        )
        self.temporal_encoder = TemporalPositionalEncoder(width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=num_encoder_layers)
        self.attn_pool = CustomTemporalAwarePooling(width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        bands = value[:, :, :-1]
        day_of_year = value[:, :, -1]
        encoded = self.embedding(bands) + self.temporal_encoder(day_of_year)
        encoded = self.transformer_encoder(encoded)
        return self.attn_pool(encoded)


class TesseraV11(nn.Module):
    def __init__(
        self,
        latent_dim: int = 192,
        representation_dim: int = 192,
        output_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 2048,
    ):
        super().__init__()
        self.s2_backbone = TransformerEncoder(
            10, latent_dim, num_heads, num_layers, dim_feedforward
        )
        self.s1_backbone = TransformerEncoder(
            2, latent_dim, num_heads, num_layers, dim_feedforward
        )
        reducer_in = latent_dim * 4 * 2
        self.dim_reducer = nn.Sequential(
            nn.Linear(reducer_in, reducer_in * 2),
            nn.LayerNorm(reducer_in * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(reducer_in * 2, representation_dim),
        )
        if output_dim > representation_dim:
            raise ValueError("output_dim cannot exceed representation_dim")
        self.output_dim = output_dim

    def forward(self, s2: torch.Tensor, s1: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([self.s2_backbone(s2), self.s1_backbone(s1)], dim=-1)
        return self.dim_reducer(fused)[..., : self.output_dim]


def _extract_encoder_state(checkpoint: object) -> dict[str, torch.Tensor]:
    if not isinstance(checkpoint, dict):
        raise ContractError("TESSERA checkpoint must be a mapping")
    raw = checkpoint.get("model_state") or checkpoint.get("model_state_dict")
    if raw is None and checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        raw = checkpoint
    if not isinstance(raw, dict):
        raise ContractError("checkpoint does not contain model_state or model_state_dict")
    cleaned: dict[str, torch.Tensor] = {}
    for original_key, value in raw.items():
        key = str(original_key)
        for prefix in ("_orig_mod.", "module."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key.startswith(("projector.", "segmented_matryoshka_projector.")):
            continue
        cleaned[key] = value
    return cleaned


def load_tessera_v11(
    checkpoint_path: str | Path,
    expected_sha256: str,
    device: str | torch.device = "cpu",
) -> TesseraV11:
    path = Path(checkpoint_path).expanduser()
    require_sha256(path, expected_sha256)
    model = TesseraV11()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = _extract_encoder_state(checkpoint)
    expected = set(model.state_dict())
    actual = set(state)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ContractError(
            "checkpoint is not exactly compatible with the pinned v1.1 graph: "
            f"missing={missing[:20]}, unexpected={unexpected[:20]}"
        )
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model
