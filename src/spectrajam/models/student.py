from __future__ import annotations

import math

import torch
from torch import nn


class MaskedTemporalEncoder(nn.Module):
    def __init__(
        self,
        band_count: int,
        model_dim: int,
        heads: int,
        layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.input_projection = nn.Sequential(
            nn.Linear(band_count, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=layers, enable_nested_tensor=False
        )
        self.pool_context = nn.GRU(model_dim, model_dim, batch_first=True)
        self.pool_query = nn.Linear(model_dim, 1)
        self.relative_time_weight = nn.Parameter(torch.tensor(0.25))

    def _temporal_encoding(self, day_of_year: torch.Tensor) -> torch.Tensor:
        position = day_of_year.unsqueeze(-1).float()
        divisor = torch.exp(
            torch.arange(0, self.model_dim, 2, device=day_of_year.device, dtype=torch.float32)
            * -(math.log(10000.0) / self.model_dim)
        )
        result = torch.zeros(
            *day_of_year.shape,
            self.model_dim,
            device=day_of_year.device,
            dtype=torch.float32,
        )
        result[..., 0::2] = torch.sin(position * divisor)
        result[..., 1::2] = torch.cos(position * divisor)
        return result

    def forward(
        self,
        bands: torch.Tensor,
        day_of_year: torch.Tensor,
        valid: torch.Tensor,
        relative_day: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if relative_day is not None and relative_day.shape != day_of_year.shape:
            raise ValueError("relative_day must match day_of_year")
        if valid.dtype is not torch.bool:
            valid = valid.bool()
        if valid.shape[1] > 1 and ((~valid[:, :-1]) & valid[:, 1:]).any():
            raise ValueError("valid timesteps must be left-packed before padding")
        all_missing = ~valid.any(dim=1)
        safe_valid = valid.clone()
        safe_valid[all_missing, 0] = True
        value = self.input_projection(bands)
        value = value + self._temporal_encoding(day_of_year).to(dtype=value.dtype)
        if relative_day is not None:
            relative_encoding = self._temporal_encoding(relative_day + 1)
            value = value + self.relative_time_weight * relative_encoding.to(value.dtype)
        value = self.encoder(value, src_key_padding_mask=~safe_valid)
        context, _ = self.pool_context(value)
        scores = self.pool_query(context).squeeze(-1).masked_fill(~safe_valid, float("-inf"))
        pooled = (torch.softmax(scores, dim=1).unsqueeze(-1) * value).sum(dim=1)
        pooled[all_missing] = 0
        return pooled


class RegionalStudent(nn.Module):
    """Compact standalone d-pixel encoder used by the teacher-student track."""

    def __init__(
        self,
        model_dim: int = 256,
        layers: int = 2,
        heads: int = 4,
        feedforward_dim: int = 1024,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.s2_encoder = MaskedTemporalEncoder(
            10, model_dim, heads, layers, feedforward_dim, dropout
        )
        self.s1_encoder = MaskedTemporalEncoder(
            2, model_dim, heads, layers, feedforward_dim, dropout
        )
        self.window_context = nn.Sequential(
            nn.Linear(5, model_dim),
            nn.GELU(),
            nn.LayerNorm(model_dim),
        )
        self.fusion = nn.Sequential(
            nn.Linear(model_dim * 3, model_dim),
            nn.LayerNorm(model_dim),
            nn.GELU(),
            nn.Linear(model_dim, output_dim),
        )

    @staticmethod
    def _context_features(
        s2_valid: torch.Tensor,
        s1_valid: torch.Tensor,
        window_duration_days: torch.Tensor | None,
        s2_observation_count: torch.Tensor | None,
        s1_observation_count: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = s2_valid.shape[0]
        device = s2_valid.device
        if window_duration_days is None:
            duration = torch.full((batch,), 365.0, device=device)
        else:
            if window_duration_days.shape != (batch,):
                raise ValueError("window_duration_days must have shape [batch]")
            duration = window_duration_days.to(device=device, dtype=torch.float32)
        s2_count = (
            s2_valid.sum(dim=1).float()
            if s2_observation_count is None
            else s2_observation_count.to(device=device, dtype=torch.float32)
        )
        s1_count = (
            s1_valid.sum(dim=1).float()
            if s1_observation_count is None
            else s1_observation_count.to(device=device, dtype=torch.float32)
        )
        if s2_count.shape != (batch,) or s1_count.shape != (batch,):
            raise ValueError("observation counts must have shape [batch]")
        return torch.stack(
            [
                torch.log1p(duration) / math.log(367.0),
                torch.log1p(s2_count) / math.log(65.0),
                torch.log1p(s1_count) / math.log(65.0),
                (s2_count == 0).float(),
                (s1_count == 0).float(),
            ],
            dim=-1,
        )

    def forward(
        self,
        s2_bands: torch.Tensor,
        s2_day_of_year: torch.Tensor,
        s2_valid: torch.Tensor,
        s1_bands: torch.Tensor,
        s1_day_of_year: torch.Tensor,
        s1_valid: torch.Tensor,
        *,
        s2_relative_day: torch.Tensor | None = None,
        s1_relative_day: torch.Tensor | None = None,
        window_duration_days: torch.Tensor | None = None,
        s2_observation_count: torch.Tensor | None = None,
        s1_observation_count: torch.Tensor | None = None,
    ) -> torch.Tensor:
        s2 = self.s2_encoder(
            s2_bands, s2_day_of_year, s2_valid, s2_relative_day
        )
        s1 = self.s1_encoder(
            s1_bands, s1_day_of_year, s1_valid, s1_relative_day
        )
        context_features = self._context_features(
            s2_valid,
            s1_valid,
            window_duration_days,
            s2_observation_count,
            s1_observation_count,
        )
        context_dtype = next(self.window_context.parameters()).dtype
        context = self.window_context(context_features.to(context_dtype))
        embedding = self.fusion(torch.cat([s2, s1, context], dim=-1))
        all_missing = ~s2_valid.any(dim=1) & ~s1_valid.any(dim=1)
        return embedding.masked_fill(all_missing[:, None], 0)
