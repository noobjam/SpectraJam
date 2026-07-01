from __future__ import annotations

import math

import torch
from torch import nn

from .tessera_v11 import TesseraV11, TransformerEncoder


class WindowedTesseraEncoder(nn.Module):
    """Mask-aware runtime around the untouched TESSERA v1.1 checkpoint graph.

    The wrapper removes padded timesteps before each checkpoint-faithful branch.
    Empty modalities contribute a zero branch; a window with no observations at
    all returns the all-zero embedding. Dense inputs remain base-model equivalent.
    """

    def __init__(
        self,
        base: TesseraV11,
        condition_windows: bool = False,
        conditioner_hidden_dim: int = 64,
    ):
        super().__init__()
        self.base = base
        self.window_conditioner: nn.Module | None = None
        if condition_windows:
            self.window_conditioner = nn.Sequential(
                nn.Linear(5, conditioner_hidden_dim),
                nn.GELU(),
                nn.Linear(conditioner_hidden_dim, base.output_dim),
            )
            nn.init.zeros_(self.window_conditioner[-1].weight)
            nn.init.zeros_(self.window_conditioner[-1].bias)

    @staticmethod
    def _context_features(
        s2_valid: torch.Tensor,
        s1_valid: torch.Tensor,
        duration_days: torch.Tensor | None,
        s2_observation_count: torch.Tensor | None,
        s1_observation_count: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = s2_valid.shape[0]
        if duration_days is None:
            duration = torch.full((batch,), 365.0, device=s2_valid.device)
        else:
            if duration_days.shape != (batch,):
                raise ValueError("window_duration_days must have shape [batch]")
            duration = duration_days.to(device=s2_valid.device, dtype=torch.float32)
        s2_count = (
            s2_valid.sum(dim=1).float()
            if s2_observation_count is None
            else s2_observation_count.to(device=s2_valid.device, dtype=torch.float32)
        )
        s1_count = (
            s1_valid.sum(dim=1).float()
            if s1_observation_count is None
            else s1_observation_count.to(device=s1_valid.device, dtype=torch.float32)
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

    @staticmethod
    def _validate_modality(
        bands: torch.Tensor,
        day_of_year: torch.Tensor,
        valid: torch.Tensor,
        channels: int,
    ) -> None:
        if bands.ndim != 3 or bands.shape[-1] != channels:
            raise ValueError(f"bands must have shape [batch, time, {channels}]")
        if day_of_year.shape != bands.shape[:2] or valid.shape != bands.shape[:2]:
            raise ValueError("band, day-of-year, and validity shapes do not align")

    @staticmethod
    def _encode_branch(
        backbone: TransformerEncoder,
        bands: torch.Tensor,
        day_of_year: torch.Tensor,
        valid: torch.Tensor,
        relative_day: torch.Tensor | None,
    ) -> torch.Tensor:
        batch = bands.shape[0]
        width = backbone.attn_pool.input_dim
        output = bands.new_zeros((batch, width))
        counts = valid.bool().sum(dim=1)
        for count in sorted(int(value) for value in counts.unique().tolist() if value):
            rows = torch.nonzero(counts == count, as_tuple=False).flatten()
            sequences: list[torch.Tensor] = []
            for row_tensor in rows:
                row = int(row_tensor)
                selected = torch.nonzero(valid[row].bool(), as_tuple=False).flatten()
                if relative_day is not None:
                    order = torch.argsort(relative_day[row, selected], stable=True)
                    selected = selected[order]
                sequences.append(
                    torch.cat(
                        [
                            bands[row, selected],
                            day_of_year[row, selected, None].to(bands),
                        ],
                        dim=-1,
                    )
                )
            encoded = backbone(torch.stack(sequences, dim=0))
            output = output.index_copy(0, rows, encoded)
        return output

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
        self._validate_modality(s2_bands, s2_day_of_year, s2_valid, 10)
        self._validate_modality(s1_bands, s1_day_of_year, s1_valid, 2)
        if s2_bands.shape[0] != s1_bands.shape[0]:
            raise ValueError("S1 and S2 batch sizes differ")
        if s2_relative_day is not None and s2_relative_day.shape != s2_valid.shape:
            raise ValueError("s2_relative_day must match s2_valid")
        if s1_relative_day is not None and s1_relative_day.shape != s1_valid.shape:
            raise ValueError("s1_relative_day must match s1_valid")

        s2 = self._encode_branch(
            self.base.s2_backbone,
            s2_bands,
            s2_day_of_year,
            s2_valid,
            s2_relative_day,
        )
        s1 = self._encode_branch(
            self.base.s1_backbone,
            s1_bands,
            s1_day_of_year,
            s1_valid,
            s1_relative_day,
        )
        embedding = self.base.dim_reducer(torch.cat([s2, s1], dim=-1))[
            ..., : self.base.output_dim
        ]
        if self.window_conditioner is not None:
            features = self._context_features(
                s2_valid,
                s1_valid,
                window_duration_days,
                s2_observation_count,
                s1_observation_count,
            )
            conditioner_dtype = next(self.window_conditioner.parameters()).dtype
            embedding = embedding + self.window_conditioner(
                features.to(conditioner_dtype)
            ).to(embedding.dtype)
        all_missing = ~s2_valid.any(dim=1) & ~s1_valid.any(dim=1)
        return embedding.masked_fill(all_missing[:, None], 0)
