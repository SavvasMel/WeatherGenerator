# (C) Copyright 2024 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
"""Per-step injection of prescribed forcings (e.g. SST, an anchor) into the forecasting engine."""

import numpy as np
import torch
import torch.nn as nn

from weathergen.datasets.utils import coords_to_hpyidxs


def build_forcing_cell_index(
    latitudes: np.ndarray, longitudes: np.ndarray, healpix_level: int
) -> torch.Tensor:
    """Map each forcing grid point to its latent HEALPix cell."""
    idx = coords_to_hpyidxs(healpix_level, np.asarray(latitudes), np.asarray(longitudes))
    return torch.as_tensor(np.asarray(idx), dtype=torch.long)


def scatter_to_cells(
    values: torch.Tensor, cell_idx: torch.Tensor, num_cells: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average forcing points onto cells, ignoring NaNs; also return the valid fraction per cell."""
    *lead, num_points, num_vars = values.shape
    flat = values.reshape(-1, num_points, num_vars)
    n_batch = flat.shape[0]

    valid = torch.isfinite(flat)
    filled = torch.where(valid, flat, torch.zeros_like(flat))
    idx = cell_idx.to(flat.device).view(1, num_points, 1).expand(n_batch, num_points, num_vars)

    def _bin(src):
        out = torch.zeros(n_batch, num_cells, num_vars, device=flat.device, dtype=flat.dtype)
        return out.scatter_add_(1, idx, src)

    sums = _bin(filled)                       # sum of valid values per cell
    counts = _bin(valid.to(flat.dtype))       # number of valid points per cell
    total = _bin(torch.ones_like(filled))     # number of points per cell

    cell_values = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    cell_valid = torch.where(total > 0, counts / total.clamp_min(1.0), torch.zeros_like(counts))
    return (cell_values.reshape(*lead, num_cells, num_vars),
            cell_valid.reshape(*lead, num_cells, num_vars))


class ForcingEmbed(nn.Module):
    """Embed per-cell forcing values together with their validity fraction."""

    def __init__(self, num_vars: int, dim_embed: int,
                 mean: np.ndarray | None = None, stdev: np.ndarray | None = None,
                 hidden_factor: int = 2) -> None:
        super().__init__()
        self.num_vars, self.dim_embed = num_vars, dim_embed
        mean = np.zeros(num_vars, dtype=np.float32) if mean is None else np.asarray(mean)
        stdev = np.ones(num_vars, dtype=np.float32) if stdev is None else np.asarray(stdev)
        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("stdev", torch.as_tensor(stdev, dtype=torch.float32))
        self.mlp = nn.Sequential(
            nn.Linear(2 * num_vars, hidden_factor * dim_embed),
            nn.SiLU(),
            nn.Linear(hidden_factor * dim_embed, dim_embed),
        )

    def reset_parameters(self) -> None:
        """Reset to identity normalisation and fresh MLP weights."""
        with torch.no_grad():
            self.mean.zero_()
            self.stdev.fill_(1.0)
        for m in self.mlp:
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    def forward(self, cell_values: torch.Tensor, cell_valid: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(cell_values.dtype)
        stdev = self.stdev.to(cell_values.dtype).clamp_min(1e-6)
        normed = (cell_values - mean) / stdev
        # zero where nothing contributed, so it reads as "absent" rather than as a value
        normed = torch.where(cell_valid > 0, normed, torch.zeros_like(normed))
        return self.mlp(torch.cat([normed, cell_valid.to(normed.dtype)], dim=-1))


class ForcingInjection(nn.Module):
    """Add the embedded forcing to the latent state, leaving auxiliary tokens untouched."""

    SUPPORTED = ("none", "additive")

    def __init__(self, num_vars: int, dim_embed: int, mode: str = "additive",
                 dim_aux: int = 0, num_heads: int = 8, gate_init: float = 0.0) -> None:
        super().__init__()
        if mode not in self.SUPPORTED:
            raise ValueError(
                f"forcing injection mode {mode!r} not supported; expected {self.SUPPORTED}"
            )
        self.mode, self.num_vars, self.dim_embed = mode, num_vars, dim_embed
        self.gate_init = float(gate_init)
        if mode == "none":
            return
        self.embed = ForcingEmbed(num_vars, dim_embed)
        # gate_init 0: no-op on warm start; use 1.0 when training from scratch
        self.gate = nn.Parameter(torch.full((1,), self.gate_init))

    def reset_parameters(self) -> None:
        """Reset the gate to ``gate_init`` and the embedding."""
        if self.mode == "none":
            return
        with torch.no_grad():
            self.gate.fill_(self.gate_init)
        self.embed.reset_parameters()

    @staticmethod
    def _empty(x) -> bool:
        return x is None or not torch.is_tensor(x) or x.numel() == 0

    def forward(self, tokens: torch.Tensor, condition: torch.Tensor,
                forcing_field: torch.Tensor, num_aux: int):
        """`forcing_field` is (num_cells, 2*num_vars) = [values | validity]; absent is a no-op."""
        if self.mode == "none" or self._empty(forcing_field):
            return tokens, condition
        if forcing_field.dim() == 3:
            forcing_field = forcing_field[0]
        n = forcing_field.shape[-1] // 2
        cell_emb = self.embed(forcing_field[..., :n], forcing_field[..., n:]).to(tokens.dtype)
        patch = tokens[:, num_aux:] + self.gate * cell_emb
        return torch.cat([tokens[:, :num_aux], patch], dim=1), condition
