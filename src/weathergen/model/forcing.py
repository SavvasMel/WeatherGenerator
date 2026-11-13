# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""
Generic per-step spatial forcing for the forecasting engine.

A *forcing* is a prescribed field (SST, sea ice, aerosols, a CO2 field, solar forcing, ...)
that is re-read at every forecast step's valid time (see the ``type: forcing`` stream) and
injected into the forecasting engine so the rolling latent state is continually reminded of
an external boundary condition. Nothing here is specific to any particular field -- SST is
simply the first instance.

To inject a forcing it is first placed on the FE latent grid: the (fixed) forcing point cloud
is binned (scatter-mean) onto the HEALPix cells the FE tokens live on, then embedded to the
model width by a small MLP. The result -- a per-cell forcing embedding -- is shared by every
injection mode (``additive`` / ``cross_attn`` / ``global`` / ``adaln_local``).

The scatter index depends only on the (fixed) forcing grid and the HEALPix level, so it is
precomputed once via :func:`build_forcing_cell_index`. Everything here is cheap: one scatter
and one matmul per forecast step, independent of the assimilation engine.
"""

import numpy as np
import torch
import torch.nn as nn
from numpy.typing import NDArray

from weathergen.datasets.utils import coords_to_hpyidxs


def build_forcing_cell_index(
    latitudes: NDArray, longitudes: NDArray, healpix_level: int
) -> torch.Tensor:
    """
    Map each forcing grid point to its HEALPix latent cell (nested convention).

    Uses the same convention (``ang2pix(..., nest=True)``) as the tokenizer, so the binned
    forcing cells align one-to-one with the FE latent token cells.

    Parameters
    ----------
    latitudes, longitudes :
        Forcing grid point coordinates, degrees, shape ``(num_points,)``.
    healpix_level :
        HEALPix level of the FE latent grid (num_cells = 12 * 4**healpix_level).

    Returns
    -------
    LongTensor of shape ``(num_points,)`` with the cell index in ``[0, num_cells)`` for
    every forcing point.
    """
    cell_idx = coords_to_hpyidxs(healpix_level, np.asarray(latitudes), np.asarray(longitudes))
    return torch.as_tensor(np.asarray(cell_idx), dtype=torch.long)


def scatter_to_cells(
    values: torch.Tensor, cell_idx: torch.Tensor, num_cells: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Scatter-mean per-point forcing values onto HEALPix cells, ignoring NaNs.

    Parameters
    ----------
    values :
        Forcing point values, shape ``(..., num_points, num_vars)``. NaNs (e.g. SST over
        land) are treated as missing and excluded from the per-cell mean.
    cell_idx :
        Cell index per point, shape ``(num_points,)`` (from ``build_forcing_cell_index``).
    num_cells :
        Number of HEALPix cells.

    Returns
    -------
    (cell_values, cell_valid)
        cell_values : ``(..., num_cells, num_vars)`` per-cell mean (0 where no valid point
            contributed to a cell/var).
        cell_valid  : ``(..., num_cells, num_vars)`` fraction of valid points that
            contributed (0 = no data, e.g. all-land cell), usable as an input feature/mask.
    """
    *batch_shape, num_points, num_vars = values.shape
    flat = values.reshape(-1, num_points, num_vars)  # (B, P, V)
    n_batch = flat.shape[0]
    device = flat.device

    valid = torch.isfinite(flat)  # (B, P, V)
    filled = torch.where(valid, flat, torch.zeros_like(flat))

    idx = cell_idx.to(device).view(1, num_points, 1).expand(n_batch, num_points, num_vars)

    sums = torch.zeros(n_batch, num_cells, num_vars, device=device, dtype=flat.dtype)
    sums.scatter_add_(1, idx, filled)
    counts = torch.zeros(n_batch, num_cells, num_vars, device=device, dtype=flat.dtype)
    counts.scatter_add_(1, idx, valid.to(flat.dtype))

    cell_values = torch.where(counts > 0, sums / counts.clamp_min(1.0), torch.zeros_like(sums))
    total = torch.zeros(n_batch, num_cells, num_vars, device=device, dtype=flat.dtype)
    total.scatter_add_(1, idx, torch.ones_like(filled))
    cell_valid = torch.where(total > 0, counts / total.clamp_min(1.0), torch.zeros_like(counts))

    cell_values = cell_values.reshape(*batch_shape, num_cells, num_vars)
    cell_valid = cell_valid.reshape(*batch_shape, num_cells, num_vars)
    return cell_values, cell_valid


class LearnedForcingPool(nn.Module):
    """
    Learnable replacement for the fixed scatter-mean in :func:`scatter_to_cells`.

    HEALPix cells don't have a fixed number of forcing grid points each (coastal cells vs.
    open-ocean cells differ), so a standard attention module (fixed-size Q/K/V + masking)
    doesn't fit directly. This instead scores every point with a small per-point MLP and
    normalises the scores *within each cell* via a segment-softmax (two ``scatter_add_``
    calls, the same primitive ``scatter_to_cells`` already uses for the mean) -- so cells
    with more points just have more terms in their softmax, no padding required.

    Same NaN-masking discipline as ``scatter_to_cells``: invalid (e.g. land) points get
    zero weight rather than corrupting the pooled value -- and, like ``scatter_to_cells``,
    each variable is masked/normalised *independently*, since e.g. ``land_sea_mask`` is
    defined everywhere while ``sst``/``sea_ice_cover`` are NaN over land at the same points.
    """

    def __init__(self, num_vars: int, hidden_factor: int = 4) -> None:
        super().__init__()
        self.num_vars = num_vars
        hidden = max(hidden_factor * num_vars, 8)
        # per-variable logits: shape (P, V) out, not a single shared per-point score, so each
        # variable's segment-softmax only involves points valid for *that* variable.
        self.score = nn.Sequential(
            nn.Linear(num_vars, hidden),
            nn.SiLU(),
            nn.Linear(hidden, num_vars),
        )

    def forward(
        self, values: torch.Tensor, cell_idx: torch.Tensor, num_cells: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        values : ``(num_points, num_vars)`` raw per-point forcing values for this step.
        cell_idx : ``(num_points,)`` cell index per point (from ``build_forcing_cell_index``).
        num_cells : number of HEALPix cells.

        Returns
        -------
        (cell_values, cell_valid), same shapes/semantics as ``scatter_to_cells``:
        ``cell_values`` a ``(num_cells, num_vars)`` learned weighted mean (0 where no valid
        point contributed), ``cell_valid`` the ``(num_cells, num_vars)`` valid-point fraction.
        """
        num_points, num_vars = values.shape
        device = values.device
        cell_idx = cell_idx.to(device)
        idx = cell_idx.view(num_points, 1).expand(num_points, num_vars)  # (P, V)

        valid = torch.isfinite(values)  # (P, V)
        filled = torch.where(valid, values, torch.zeros_like(values))

        logits = self.score(filled)  # (P, V) -- one logit per (point, variable)
        logits = logits.masked_fill(~valid, -1e9)

        # segment-softmax per variable: normalise exp(logits) within each cell, independently
        # per column, using the same scatter_add_ primitive scatter_to_cells uses for the mean.
        exp_logits = torch.exp(logits - logits.detach().amax(dim=0, keepdim=True))
        denom = torch.zeros(num_cells, num_vars, device=device, dtype=values.dtype)
        denom.scatter_add_(0, idx, exp_logits)
        weights = exp_logits / denom.gather(0, idx).clamp_min(1e-12)  # (P, V)

        cell_values = torch.zeros(num_cells, num_vars, device=device, dtype=values.dtype)
        cell_values.scatter_add_(0, idx, weights * filled)

        counts = torch.zeros(num_cells, num_vars, device=device, dtype=values.dtype)
        counts.scatter_add_(0, idx, valid.to(values.dtype))
        total = torch.zeros(num_cells, num_vars, device=device, dtype=values.dtype)
        total.scatter_add_(0, idx, torch.ones_like(filled))
        cell_valid = torch.where(total > 0, counts / total.clamp_min(1.0), torch.zeros_like(counts))

        # cells with no valid points for a variable: weights there are garbage over masked-out
        # (-1e9 logit) points; zero explicitly rather than rely on the exp(-1e9) underflow.
        cell_values = torch.where(cell_valid > 0, cell_values, torch.zeros_like(cell_values))

        return cell_values, cell_valid


class ForcingEmbed(nn.Module):
    """
    Per-cell forcing embedding shared by all injection modes.

    Normalises the per-cell forcing field with the dataset statistics (or identity if the
    field is already normalised upstream), appends a validity feature (valid fraction per
    cell/var) so the model can tell present from absent/no-data, and maps to the model width
    with a small MLP.

    Parameters
    ----------
    num_vars :
        Number of forcing source channels (e.g. sea_surface_temperature, sea_ice_cover,
        land_sea_mask -> 3).
    dim_embed :
        Output embedding width (typically the FE model dim for additive injection).
    mean, stdev :
        Per-channel normalisation statistics, shape ``(num_vars,)``. If None, the input is
        assumed already normalised.
    hidden_factor :
        Width multiplier for the MLP hidden layer.
    """

    def __init__(
        self,
        num_vars: int,
        dim_embed: int,
        mean: NDArray | None = None,
        stdev: NDArray | None = None,
        hidden_factor: int = 2,
    ) -> None:
        super().__init__()
        self.num_vars = num_vars
        self.dim_embed = dim_embed

        if mean is None:
            mean = np.zeros(num_vars, dtype=np.float32)
        if stdev is None:
            stdev = np.ones(num_vars, dtype=np.float32)
        # kept as plain numpy (not a tensor/buffer) so reset_parameters() can restore these
        # exact values after an FSDP2 meta-device to_empty() wipes the buffers -- see
        # ForcingInjection.reset_parameters() for why this is necessary. np.array(..., copy=True)
        # (not asarray) so this is never aliased into the buffer below: torch.as_tensor shares
        # memory with a same-dtype CPU numpy array rather than copying, so without an explicit
        # copy here, in-place mutation of the buffer (e.g. the to_empty() corruption itself)
        # would silently corrupt this "backup" too.
        self._mean_init = np.array(mean, dtype=np.float32, copy=True)
        self._stdev_init = np.array(stdev, dtype=np.float32, copy=True)
        self.register_buffer("mean", torch.as_tensor(self._mean_init).clone())
        self.register_buffer("stdev", torch.as_tensor(self._stdev_init).clone())

        # input = normalised values (num_vars) concatenated with validity (num_vars)
        in_dim = 2 * num_vars
        hidden = hidden_factor * dim_embed
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim_embed),
        )

    def reset_parameters(self) -> None:
        """
        Restore mean/stdev to their intended values and reinitialise the MLP.

        FSDP2's meta-device build (``model.to_empty()`` + ``Model.reset_parameters()``)
        only auto-resets ``nn.Linear``/``nn.LayerNorm`` submodules (see model.py); buffers
        registered directly here (mean/stdev) are left as uninitialised memory otherwise,
        which silently breaks the "identity if unspecified" normalisation this class
        documents (observed in practice as stdev=0 -> division amplifying values ~1e6x).
        """
        self.mean.data.copy_(torch.as_tensor(self._mean_init, device=self.mean.device))
        self.stdev.data.copy_(torch.as_tensor(self._stdev_init, device=self.stdev.device))
        for module in self.mlp.modules():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def forward(self, cell_values: torch.Tensor, cell_valid: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        cell_values : ``(..., num_cells, num_vars)`` per-cell forcing values (from scatter).
        cell_valid  : ``(..., num_cells, num_vars)`` per-cell validity fraction.

        Returns
        -------
        ``(..., num_cells, dim_embed)`` per-cell forcing embedding.
        """
        mean = self.mean.to(cell_values.dtype)
        stdev = self.stdev.to(cell_values.dtype).clamp_min(1e-6)
        normed = (cell_values - mean) / stdev
        # zero-out normalised value where there is no data so it reads as "absent"
        normed = torch.where(cell_valid > 0, normed, torch.zeros_like(normed))
        x = torch.cat([normed, cell_valid.to(normed.dtype)], dim=-1)
        return self.mlp(x)


class ForcingInjection(nn.Module):
    """
    Inject a per-step spatial forcing into the forecasting engine, one of four modes.

    Applied once per forecast step, in the model rollout loop, just before the FE advances the
    ``(B, num_tokens, dim)`` latent state. The auxiliary tokens (register/class) are left
    untouched; only the per-cell patch tokens / the FE conditioning are affected. The forcing
    field for a step is global across the batch (same forecast time), so a per-cell tensor
    ``(num_cells, dim)`` broadcasts across the batch dimension.

    Modes (all a **zero-initialised gated residual** -> exact no-op at init, so a model
    fine-tuned from a pretrained checkpoint starts identically and learns to use the forcing):

      * ``additive``    : add the per-cell forcing embedding to the cell tokens.
      * ``cross_attn``  : cell tokens cross-attend to the per-cell forcing tokens (non-local
                          coupling / teleconnections).
      * ``global``      : pool the forcing over cells and add the projected vector to the
                          (global) FE conditioning -- the cheap "index" flavour; the existing
                          FE AdaLN is untouched.
      * ``adaln_local`` : make the FE conditioning per-cell (daytime condition + projected
                          per-cell forcing); every FE block's AdaLN then modulates each cell
                          token by its local forcing. Requires a condition stream (dim_aux > 0).

    ``forward`` returns ``(tokens, condition)``: token-nudge modes modify ``tokens``,
    conditioning modes modify ``condition``.
    """

    _MODES = ("none", "additive", "cross_attn", "global", "adaln_local")

    def __init__(
        self,
        num_vars: int,
        dim_embed: int,
        mode: str,
        dim_aux: int = 0,
        num_heads: int = 8,
        learned_pool: bool = False,
    ) -> None:
        super().__init__()
        assert mode in self._MODES, f"unknown forcing mode {mode!r}, expected {self._MODES}"
        self.mode = mode
        self.num_vars = num_vars
        self.dim_embed = dim_embed
        self.dim_aux = dim_aux
        self.learned_pool_module = LearnedForcingPool(num_vars) if learned_pool else None

        if mode == "none":
            return

        if mode in ("global", "adaln_local") and dim_aux <= 0:
            raise ValueError(
                f"forcing mode {mode!r} routes the forcing through the FE conditioning (AdaLN) "
                "and needs a condition stream (dim_aux > 0); use additive/cross_attn otherwise."
            )

        # Forcing field is already normalised by the sampler (reader stats) -> identity here.
        self.embed = ForcingEmbed(num_vars, dim_embed)
        # zero-init gate -> injection starts as an exact no-op
        self.gate = nn.Parameter(torch.zeros(1))

        if mode == "cross_attn":
            self.norm_q = nn.LayerNorm(dim_embed)
            self.cross_attn = nn.MultiheadAttention(dim_embed, num_heads, batch_first=True)
        elif mode in ("global", "adaln_local"):
            self.cond_proj = nn.Linear(dim_embed, dim_aux)

    def reset_parameters(self) -> None:
        """
        Restore the documented zero-init-gate no-op-at-init guarantee.

        Model.reset_parameters()'s generic sweep (model.py) only auto-resets
        nn.Linear/nn.LayerNorm submodules, so it's called explicitly on this module too --
        self-contained (doesn't rely on that generic sweep for its own children) so it also
        works standalone from model_interface.py's "new module not found in checkpoint" path
        (module_to_init.reset_parameters()), which previously raised AttributeError here
        since this method didn't exist.

        Without this, gate/embed.mean/embed.stdev -- none of which are nn.Linear/nn.LayerNorm
        -- are left as whatever uninitialised memory model.to_empty() produced (observed in
        practice: stdev=0, amplifying forcing values ~1e6x once gate drifts off zero).
        """
        if self.mode == "none":
            return
        self.gate.data.zero_()
        self.embed.reset_parameters()
        if self.mode == "cross_attn":
            self.norm_q.reset_parameters()
            self.cross_attn._reset_parameters()  # nn.MultiheadAttention's own (underscored) API
        elif self.mode in ("global", "adaln_local"):
            self.cond_proj.reset_parameters()
        if self.learned_pool_module is not None:
            for module in self.learned_pool_module.score.modules():
                if isinstance(module, nn.Linear):
                    module.reset_parameters()

    @staticmethod
    def _is_empty(x) -> bool:
        return x is None or not torch.is_tensor(x) or x.numel() == 0

    def _cell_emb(
        self,
        forcing_field: torch.Tensor,
        dtype,
        cell_idx: torch.Tensor | None,
        num_cells: int | None,
    ) -> torch.Tensor:
        """
        Per-cell embedding (num_cells, dim).

        Two input shapes for ``forcing_field``, selected by ``learned_pool_module``:
          * fixed mean (default): already-scattered ``(num_cells, 2*num_vars)`` = [cell_values
            | cell_valid], produced upstream in the dataloader (see ``scatter_to_cells``).
          * learned pool: raw, unscattered ``(num_points, num_vars)`` per-point values --
            pooled here, on the GPU, so ``learned_pool_module`` gets gradients.
        """
        if forcing_field.dim() == 3:  # tolerate a leading batch dim
            forcing_field = forcing_field[0]

        if self.learned_pool_module is not None:
            assert cell_idx is not None and num_cells is not None, (
                "learned_pool requires cell_idx/num_cells (raw per-point forcing_field)"
            )
            cell_values, cell_valid = self.learned_pool_module(forcing_field, cell_idx, num_cells)
        else:
            num_vars = forcing_field.shape[-1] // 2
            cell_values = forcing_field[..., :num_vars]
            cell_valid = forcing_field[..., num_vars:]

        return self.embed(cell_values, cell_valid).to(dtype)

    def forward(
        self,
        tokens: torch.Tensor,
        condition: torch.Tensor,
        forcing_field: torch.Tensor,
        num_aux: int,
        cell_idx: torch.Tensor | None = None,
        num_cells: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        tokens :
            FE latent state ``(B, num_tokens, dim)`` (num_aux leading aux tokens then num_cells).
        condition :
            FE conditioning fed to the AdaLN, global ``(dim_aux,)`` vector (or empty).
        forcing_field :
            Forcing data for this step; empty when no forcing is available (then this is a
            no-op). Shape depends on the pooling mode -- see ``_cell_emb``.
        num_aux :
            Number of leading auxiliary tokens to leave untouched.
        cell_idx, num_cells :
            Only required when ``learned_pool=True`` was passed to ``__init__`` -- the
            point -> HEALPix-cell index and cell count needed to pool ``forcing_field`` here.

        Returns
        -------
        (tokens, condition) with the forcing injected according to the mode.
        """
        if self.mode == "none" or self._is_empty(forcing_field):
            return tokens, condition

        cell_emb = self._cell_emb(
            forcing_field, tokens.dtype, cell_idx, num_cells
        )  # (num_cells, dim)

        if self.mode == "additive":
            patch = tokens[:, num_aux:] + self.gate * cell_emb
            tokens = torch.cat([tokens[:, :num_aux], patch], dim=1)
        elif self.mode == "cross_attn":
            q = self.norm_q(tokens[:, num_aux:])  # (B, num_cells, dim)
            kv = cell_emb.unsqueeze(0).expand(q.shape[0], -1, -1)
            delta, _ = self.cross_attn(q, kv, kv, need_weights=False)
            patch = tokens[:, num_aux:] + self.gate * delta
            tokens = torch.cat([tokens[:, :num_aux], patch], dim=1)
        elif self.mode == "global":
            g = self.cond_proj(cell_emb.mean(dim=0))  # (dim_aux,)
            condition = condition + self.gate * g
        elif self.mode == "adaln_local":
            # per-cell conditioning: broadcast the global daytime condition, add per-cell forcing
            base = condition.reshape(1, -1) if condition.numel() > 0 else 0.0
            condition = base + self.gate * self.cond_proj(cell_emb)  # (num_cells, dim_aux)

        return tokens, condition
