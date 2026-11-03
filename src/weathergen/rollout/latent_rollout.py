"""Rollout that carries the forecasting-engine latent between iterations.

``reencode_every``: 1 = physical rollout, N = re-encode every N iterations, 0 = pure latent.
A latent sidecar is saved next to the zarr for restarts.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .amip_rollout import AMIPRollout, RolloutState

logger = logging.getLogger(__name__)

# latent-norm growth treated as divergence
LATENT_NORM_GROWTH = 20.0

# forcing identical for this many steps = not refreshed
MAX_IDENTICAL_FORCING_STEPS = 3


@dataclass
class LatentRolloutState(RolloutState):
    """Physical state (for output and re-anchoring) plus the carried latent."""

    tokens: torch.Tensor | None = None


class LatentRollout(AMIPRollout):
    """Rollout carrying the forecasting-engine latent between outer iterations."""

    def __init__(self, *args, reencode_every: int = 0, **kwargs):
        super().__init__(*args, **kwargs)

        self.reencode_every = int(reencode_every)
        if self.reencode_every < 0:
            raise ValueError(
                f"rollout.latent.reencode_every={self.reencode_every} must be >= 0 "
                "(0 = never re-encode, N = re-encode every N iterations)."
            )

        # FSDP shards inner modules; DDP wraps the top level
        self.core = getattr(self.model, "module", self.model)
        for name in ("encode_to_latent", "advance_latent"):
            if not hasattr(self.core, name):
                raise RuntimeError(
                    f"model has no {name}(); this build predates the latent-rollout split "
                    "in model.py. Running here would silently fall back to nothing."
                )

        if not getattr(self.core, "forcing_modules", None):
            logger.warning(
                "No forcing injection is configured. In a latent rollout the encoder runs "
                "once, so a prescribed boundary condition that arrives only through the "
                "encoder is frozen at initialisation for the whole run."
            )

        self.tokens: torch.Tensor | None = None
        # set here and not only in initialize(): a resume skips initialize() entirely
        self._latent_norm0: float | None = None
        self._prev_forcing_sig = None
        self._identical_forcing_steps = 0
        self._n_reencodes = 0

        logger.info(
            "LatentRollout: reencode_every=%d (%s), %d forecast steps/iter",
            self.reencode_every,
            {0: "pure latent"}.get(self.reencode_every, "hybrid")
            if self.reencode_every != 1
            else "equivalent to the physical rollout",
            self.num_steps,
        )


    def _check_forcing_is_live(self, batch) -> None:
        """A prescribed forcing must be present and must actually change between steps."""
        fields = getattr(batch, "forcing", None)
        present = [
            f
            for step_fields in (fields or [])
            if isinstance(step_fields, dict)
            for f in step_fields.values()
            if torch.is_tensor(f) and f.numel() > 0
        ]
        if not present:
            if getattr(self.core, "forcing_modules", None):
                raise RuntimeError(
                    "Forcing injection is configured but the batch carries no forcing "
                    "field. The rollout would integrate with no prescribed boundary "
                    "condition at all, which is not an AMIP run."
                )
            return

        sig = tuple(float(f.double().sum()) for f in present)
        if sig == self._prev_forcing_sig:
            self._identical_forcing_steps += 1
            if self._identical_forcing_steps >= MAX_IDENTICAL_FORCING_STEPS:
                raise RuntimeError(
                    f"The prescribed forcing has been bitwise identical for "
                    f"{self._identical_forcing_steps} consecutive iterations. It is not "
                    "being refreshed from disk, so the ocean is frozen."
                )
        else:
            self._identical_forcing_steps = 0
        self._prev_forcing_sig = sig

    def _check_latent(self, tokens: torch.Tensor, t) -> None:
        norm = float(torch.linalg.vector_norm(tokens.float()))
        if not np.isfinite(norm):
            raise FloatingPointError(f"Latent rollout diverged at {t}: latent norm is {norm}.")
        if self._latent_norm0 is None:
            self._latent_norm0 = norm
        elif norm > LATENT_NORM_GROWTH * self._latent_norm0:
            raise FloatingPointError(
                f"Latent rollout blew up at {t}: latent norm {norm:.4g} is "
                f"{norm / self._latent_norm0:.1f}x its value at initialisation "
                f"(threshold {LATENT_NORM_GROWTH}x)."
            )


    def _should_reencode(self) -> bool:
        if self.tokens is None:
            return True  # nothing carried yet
        if self.reencode_every == 0:
            return False
        return self.state.step % self.reencode_every == 0

    @torch.no_grad()
    def step(self) -> LatentRolloutState:
        """Advance one outer iteration, carrying the latent unless a re-encode is due."""
        if self.state is None:
            raise RuntimeError("Rollout not initialized; call initialize() first.")

        t = self.state.timestamp
        idx = self._index_for(t)
        reencode = self._should_reencode()

        batch = self._build_batch(idx, inject=reencode)
        self._check_forcing_is_live(batch)

        batch.to_device(self.device)
        source_samples = batch.get_source_samples()

        with self._autocast:
            if reencode:
                tokens, output = self.core.encode_to_latent(self.model_params, source_samples)
                self._n_reencodes += 1
            else:
                tokens, output = self.tokens, None
            tokens, output = self.core.advance_latent(
                self.model_params, source_samples, tokens, output
            )

        last = max(source_samples.get_output_idxs())
        next_data = self._decode_to_physical(output, batch, last)

        t_next = t + np.timedelta64(self.hours_per_iter, "h")
        if next_data.shape != self.state.data.shape:
            raise ValueError(
                f"State shape changed {self.state.data.shape} -> {next_data.shape} at "
                f"{t_next}. Almost certainly the forecast window ran past the end of "
                f"the dataset and the reader returned spoofed data. The last usable "
                f"start time is (test_config.end_date - {self.hours_per_iter}h)."
            )
        self._check_latent(tokens, t_next)
        self._check_finite(next_data, t_next)

        # detach so the carried latent never retains a graph across iterations
        self.tokens = tokens.detach()
        self.state = LatentRolloutState(
            timestamp=t_next, data=next_data, step=self.state.step + 1, tokens=self.tokens
        )
        return self.state


    def initialize(self, start) -> LatentRolloutState:
        state = super().initialize(start)
        self._latent_norm0 = None
        self.tokens = None  # encoded on the first step, from the seeded physical state
        return LatentRolloutState(
            timestamp=state.timestamp, data=state.data, step=state.step, tokens=None
        )

    def _latent_sidecar(self, output_path) -> Path:
        return Path(str(output_path).rstrip("/") + ".latent.pt")

    def save_latent(self, output_path) -> None:
        """Checkpoint the carried latent; the zarr alone cannot restart a latent rollout."""
        if self.tokens is None:
            return
        path = self._latent_sidecar(output_path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(
            {
                "tokens": self.tokens.cpu(),
                "timestamp": str(self.state.timestamp),
                "step": int(self.state.step),
                "reencode_every": self.reencode_every,
            },
            tmp,
        )
        tmp.replace(path)  # atomic: a torn sidecar is worse than none

    def _on_checkpoint(self, output_path, store) -> None:
        self.save_latent(output_path)
        store.attrs["latent_reencodes"] = self._n_reencodes

    def _after_resume(self, output_path, n_done: int, last_time, store) -> None:
        """Restore the carried latent; record in the attributes whether it was restored."""
        restored = self.load_latent(output_path, n_done, last_time)
        store.attrs["latent_resume_restored"] = bool(restored)
        if restored:
            logger.info("restored carried latent from sidecar at step %d", n_done)
        elif self.reencode_every == 0:
            raise RuntimeError(
                f"Cannot resume a pure latent rollout ({output_path}): the latent sidecar "
                "is missing or stale, and the zarr holds only decoded physical output. "
                "Re-encoding from it would start a different trajectory. Either restore "
                "the sidecar or start the rollout again."
            )
        else:
            logger.warning(
                "latent sidecar unusable; re-encoding from the last written physical "
                "state. This re-anchors the trajectory at step %d.", n_done,
            )
            self.tokens = None

    def load_latent(self, output_path, expect_step: int, expect_time) -> bool:
        """Restore the carried latent. False when it is absent or does not match."""
        path = self._latent_sidecar(output_path)
        if not path.exists():
            return False
        blob = torch.load(path, map_location="cpu", weights_only=False)
        if int(blob.get("step", -1)) != int(expect_step):
            logger.warning(
                "latent sidecar is at step %s but the zarr is at step %d; ignoring it",
                blob.get("step"), expect_step,
            )
            return False
        # compare as datetimes, not strings
        if np.datetime64(blob.get("timestamp"), "s") != np.datetime64(expect_time, "s"):
            logger.warning(
                "latent sidecar is at %s but the zarr is at %s; ignoring it",
                blob.get("timestamp"), expect_time,
            )
            return False
        self.tokens = blob["tokens"].to(self.device)
        self._latent_norm0 = None
        return True
