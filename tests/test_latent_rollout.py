"""Guards and scheduling for the latent rollout.

Every failure mode here is silent if unguarded: a frozen ocean, a latent that drifts while
the decoded fields still look plausible, or a resume that quietly re-anchors the trajectory.
"""

import numpy as np
import pytest
import torch

from weathergen.rollout import LatentRollout
from weathergen.rollout.latent_rollout import (
    LATENT_NORM_GROWTH,
    MAX_IDENTICAL_FORCING_STEPS,
)


def _bare(reencode_every=0, has_forcing=True):
    """A LatentRollout with only the attributes the logic under test touches."""
    r = object.__new__(LatentRollout)
    r.reencode_every = reencode_every
    r.tokens = None
    r._latent_norm0 = None
    r._prev_forcing_sig = None
    r._identical_forcing_steps = 0
    r._n_reencodes = 0
    r.core = type("C", (), {"forcing_modules": {"OCEAN": 1} if has_forcing else {}})()
    return r


class _Batch:
    def __init__(self, forcing):
        self.forcing = forcing


# --------------------------------------------------------------- re-encode schedule


@pytest.mark.parametrize(
    "every,expected",
    [
        (1, [True] * 6),  # every iteration -> the physical rollout
        (0, [True] + [False] * 5),  # only the initial encode
        (3, [True, False, False, True, False, False]),
    ],
)
def test_reencode_schedule(every, expected):
    r = _bare(reencode_every=every)
    got = []
    for step in range(6):
        r.state = type("S", (), {"step": step})()
        got.append(r._should_reencode())
        r.tokens = torch.zeros(1)  # something is carried after the first iteration
    assert got == expected


# ------------------------------------------------------------------ forcing guard


def _field(val):
    return torch.full((4, 2), float(val))


def test_frozen_forcing_is_caught():
    r = _bare()
    batch = _Batch([{"OCEAN": _field(1.0)}])
    # the same field repeated is the signature of a forcing that is not being refreshed
    for _ in range(MAX_IDENTICAL_FORCING_STEPS):
        r._check_forcing_is_live(batch)
    with pytest.raises(RuntimeError, match="bitwise identical"):
        r._check_forcing_is_live(batch)


def test_changing_forcing_passes():
    r = _bare()
    for i in range(10):
        r._check_forcing_is_live(_Batch([{"OCEAN": _field(i)}]))


def test_missing_forcing_raises_when_injection_configured():
    r = _bare(has_forcing=True)
    with pytest.raises(RuntimeError, match="no forcing field"):
        r._check_forcing_is_live(_Batch([{}]))


def test_missing_forcing_is_fine_without_injection():
    r = _bare(has_forcing=False)
    r._check_forcing_is_live(_Batch([{}]))


# ------------------------------------------------------------------- latent guard


def test_latent_blowup_is_caught():
    r = _bare()
    t = np.datetime64("2000-01-01")
    base = torch.ones(100)
    r._check_latent(base, t)  # establishes the reference norm
    r._check_latent(base * (LATENT_NORM_GROWTH - 1), t)  # still under threshold
    with pytest.raises(FloatingPointError, match="blew up"):
        r._check_latent(base * (LATENT_NORM_GROWTH + 1), t)


def test_nonfinite_latent_is_caught():
    r = _bare()
    bad = torch.full((10,), float("nan"))
    with pytest.raises(FloatingPointError, match="diverged"):
        r._check_latent(bad, np.datetime64("2000-01-01"))


# ----------------------------------------------------------------------- sidecar


def test_latent_sidecar_roundtrip(tmp_path):
    r = _bare(reencode_every=5)
    out = tmp_path / "amip_test.zarr"
    r.device = "cpu"
    r.tokens = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    r.state = type("S", (), {"step": 7, "timestamp": np.datetime64("2001-02-03")})()

    r.save_latent(out)
    assert r._latent_sidecar(out).exists()

    r2 = _bare(reencode_every=5)
    r2.device = "cpu"
    assert r2.load_latent(out, 7, np.datetime64("2001-02-03"))
    assert torch.equal(r2.tokens, r.tokens)


def test_sidecar_at_wrong_step_is_rejected(tmp_path):
    """A sidecar from a different step would splice two trajectories together."""
    r = _bare()
    out = tmp_path / "amip_test.zarr"
    r.device = "cpu"
    r.tokens = torch.ones(1, 4, 3)
    r.state = type("S", (), {"step": 7, "timestamp": np.datetime64("2001-02-03")})()
    r.save_latent(out)

    r2 = _bare()
    r2.device = "cpu"
    assert not r2.load_latent(out, 9, np.datetime64("2001-02-03"))  # wrong step
    assert not r2.load_latent(out, 7, np.datetime64("2001-02-04"))  # wrong time
    assert r2.tokens is None


def test_missing_sidecar_is_absent_not_an_error(tmp_path):
    r = _bare()
    r.device = "cpu"
    assert not r.load_latent(tmp_path / "nope.zarr", 0, np.datetime64("2000-01-01"))


def test_sidecar_timestamp_compares_as_instant_not_text(tmp_path):
    """numpy renders datetime64 at its own precision, so string comparison is not equality.

    The sidecar stores str(np.datetime64(...)) which may be minute precision, while the
    zarr's dates array is datetime64[s]. Comparing the two as text made a pure latent
    rollout reject the sidecar it had just written and refuse to resume.
    """
    r = _bare(reencode_every=0)
    out = tmp_path / "amip_test.zarr"
    r.device = "cpu"
    r.tokens = torch.ones(1, 4, 3)
    # minute precision going in, as np.datetime64 arithmetic produces
    r.state = type("S", (), {"step": 5, "timestamp": np.datetime64("2015-01-06T00:00")})()
    r.save_latent(out)

    r2 = _bare(reencode_every=0)
    r2.device = "cpu"
    # second precision coming back, as the zarr's M8[s] dates array yields
    assert r2.load_latent(out, 5, np.datetime64("2015-01-06T00:00:00"))
    assert torch.equal(r2.tokens, r.tokens)
