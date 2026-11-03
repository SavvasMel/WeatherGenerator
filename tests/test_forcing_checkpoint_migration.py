"""Checkpoints predating per-stream forcing injection must still load their weights.

Forcing injection used to be one module named `forcing_module`. Every checkpoint trained
before it became a ModuleDict stores its weights there. Without a migration the keys simply
do not match: the trained gate and embedding are reported as unused, the new module is
initialised from scratch, and the run proceeds with no prescribed forcing while looking
entirely healthy - which is what happened on the first E4 run.
"""

import pytest
import torch

from weathergen.model.model_interface import _migrate_legacy_forcing_keys

LEGACY = {
    "forcing_module.gate": torch.tensor([0.0996]),
    "forcing_module.embed.mlp.0.weight": torch.zeros(4, 4),
    "encoder.something.weight": torch.zeros(2, 2),
}


class _Model:
    def __init__(self, names):
        self.forcing_modules = torch.nn.ModuleDict(
            {n: torch.nn.Linear(1, 1) for n in names}
        )


def test_legacy_keys_are_remapped_onto_the_single_stream():
    out = _migrate_legacy_forcing_keys(dict(LEGACY), _Model(["ERA5_FORCING"]), "old.chkpt")
    assert "forcing_modules.ERA5_FORCING.gate" in out
    assert "forcing_modules.ERA5_FORCING.embed.mlp.0.weight" in out
    assert not [k for k in out if k.startswith("forcing_module.")]
    # the trained value must survive, not just the name
    assert out["forcing_modules.ERA5_FORCING.gate"].item() == pytest.approx(0.0996)
    # unrelated keys untouched
    assert "encoder.something.weight" in out


def test_ambiguous_target_refuses_rather_than_guessing():
    """Loading a trained forcing onto the wrong stream would be silent and wrong."""
    with pytest.raises(RuntimeError, match="cannot be inferred"):
        _migrate_legacy_forcing_keys(dict(LEGACY), _Model(["ERA5_FORCING", "ERA5_ANCHOR"]), "x")


def test_modern_checkpoint_is_untouched():
    modern = {"forcing_modules.ERA5_FORCING.gate": torch.tensor([1.0])}
    assert _migrate_legacy_forcing_keys(dict(modern), _Model(["ERA5_FORCING"]), "x") == modern


def test_no_injection_configured_leaves_keys_alone():
    """Previous behaviour: the keys stay unused rather than erroring."""
    out = _migrate_legacy_forcing_keys(dict(LEGACY), _Model([]), "x")
    assert "forcing_module.gate" in out


def test_ddp_wrapped_model_is_unwrapped():
    class _DDP:
        def __init__(self, inner):
            self.module = inner

    out = _migrate_legacy_forcing_keys(
        dict(LEGACY), _DDP(_Model(["ERA5_FORCING"])), "x"
    )
    assert "forcing_modules.ERA5_FORCING.gate" in out
