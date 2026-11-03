"""Several prescribed forcings must reach the forecasting engine independently.

The failure these guard against is silent: a single per-step forcing slot let the last
forcing stream overwrite the others, so a run configured with SST *and* a climatology
anchor would quietly inject only the anchor and still train, converge and produce
plausible output.
"""

import torch

from weathergen.model.forcing import ForcingInjection

NUM_CELLS, DIM_EMBED = 32, 16


def _field(num_vars: int, fill: float) -> torch.Tensor:
    """(num_cells, 2*num_vars) = [values | validity], all cells valid."""
    values = torch.full((NUM_CELLS, num_vars), fill)
    valid = torch.ones(NUM_CELLS, num_vars)
    return torch.cat([values, valid], dim=-1)


class _StubModel:
    """Carries just what Model._inject_forcings touches."""

    from weathergen.model.model import Model

    _inject_forcings = Model._inject_forcings

    def __init__(self, modules, num_aux=0):
        self.forcing_modules = torch.nn.ModuleDict(modules)
        self.num_aux_tokens = num_aux


class _StubBatch:
    def __init__(self, forcing):
        self.forcing = forcing


def test_gate_init_is_honoured():
    assert ForcingInjection(2, DIM_EMBED).gate.item() == 0.0
    assert ForcingInjection(2, DIM_EMBED, gate_init=1.0).gate.item() == 1.0


def test_reset_parameters_restores_configured_gate():
    inj = ForcingInjection(2, DIM_EMBED, gate_init=1.0)
    with torch.no_grad():
        inj.gate.fill_(0.42)
    inj.reset_parameters()
    assert inj.gate.item() == 1.0


def test_both_forcings_contribute():
    """Injecting two streams must differ from injecting either one alone."""
    torch.manual_seed(0)
    ocean = ForcingInjection(3, DIM_EMBED, gate_init=1.0)
    anchor = ForcingInjection(1, DIM_EMBED, gate_init=1.0)
    tokens = torch.zeros(1, NUM_CELLS, DIM_EMBED)
    fields = {"OCEAN": _field(3, 0.5), "ANCHOR": _field(1, -0.5)}

    both = _StubModel({"OCEAN": ocean, "ANCHOR": anchor})._inject_forcings(
        tokens, _StubBatch([fields]), 0
    )
    only_ocean = _StubModel({"OCEAN": ocean})._inject_forcings(tokens, _StubBatch([fields]), 0)
    only_anchor = _StubModel({"ANCHOR": anchor})._inject_forcings(tokens, _StubBatch([fields]), 0)

    assert not torch.allclose(both, only_ocean), "anchor contribution lost"
    assert not torch.allclose(both, only_anchor), "ocean contribution lost"
    # additive and independent: the two contributions must sum
    assert torch.allclose(both, only_ocean + only_anchor, atol=1e-6)


def test_missing_stream_is_a_noop_not_a_crash():
    """A step with no data for one stream must not disturb the others or raise."""
    ocean = ForcingInjection(3, DIM_EMBED, gate_init=1.0)
    anchor = ForcingInjection(1, DIM_EMBED, gate_init=1.0)
    tokens = torch.zeros(1, NUM_CELLS, DIM_EMBED)
    model = _StubModel({"OCEAN": ocean, "ANCHOR": anchor})

    partial = model._inject_forcings(tokens, _StubBatch([{"OCEAN": _field(3, 0.5)}]), 0)
    ocean_only = _StubModel({"OCEAN": ocean})._inject_forcings(
        tokens, _StubBatch([{"OCEAN": _field(3, 0.5)}]), 0
    )
    assert torch.allclose(partial, ocean_only)
    # no forcing at all, and a step past the end, are both no-ops
    assert torch.allclose(model._inject_forcings(tokens, _StubBatch([{}]), 0), tokens)
    assert torch.allclose(model._inject_forcings(tokens, _StubBatch([]), 0), tokens)


def test_aux_tokens_are_untouched():
    num_aux = 2
    anchor = ForcingInjection(1, DIM_EMBED, gate_init=1.0)
    tokens = torch.randn(1, num_aux + NUM_CELLS, DIM_EMBED)
    out = _StubModel({"ANCHOR": anchor}, num_aux=num_aux)._inject_forcings(
        tokens, _StubBatch([{"ANCHOR": _field(1, 0.5)}]), 0
    )
    assert torch.allclose(out[:, :num_aux], tokens[:, :num_aux])
    assert not torch.allclose(out[:, num_aux:], tokens[:, num_aux:])


def test_gate_survives_meta_device_materialisation():
    """From-scratch FSDP puts the model on `meta`, so constructor values are never stored.

    Model.reset_parameters resets only Linear and LayerNorm, so a bare nn.Parameter such as
    the injection gate keeps whatever to_empty() allocated. An uninitialised gate multiplies
    the injection to nothing: the forcing is silently absent, while the embedding MLPs -
    being Linear - are reset correctly and look healthy. A single-GPU run never sees this,
    because it skips the meta device entirely.
    """
    from weathergen.model.model import Model

    class _M(torch.nn.Module):
        reset_parameters = Model.reset_parameters

        def __init__(self):
            super().__init__()
            self.forcing_modules = torch.nn.ModuleDict(
                {
                    "OCEAN": ForcingInjection(3, DIM_EMBED, gate_init=1.0),
                    "ANCHOR": ForcingInjection(1, DIM_EMBED, gate_init=1.0),
                }
            )

    with torch.device("meta"):
        m = _M()
    m.to_empty(device="cpu")

    # the failure mode: whatever to_empty allocated, almost never the configured value
    m.reset_parameters()

    for name in ("OCEAN", "ANCHOR"):
        gate = m.forcing_modules[name].gate.item()
        assert gate == 1.0, f"{name} gate is {gate}, not the configured gate_init 1.0"
        # the normalisation buffers must come back as identity, not garbage
        assert torch.equal(m.forcing_modules[name].embed.stdev,
                           torch.ones_like(m.forcing_modules[name].embed.stdev))


def test_startup_assertion_catches_uninitialised_gates():
    """The run must refuse to start rather than train with a silently absent forcing."""
    import pytest

    from weathergen.model.model_interface import _assert_forcing_gates_initialised

    class _M:
        def __init__(self, gates):
            self.forcing_modules = torch.nn.ModuleDict(
                {
                    n: ForcingInjection(1, DIM_EMBED, gate_init=init)
                    for n, (init, _) in gates.items()
                }
            )
            for n, (_, actual) in gates.items():
                with torch.no_grad():
                    self.forcing_modules[n].gate.fill_(actual)

    # configured 1.0, actually uninitialised memory -> must raise
    with pytest.raises(RuntimeError, match="not initialised to their configured"):
        _assert_forcing_gates_initialised(_M({"OCEAN": (1.0, 3e-5)}))

    # gate_init 0.0 is legitimate for an anchor and must NOT trip the check
    _assert_forcing_gates_initialised(_M({"ANCHOR": (0.0, 0.0), "OCEAN": (1.0, 1.0)}))

    # ...but uninitialised memory that happens to be near zero must still be caught when
    # the configured value was 1.0
    with pytest.raises(RuntimeError, match="OCEAN"):
        _assert_forcing_gates_initialised(_M({"ANCHOR": (0.0, 0.0), "OCEAN": (1.0, 1e-5)}))
