"""The shipped injection configuration must actually reach the forecasting engine.

`type: anemoi` + `forcing: True` and `type: forcing` both describe a prescribed boundary
condition, but only the second is routed to ForcingInjection. A config using the first
still parses, still trains and still produces plausible output - while building no
injection module at all, which in a rollout means the boundary condition is frozen
at initialisation for the entire run. Nothing downstream reports it.
"""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parent.parent / "config"
STREAMS = CONFIG_DIR / "streams" / "era5_1deg_injection"

# what model.py requires before it will build an injection module
INJECTED = {"ERA5_FORCING": ["sst", "ci", "sst_anom"], "ERA5_ANCHOR": ["2t_clim"]}


def _streams() -> dict:
    merged = {}
    for f in sorted(STREAMS.glob("*.yml")):
        merged.update(OmegaConf.to_container(OmegaConf.load(f), resolve=True))
    return merged


def test_injection_config_parses():
    cfg = OmegaConf.load(CONFIG_DIR / "config_injection.yml")
    assert cfg.streams_directory == "./config/streams/era5_1deg_injection/"
    assert cfg.rollout.mode == "physical"
    assert cfg.rollout.latent.reencode_every == 0


@pytest.mark.parametrize("name,channels", INJECTED.items())
def test_forcing_stream_is_injected_not_assimilated(name, channels):
    stream = _streams()[name]
    # the exact condition model.py gates on
    assert stream["type"] == "forcing", (
        f"{name} has type={stream['type']!r}; only 'forcing' builds a ForcingInjection. "
        "'anemoi' + forcing:True is the encode-once pattern and is silently inert here."
    )
    assert stream["injection"]["mode"] != "none"
    assert stream["source"] == channels


def test_gate_init_asymmetry_between_ocean_and_anchor():
    """The two forcings want opposite initialisations, for opposite reasons.

    The ocean carries information the model cannot obtain anywhere else, so a gate starting
    at no influence risks the optimiser never finding it -> 1.0. The anchor carries
    information that overlaps with what the model should be learning to predict, so full
    strength from step one makes reading the anchor the cheapest way to cut early 2t loss,
    before any dynamics exist to compete -> 0.0.
    """
    streams = _streams()
    assert streams["ERA5_FORCING"]["injection"]["gate_init"] == 1.0
    assert streams["ERA5_ANCHOR"]["injection"]["gate_init"] == 0.0


def test_anchor_uses_the_compliant_sstreg_field():
    """TRENDCLIM prescribes 2t directly and pins the response it is meant to measure."""
    streams = _streams()
    src = str(streams["ERA5_ANCHOR"]["filenames"])
    assert "ANCHOR-SSTREG" in src and "TRENDCLIM" not in src


def test_anchor_and_ocean_are_separate_modules():
    """A shared gate could not be ablated or rescaled per forcing."""
    streams = _streams()
    assert streams["ERA5_FORCING"]["stream_id"] != streams["ERA5_ANCHOR"]["stream_id"]
    assert "2t_clim" not in streams["ERA5_FORCING"]["source"]


def test_tendency_channels_are_excluded():
    """Measured dR^2 <= 0.0014 on top of sst_anom; they were deliberately dropped."""
    source = _streams()["ERA5_FORCING"]["source"]
    assert not [c for c in source if "tend" in c]


def test_no_pushforward_at_eight_steps():
    """Pushforward at num_steps 8 is what collapsed sub-monthly variability to ~40%."""
    fc = OmegaConf.load(CONFIG_DIR / "config_injection.yml").training_config.forecast
    assert fc.num_steps == 8
    assert fc.pushforward is False
    assert fc.timestep_weight.gamma_decay.decay_factor < 1.0
