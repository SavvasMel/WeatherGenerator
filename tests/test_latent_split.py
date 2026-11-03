"""``forward`` must stay exactly ``encode_to_latent`` followed by ``advance_latent``.

The latent rollout encodes once and then advances the latent repeatedly, so it calls the
two halves directly. If ``forward`` ever diverges from their composition, training and
rollout silently stop being the same model.
"""

import types

import torch

from weathergen.model.model import Model, ModelOutput

NCELL, DIM, BS, NSRC = 12, 8, 2, 2


class _StubBatch:
    def __init__(self, idxs):
        self.output_idxs = idxs
        self.forcing = [{} for _ in range(max(idxs) + 1)]

    def __len__(self):
        return BS

    def get_output_len(self):
        return len(self.output_idxs)

    def get_num_source_steps(self):
        return NSRC

    def get_output_idxs(self):
        return self.output_idxs


class _Harness:
    """Stubs every component ``forward`` touches, and records the call sequence."""

    forward = Model.forward
    encode_to_latent = Model.encode_to_latent
    advance_latent = Model.advance_latent

    def __init__(self, pushforward=False, training=False):
        self.log = []
        self.num_aux_tokens = 0
        self.training = training
        self.forcing_modules = {}
        self.cf = types.SimpleNamespace(
            training_config={"forecast": {"pushforward": pushforward}}
        )

    def encoder(self, model_params, batch):
        self.log.append("encode")
        g = torch.Generator().manual_seed(7)
        return torch.randn(BS * NSRC, NCELL, DIM, generator=g), "POSTERIORS"

    def _inject_forcings(self, tokens, batch, step):
        self.log.append(f"inject{step}")
        return tokens + 0.5

    def forecast_engine(self, tokens, step, coords):
        self.log.append(f"fe{step}")
        return tokens * 1.1 + step

    def predict_decoders(self, model_params, step, tokens, batch, output):
        self.log.append(f"dec{step}")
        output.add_physical_prediction(step, "S", tokens.sum())
        return output

    def predict_latent(self, model_params, step, tokens, batch, output):
        self.log.append(f"lat{step}")
        output.add_latent_prediction(step, "L", tokens.sum())
        return output


def _params():
    return types.SimpleNamespace(rope_coords=None)


def test_forward_equals_encode_then_advance():
    for pushforward in (False, True):
        for training in (False, True):
            for idxs in ([0], [0, 1, 2], [0, 1, 2, 3, 4, 5, 6, 7]):
                a = _Harness(pushforward, training)
                out_a = a.forward(_params(), _StubBatch(idxs))

                b = _Harness(pushforward, training)
                batch = _StubBatch(idxs)
                output = ModelOutput(batch.get_output_len())
                tokens, output = b.encode_to_latent(_params(), batch, output)
                _tokens, out_b = b.advance_latent(_params(), batch, tokens, output)

                msg = f"pushforward={pushforward} training={training} steps={len(idxs)}"
                assert a.log == b.log, msg
                assert str(out_a.__dict__) == str(out_b.__dict__), msg


def test_advance_latent_returns_a_carryable_state():
    """The advanced latent must come back so a caller can feed it in again."""
    h = _Harness()
    batch = _StubBatch([0, 1])
    tokens, _ = h.encode_to_latent(_params(), batch)
    advanced, _ = h.advance_latent(_params(), batch, tokens)
    assert advanced.shape == tokens.shape
    assert not torch.allclose(advanced, tokens), "latent did not advance"


def test_decode_false_skips_decoders_but_still_advances():
    """Latent rollout advances many steps between writes; decoding every step would be wasted."""
    h = _Harness()
    batch = _StubBatch([0, 1, 2])
    tokens, _ = h.encode_to_latent(_params(), batch)
    advanced, _ = h.advance_latent(_params(), batch, tokens, decode=False)
    assert [c for c in h.log if c.startswith(("dec", "lat"))] == []
    assert [c for c in h.log if c.startswith("fe")] == ["fe0", "fe1", "fe2"]
    assert advanced.shape == tokens.shape
