# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import dataclasses
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from weathergen.common.config import Config
from weathergen.datasets.batch import ModelBatch
from weathergen.model.embeddings import StreamEmbedLinear, StreamEmbedTransformer
from weathergen.model.layers import MLP
from weathergen.utils.utils import get_dtype


@dataclasses.dataclass
class ConditioningData:
    embeddings: dict[str, torch.Tensor]
    dim_embed: int

    def get(self, stream_name: str) -> torch.Tensor | None:
        return self.embeddings.get(stream_name)

    def merge(self, other: "ConditioningData") -> "ConditioningData":
        breakpoint()
        combined = {**self.embeddings, **other.embeddings}
        dim = max(self.dim_embed, other.dim_embed)
        return ConditioningData(embeddings=combined, dim_embed=dim)

    def is_empty(self) -> bool:
        return len(self.embeddings) == 0


class ConditioningEmbedder(ABC):
    @abstractmethod
    def embed(self, batch: ModelBatch, step: int) -> ConditioningData:
        pass

    @abstractmethod
    def get_dim_embed(self) -> int:
        pass

    @abstractmethod
    def reset(self) -> None:
        pass


class ScalarConditioningEmbedder(ConditioningEmbedder, nn.Module):
    def __init__(
        self,
        cf: Config,
        stream_names: list[str],
        device: torch.device | None = None,
    ):
        super().__init__()
        self.cf = cf
        self.stream_names = stream_names
        self.device = device
        self.dim_embed = cf.ae_global_dim_embed

        self.embeds = nn.ModuleDict()
        self.input_dims: dict[str, int] = {}
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False

    def get_dim_embed(self) -> int:
        return self.dim_embed

    def _lazy_init(self, batch: ModelBatch) -> None:
        if self._initialized:
            return

        for stream_name in self.stream_names:
            si = self.cf.streams.get(stream_name)
            if si is None:
                continue

            values = batch.get_scalar_conditioning_values(stream_name, step=0)
            if values is None:
                continue

            breakpoint()

            self.input_dims[stream_name] = values.shape[1]
            self.dim_embed = si.get("embed", {}).get("dim_embed", self.dim_embed)
            net = si.get("embed", {}).get("net", "linear")

            if net == "linear":
                self.embeds[stream_name] = nn.Linear(
                    self.input_dims[stream_name],
                    self.dim_embed,
                )
            elif net == "mlp":
                self.embeds[stream_name] = MLP(
                    self.input_dims[stream_name],
                    self.dim_embed,
                    hidden_factor=4,
                    with_residual=False,
                    dropout_rate=self.cf.get("embed_dropout_rate", 0.0),
                )
            else:
                raise ValueError(f"Unknown embed net for scalar conditioning: {net}")

        self._initialized = True

    def embed(self, batch: ModelBatch, step: int) -> ConditioningData:
        self._lazy_init(batch)

        breakpoint()
        embeddings = {}
        for stream_name in self.stream_names:
            if stream_name not in self.embeds:
                continue

            breakpoint()
            values = batch.get_scalar_conditioning_values(stream_name, step)
            if values is None:
                continue

            embeddings[stream_name] = self.embeds[stream_name](values)

        return ConditioningData(embeddings=embeddings, dim_embed=self.dim_embed)


class FieldConditioningEmbedder(ConditioningEmbedder, nn.Module):
    def __init__(
        self,
        cf: Config,
        stream_names: list[str],
        device: torch.device | None = None,
    ):
        super().__init__()
        self.cf = cf
        self.stream_names = stream_names
        self.device = device
        self.dim_embed = cf.ae_global_dim_embed

        self.embeds = nn.ModuleDict()
        self.source_sizes: dict[str, int] = {}
        self._initialized = False

    def get_dim_embed(self) -> int:
        return self.dim_embed

    def reset(self) -> None:
        self._initialized = False

    def _lazy_init(self, batch: ModelBatch) -> None:
        if self._initialized:
            return

        for stream_name in self.stream_names:
            si = self.cf.streams.get(stream_name)
            if si is None:
                continue

            stream_data = batch.conditioning_samples.streams_data.get(stream_name)
            if stream_data is None:
                continue

            self.source_sizes[stream_name] = stream_data.data.shape[-1]
            self.dim_embed = si.get("embed", {}).get("dim_embed", self.dim_embed)

            encoder_cfg = si.get("encoder", {})
            embed_cfg = si.get("embed", {})

            if encoder_cfg.get("model_id") is not None:
                self._load_encoder_from_checkpoint(stream_name, encoder_cfg)
            else:
                self._create_embed_network(stream_name, si, embed_cfg)

        self._initialized = True

    def _load_encoder_from_checkpoint(
        self,
        stream_name: str,
        encoder_cfg: dict,
    ) -> None:
        model_id = encoder_cfg["model_id"]
        model_epoch = encoder_cfg.get("model_epoch")
        mini_epoch_id = (
            f"chkpt{model_epoch:05d}"
            if model_epoch is not None and model_epoch != -1
            else "latest"
        )
        filename = f"{model_id}_{mini_epoch_id}.chkpt"

        from weathergen.utils.utils import get_path_model

        path_run = (
            Path(self.cf.get("model_path", get_path_model(run_id=model_id))) / model_id
        )
        params = torch.load(
            path_run / filename, map_location="cpu", mmap=True, weights_only=True
        )

        embedder_params = {
            k: v
            for k, v in params.items()
            if k.startswith(f"encoder.embed_engine.embeds.{stream_name}.")
        }

        self.embeds[stream_name] = nn.Linear(
            self.source_sizes[stream_name], self.dim_embed
        )
        dummy_state = self.embeds[stream_name].state_dict()
        mapped_params = {}
        for old_k, v in embedder_params.items():
            new_k = old_k.replace(f"encoder.embed_engine.embeds.{stream_name}.", "")
            if new_k in dummy_state:
                mapped_params[new_k] = v
        self.embeds[stream_name].load_state_dict(mapped_params, strict=False)

    def _create_embed_network(
        self,
        stream_name: str,
        si: dict,
        embed_cfg: dict,
    ) -> None:
        net = embed_cfg.get("net", "linear")

        if net == "linear":
            token_size = si.get("token_size", 1)
            self.embeds[stream_name] = StreamEmbedLinear(
                self.source_sizes[stream_name] * token_size,
                self.dim_embed,
                stream_name=stream_name,
            )
        elif net == "transformer":
            self.embeds[stream_name] = StreamEmbedTransformer(
                num_tokens=embed_cfg.get("num_tokens", 1),
                token_size=si.get("token_size", 1),
                num_channels=self.source_sizes[stream_name],
                dim_embed=embed_cfg.get("dim_embed", self.dim_embed),
                dim_out=self.cf.ae_local_dim_embed,
                num_blocks=embed_cfg.get("num_blocks", 2),
                num_heads=embed_cfg.get("num_heads", 8),
                dropout_rate=self.cf.get("embed_dropout_rate", 0.0),
                norm_type=self.cf.norm_type,
                unembed_mode=self.cf.get("embed_unembed_mode", "full"),
                stream_name=stream_name,
            )
        else:
            raise ValueError(f"Unknown embed net for field conditioning: {net}")

    def embed(self, batch: ModelBatch, step: int) -> ConditioningData:
        self._lazy_init(batch)

        embeddings = {}
        for stream_name in self.stream_names:
            if stream_name not in self.embeds:
                continue

            stream_data = batch.conditioning_samples.streams_data.get(stream_name)
            if stream_data is None:
                continue

            x = stream_data.data.to(
                dtype=get_dtype(self.cf.mixed_precision_dtype),
                device=self.device or "cpu",
            )
            x = checkpoint(self.embeds[stream_name], x, use_reentrant=False)
            embeddings[stream_name] = x

        return ConditioningData(embeddings=embeddings, dim_embed=self.dim_embed)


class ConditioningEmbedder(nn.Module):
    def __init__(self, cf: Config, device: torch.device | None = None):
        super().__init__()
        self.cf = cf
        self.device = device
        self.dim_embed = cf.ae_global_dim_embed

        self.scalar_embedder: ScalarConditioningEmbedder | None = None
        self.field_embedder: FieldConditioningEmbedder | None = None

        scalar_streams = []
        field_streams = []

        for stream_name, si in cf.streams.items():
            timestep_cond = si.get("timestep_conditioning")
            if timestep_cond == "scalar":
                scalar_streams.append(stream_name)
            elif timestep_cond == "field":
                field_streams.append(stream_name)

        if scalar_streams:
            self.scalar_embedder = ScalarConditioningEmbedder(
                cf, scalar_streams, device
            )

        if field_streams:
            self.field_embedder = FieldConditioningEmbedder(cf, field_streams, device)

    def embed(self, batch: ModelBatch, step: int) -> ConditioningData:
        result = ConditioningData(embeddings={}, dim_embed=self.dim_embed)

        if self.scalar_embedder is not None:
            scalar_data = self.scalar_embedder.embed(batch, step)
            result = result.merge(scalar_data)

        if self.field_embedder is not None:
            field_data = self.field_embedder.embed(batch, step)
            result = result.merge(field_data)

        return result

    def get_dim_embed(self) -> int:
        return self.dim_embed
