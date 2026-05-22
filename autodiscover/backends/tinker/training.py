"""TinkerTrainingClient: adapter over ``tinker.TrainingClient`` + backend factory.

Conforms structurally to ``autodiscover.backends.protocol.TrainingClient``.
"""
from __future__ import annotations

from typing import cast

import tinker
import torch
from tinker import TensorData

from autodiscover.backends.protocol import SamplingClient, TrainingClient
from autodiscover.backends.tinker.sampling import TinkerSamplingClient
from autodiscover.backends.types import TrainingDatum


# ---------------------------------------------------------------------------
# Conversions (training side)
# ---------------------------------------------------------------------------


def _datum_to_tinker(d: TrainingDatum) -> tinker.Datum:
    return tinker.Datum(
        model_input=tinker.ModelInput(
            chunks=[tinker.types.EncodedTextChunk(tokens=d.input_tokens)],
        ),
        loss_fn_inputs={
            "target_tokens": TensorData.from_torch(torch.tensor(d.target_tokens)),
            "logprobs":      TensorData.from_torch(torch.tensor(d.old_logprobs)),
            "advantages":    TensorData.from_torch(torch.tensor(d.advantages)),
            "mask":          TensorData.from_torch(torch.tensor(d.mask)),
        },
    )


def _strip_mask(d: tinker.Datum) -> tinker.Datum:
    """Tinker requires ``mask`` to be present when constructing a Datum, but
    ``forward_backward_async`` does not expect it (matches the behavior of
    the deleted ``autodiscover.training.step.remove_mask``)."""
    return tinker.Datum(
        model_input=d.model_input,
        loss_fn_inputs={k: v for k, v in d.loss_fn_inputs.items() if k != "mask"},
    )


# ---------------------------------------------------------------------------
# Training client
# ---------------------------------------------------------------------------


class TinkerTrainingClient:
    def __init__(
        self,
        inner: tinker.TrainingClient,
        log_dir: str,
        save_every: int,
    ) -> None:
        self._inner = inner
        self._log_dir = log_dir
        self._save_every = save_every
        self._batch_count = 0
        # The pending forward_backward future, to be awaited as part of
        # optim_step. Mirrors how step.py interleaves enqueue/consume.
        self._pending_fb: tinker.APIFuture | None = None

    async def forward_backward(
        self,
        batch: list[TrainingDatum],
        loss_fn: str,
    ) -> dict[str, float]:
        tinker_data = [_strip_mask(_datum_to_tinker(d)) for d in batch]
        # cast to tinker.types.LossFnType — at runtime it's just a str.
        fb_future = await self._inner.forward_backward_async(
            tinker_data, loss_fn=cast("tinker.types.LossFnType", loss_fn),
        )
        result = await fb_future.result_async()
        # Return any scalar metrics tinker reports. Be defensive — schema
        # has varied across SDK versions.
        metrics: dict[str, float] = {}
        for k, v in getattr(result, "metrics", {}).items():
            try:
                metrics[str(k)] = float(v)
            except (TypeError, ValueError):
                pass
        return metrics

    async def optim_step(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        adam = tinker.AdamParams(
            learning_rate=learning_rate, beta1=beta1, beta2=beta2, eps=eps,
        )
        fut = await self._inner.optim_step_async(adam)
        await fut.result_async()
        self._batch_count += 1

    async def save_checkpoint(self, name: str) -> str:
        # Defer to existing helper so behavior matches today's checkpoint
        # layout exactly.
        from autodiscover.checkpointing import save_checkpoint_async
        path_dict = await save_checkpoint_async(
            training_client=self._inner,
            name=name,
            log_path=self._log_dir,
            loop_state={"batch": self._batch_count},
            kind="both",
        )
        return path_dict["sampler_path"]

    async def get_post_training_sampling_client(self) -> SamplingClient:
        # Match the legacy logic of step.py:save_checkpoint_and_get_sampling_client
        if (
            self._save_every > 0
            and self._batch_count > 0
            and self._batch_count % self._save_every == 0
        ):
            sampler_path = await self.save_checkpoint(f"{self._batch_count:06d}")
            inner = self._inner.create_sampling_client(sampler_path)
        else:
            inner = await self._inner.save_weights_and_get_sampling_client_async()
        return TinkerSamplingClient(inner)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


async def make_tinker_backend(
    *,
    model_name: str,
    lora_rank: int,
    log_dir: str,
    save_every: int,
) -> tuple[SamplingClient, SamplingClient, TrainingClient]:
    """Returns (base_sampling, sampling, training) — all protocol-typed."""
    service_client = tinker.ServiceClient(base_url=None)
    base_inner = service_client.create_sampling_client(base_model=model_name)
    training_inner = await service_client.create_lora_training_client_async(
        model_name, rank=lora_rank,
    )
    sampling_inner = await training_inner.save_weights_and_get_sampling_client_async()
    return (
        TinkerSamplingClient(base_inner),
        TinkerSamplingClient(sampling_inner),
        TinkerTrainingClient(training_inner, log_dir=log_dir, save_every=save_every),
    )
