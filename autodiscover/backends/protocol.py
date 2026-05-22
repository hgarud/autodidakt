"""Backend protocol surface.

Any backend (Tinker, MLX-local, ...) provides a ``SamplingClient`` and a
``TrainingClient``. The trainer server's ``TrainingLoop`` and the
``sample_plans`` coroutine consume only this surface.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from autodiscover.backends.types import (
    SampledSequence,
    SamplingParams,
    TokenSequence,
    TrainingDatum,
)


@runtime_checkable
class SamplingClient(Protocol):
    async def sample(
        self,
        prompt: TokenSequence,
        params: SamplingParams,
        num_samples: int = 1,
    ) -> list[SampledSequence]:
        """Draw ``num_samples`` continuations from ``prompt``."""
        ...

    async def compute_logprobs(
        self,
        sequences: list[TokenSequence],
    ) -> list[list[float]]:
        """Compute the per-token logprobs of each full sequence under the
        current policy. Length matches sequence length (first token's
        logprob is conventionally 0)."""
        ...


@runtime_checkable
class TrainingClient(Protocol):
    async def forward_backward(
        self,
        batch: list[TrainingDatum],
        loss_fn: str,  # "importance_sampling" | "ppo"
    ) -> dict[str, float]:
        """Run one forward/backward pass. Gradients are accumulated by the
        backend; ``optim_step`` applies them. Returns a metrics dict
        (e.g. ``{"loss": ...}``)."""
        ...

    async def optim_step(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        ...

    async def save_checkpoint(self, name: str) -> str:
        """Persist current LoRA weights. Returns a string identifier
        (path or URI) the backend can later resolve."""
        ...

    async def get_post_training_sampling_client(self) -> SamplingClient:
        """Return a SamplingClient bound to the *current* policy weights
        (after the most recent optim_step). Called once per batch by
        TrainingLoop; the returned client is what /rollout/begin uses
        next."""
        ...
