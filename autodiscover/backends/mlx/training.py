"""Local training client backed by in-process MLX.

This file owns:
- Loading the base model + tokenizer into MLX.
- Wrapping the target linear layers with LoRA.
- Holding the optimizer state.
- A toggle to compute base-model logprobs with LoRA disabled.
- Optim step, checkpoint save, and sampler-client rotation.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import httpx
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.tuner.utils import (
    linear_to_lora_layers,
    print_trainable_parameters,
)
from mlx_lm.utils import load

from autodiscover.backends.types import (
    SamplingParams,
    TokenSequence,
    TrainingDatum,
)

if TYPE_CHECKING:
    from autodiscover.backends.protocol import SamplingClient


_PPO_CLIP_EPS = 0.2


def _gather_logprobs(logits: mx.array, targets: mx.array) -> mx.array:
    """Given logits (T, V) and targets (T,), return per-step logprobs (T,)."""
    logp = nn.log_softmax(logits, axis=-1)
    # mx doesn't have torch.gather; advanced indexing works:
    idx = mx.arange(logp.shape[0])
    return logp[idx, targets]


def _datum_loss(
    model,
    d: TrainingDatum,
    loss_fn: str,
) -> tuple[mx.array, float]:
    """Returns (loss_scalar, mass) where mass = sum(mask) -- used for
    weighting across data in a batch."""
    inp = mx.array(d.input_tokens, dtype=mx.int32)[None, :]      # (1, T)
    targets = mx.array(d.target_tokens, dtype=mx.int32)          # (T,)
    old_lp = mx.array(d.old_logprobs, dtype=mx.float32)
    adv = mx.array(d.advantages, dtype=mx.float32)
    mask = mx.array(d.mask, dtype=mx.float32)
    mass = float(mask.sum().item())
    if mass == 0:
        return mx.array(0.0), 0.0

    logits = model(inp)                                          # (1, T, V)
    logits = logits[0]                                           # (T, V)
    new_lp = _gather_logprobs(logits, targets)                   # (T,)

    ratio = mx.exp(new_lp - old_lp)
    if loss_fn == "importance_sampling":
        per_token = ratio * adv
    elif loss_fn == "ppo":
        clipped = mx.clip(ratio, 1.0 - _PPO_CLIP_EPS, 1.0 + _PPO_CLIP_EPS)
        per_token = mx.minimum(ratio * adv, clipped * adv)
    else:
        raise ValueError(f"unknown loss_fn: {loss_fn}")

    masked = per_token * mask
    loss = -(masked.sum() / mass)
    return loss, mass


_DEFAULT_LORA_TARGETS = [
    # Linear projections inside each transformer block. Exact names depend
    # on the MLX gpt-oss implementation. The pattern below matches
    # mlx_lm's default LoRA targets; adjust per the model card if needed.
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]


class MlxTrainingClient:
    """Local training client. forward_backward / optim_step / checkpoint
    are added in subsequent steps; this step only handles construction.
    """

    def __init__(
        self,
        *,
        model_name: str,
        lora_rank: int,
        adapter_dir: Path,
        save_every: int,
        log_dir: str,
        learning_rate: float = 1e-5,
    ) -> None:
        self._model_name = model_name
        self._lora_rank = lora_rank
        self._adapter_dir = adapter_dir
        self._save_every = save_every
        self._log_dir = log_dir

        # Load the base model. mlx_lm.utils.load handles fetching MXFP4
        # weights from Hugging Face (or a local snapshot for air-gapped
        # use; pre-stage the snapshot in the customer environment).
        self._model, self._tokenizer = load(model_name)

        # Freeze the base model. Only LoRA params will be trainable.
        self._model.freeze()

        lora_config: dict[str, Any] = {
            "rank": lora_rank,
            "scale": 20.0,           # standard LoRA scale
            "dropout": 0.0,
            "keys": _DEFAULT_LORA_TARGETS,
        }
        # Wrap target linear layers with LoRA. ``num_layers=-1`` means all.
        # API checked against mlx-lm >= 0.20.0.
        linear_to_lora_layers(self._model, num_layers=-1, config=lora_config)
        # Quick sanity print for the first run -- comment out in prod if
        # noisy.
        print_trainable_parameters(self._model)

        # AdamW only on trainable (LoRA) params. Defaults match Tinker's
        # tinker.AdamParams.
        self._optimizer = optim.AdamW(
            learning_rate=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        # Materialize parameter arrays.
        mx.eval(self._model.parameters())

        self._batch_count = 0
        # Gradients produced by forward_backward; consumed by optim_step.
        # ``None`` until the first forward_backward call.
        self._pending_grads: Any = None
        # Sampler URL stashed by ``make_mlx_backend`` for adapter-reload
        # requests in ``get_post_training_sampling_client``. ``None`` in
        # tests / when no sampler is wired up.
        self._sampler_url: str | None = None
        # Concurrency cap for sampler HTTP clients minted by
        # ``get_post_training_sampling_client``. Stashed by
        # ``make_mlx_backend`` so post-training rotations honour the
        # same limit as the initial client.
        self._sampler_max_concurrency: int = 1

    # --- helpers used by later steps ---

    def model(self):
        """The underlying MLX model (LoRA enabled by default)."""
        return self._model

    def tokenizer(self):
        return self._tokenizer

    def optimizer(self):
        return self._optimizer

    def trainable_parameters(self):
        return self._model.trainable_parameters()

    def adapter_dir(self) -> Path:
        return self._adapter_dir

    def save_every(self) -> int:
        return self._save_every

    def log_dir(self) -> str:
        return self._log_dir

    # --- forward / backward ---

    async def forward_backward(
        self,
        batch: list[TrainingDatum],
        loss_fn: str,
    ) -> dict[str, float]:
        """Compute loss + accumulate gradients into the optimizer state.

        Returns metrics (loss_mean).
        """
        if not batch:
            return {}

        total_mass = sum(float(np.array(d.mask).sum()) for d in batch)
        if total_mass == 0:
            return {"loss": 0.0}

        model = self._model

        # Weight each datum by its mass / total_mass so the gradient is
        # the average across masked tokens of the whole batch.
        def loss_closure(_params):
            total = mx.array(0.0)
            for d in batch:
                per, mass = _datum_loss(model, d, loss_fn)
                if mass == 0:
                    continue
                total = total + per * (mass / total_mass)
            return total

        loss_and_grad = nn.value_and_grad(model, loss_closure)
        loss_value, grads = loss_and_grad(model.trainable_parameters())
        # Cache grads for optim_step (Step 14).
        self._pending_grads = grads
        # Evaluate to materialize.
        mx.eval(loss_value, grads)
        return {"loss": float(loss_value.item())}

    # --- optim step / checkpoint / rotation ---

    async def optim_step(
        self,
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.95,
        eps: float = 1e-8,
    ) -> None:
        """Apply accumulated gradients to LoRA params and clear them.

        Hyperparameters (lr, betas, eps) are set at construction. They are
        constant across batches in the current code path, and MLX's AdamW
        does not document settable ``betas``/``eps`` properties. If you
        need an LR schedule in the future, pass a schedule callable to
        ``mlx.optimizers.AdamW(learning_rate=...)`` at construction.
        """
        # Sanity guard so future callers don't silently change hyperparams.
        # ``self._optimizer.learning_rate`` may be an ``mx.array`` scalar;
        # coerce to float for the comparison.
        current_lr = float(
            self._optimizer.learning_rate.item()
            if hasattr(self._optimizer.learning_rate, "item")
            else self._optimizer.learning_rate
        )
        if (
            learning_rate != current_lr
            or beta1 != 0.9
            or beta2 != 0.95
            or eps != 1e-8
        ):
            raise NotImplementedError(
                "MlxTrainingClient.optim_step does not honor per-step "
                "hyperparameter changes. Set them at construction or "
                "introduce a schedule (see docstring)."
            )

        grads = getattr(self, "_pending_grads", None)
        if grads is None:
            return  # nothing to do
        self._optimizer.update(self._model, grads)
        mx.eval(self._model.parameters(), self._optimizer.state)
        self._pending_grads = None
        self._batch_count += 1

    async def save_checkpoint(self, name: str) -> str:
        """Persist the current LoRA adapter.

        Writes ``adapter_dir/iter_<name>.safetensors`` and atomically
        updates ``adapter_dir/current.safetensors`` to point at it.

        Returns the path to the versioned snapshot.
        """
        self._adapter_dir.mkdir(parents=True, exist_ok=True)
        snapshot = self._adapter_dir / f"iter_{name}.safetensors"
        current = self._adapter_dir / "current.safetensors"

        # Write a single safetensors file with just the LoRA params,
        # in the layout mlx_lm.server's --adapter-path expects.
        adapter_weights = dict(tree_flatten(self._model.trainable_parameters()))
        mx.save_safetensors(str(snapshot), adapter_weights)

        # Atomically swap "current.safetensors" to point at the new
        # snapshot. Hardlink + rename avoids a partial-write window the
        # sampler could otherwise observe.
        tmp = current.with_suffix(".safetensors.tmp")
        if tmp.exists():
            tmp.unlink()
        os.link(snapshot, tmp)
        os.replace(tmp, current)

        return str(snapshot)

    async def get_post_training_sampling_client(self) -> SamplingClient:
        """Save adapter, ask sampler to reload, return a fresh client."""
        from autodiscover.backends.mlx.sampling import MlxSamplingClient

        # Choose a name: versioned snapshot at the configured cadence,
        # otherwise a "live" overwrite so disk doesn't fill up.
        save_now = (
            self._save_every > 0
            and self._batch_count > 0
            and self._batch_count % self._save_every == 0
        )
        name = f"{self._batch_count:06d}" if save_now else "live"
        await self.save_checkpoint(name)

        # Ask the sampler to reload. If the trainer was constructed
        # without a sampler URL (tests), skip the reload silently.
        sampler_url = self._sampler_url
        if sampler_url:
            await _request_adapter_reload(
                sampler_url,
                Path(self._adapter_dir) / "current.safetensors",
            )

        return MlxSamplingClient(
            base_url=sampler_url or "",
            tokenizer=self._tokenizer,
            max_concurrency=self._sampler_max_concurrency,
        )

    # --- LoRA enable/disable for base-model logprob path ---

    def _set_lora_active(self, active: bool) -> None:
        """Enable or disable all LoRA adapters in the model in-place.

        ``mlx_lm.tuner`` exposes LoRA modules that have a ``scale`` or
        ``active`` attribute. The exact API varies; the simplest robust
        approach is to multiply the LoRA scale by 0 to disable, restore on
        re-enable. Inspect ``self._model.modules()`` to find the LoRA
        wrappers and switch the appropriate flag.
        """
        from mlx_lm.tuner.lora import LoRALinear  # adjust import per version
        for _, m in self._model.named_modules():
            if isinstance(m, LoRALinear):
                # Most versions expose an ``active`` boolean. If yours
                # doesn't, toggle ``m.scale = 20.0 if active else 0.0``.
                m.active = active

    def with_base_only(self):
        """Context manager that disables LoRA for the duration of the
        block. Used by incorporate_kl_penalty's base-model logprob calc.
        """
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                outer._set_lora_active(False)

            def __exit__(self_inner, exc_type, exc, tb):
                outer._set_lora_active(True)

        return _Ctx()


class _InProcessBaseSampling:
    """SamplingClient that computes base-model logprobs (LoRA off) by
    forwarding through the trainer's in-process MLX model.

    Only ``compute_logprobs`` is implemented -- KL penalty is the only
    caller. ``sample`` raises; if you need base-model sampling, route
    through mlx_lm.server with the adapter unloaded.
    """

    def __init__(self, trainer: MlxTrainingClient) -> None:
        self._trainer = trainer

    async def sample(
        self,
        prompt: TokenSequence,
        params: SamplingParams,
        num_samples: int = 1,
    ):
        raise NotImplementedError(
            "base-model sampling not provided by in-process MLX client",
        )

    async def compute_logprobs(
        self,
        sequences: list[TokenSequence],
    ) -> list[list[float]]:
        out: list[list[float]] = []
        with self._trainer.with_base_only():
            for seq in sequences:
                inp = mx.array(seq.tokens, dtype=mx.int32)[None, :]  # (1, T)
                logits = self._trainer.model()(inp)[0]               # (T, V)
                logp = nn.log_softmax(logits, axis=-1)
                # logprob of token_t under prefix ending at t-1.
                # The first token has no preceding context -> 0.0.
                T = len(seq.tokens)
                per_step: list[float] = [0.0]
                if T > 1:
                    idx = mx.arange(T - 1)
                    next_tokens = mx.array(seq.tokens[1:], dtype=mx.int32)
                    # logp at position i predicts token at position i+1.
                    taken = logp[:-1][idx, next_tokens]
                    mx.eval(taken)
                    per_step.extend([float(x) for x in taken.tolist()])
                out.append(per_step)
        return out


async def _request_adapter_reload(
    sampler_url: str,
    adapter_path: Path,
) -> None:
    """POST to the sampler's adapter-reload endpoint.

    See Step 15 for the endpoint definition / sampler-side wrapper.
    """
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(
            f"{sampler_url.rstrip('/')}/reload_adapter",
            json={"path": str(adapter_path)},
        )
        resp.raise_for_status()


async def make_mlx_backend(
    *,
    sampler_url: str,
    adapter_dir: Path,
    model_name: str,
    lora_rank: int,
    save_every: int,
    log_dir: str,
    learning_rate: float = 1e-5,
    sampler_max_concurrency: int = 1,
):
    """Factory used by the trainer-server backend factory (Step 09).

    Returns (base_sampling, sampling, training).

    Note: today this only constructs the training client and a sampler
    client wired to the running mlx_lm.server. base_sampling is provided
    by the SAME training client (LoRA-off path) via a small adapter -- see
    Step 13 for the implementation.
    """
    from autodiscover.backends.mlx.sampling import MlxSamplingClient

    training = MlxTrainingClient(
        model_name=model_name,
        lora_rank=lora_rank,
        adapter_dir=adapter_dir,
        save_every=save_every,
        log_dir=log_dir,
        learning_rate=learning_rate,
    )
    # Stash the sampler URL so post-training rotations can request an
    # adapter reload (see ``get_post_training_sampling_client``).
    training._sampler_url = sampler_url
    training._sampler_max_concurrency = sampler_max_concurrency
    sampling = MlxSamplingClient(
        base_url=sampler_url,
        tokenizer=training._tokenizer,
        max_concurrency=sampler_max_concurrency,
    )
    # base_sampling computes logprobs through the same in-process MLX
    # model with LoRA disabled. Used only by incorporate_kl_penalty.
    base_sampling = _InProcessBaseSampling(training)
    return base_sampling, sampling, training
