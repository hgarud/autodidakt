"""
RL training step primitives.

Extracted from the original discover RL training script; only the four
public entrypoints (`compute_advantages`, `incorporate_kl_penalty`,
`train_step`, `save_checkpoint_and_get_sampling_client`) plus their
private helpers are kept here.
"""

import asyncio
import logging
import math
from typing import Any, Dict, List, cast

import numpy as np
import tinker
import torch
from tinker.types import LossFnType

from autodiscover.checkpointing import save_checkpoint_async
from autodiscover.trace import scope
from autodiscover.types import TrajectoryGroup
from autodiscover.utils import safezip, split_list, timed

logger = logging.getLogger(__name__)


@scope
async def incorporate_kl_penalty(
    data_D: List[tinker.Datum],
    base_sampling_client: tinker.SamplingClient,
    kl_penalty_coef: float,
) -> Dict[str, float]:
    """
    Compute KL against base model. Adjust advantages in-place by logp_base - logp_current - avg_kl,
    where avg_kl is the average of logp_base - logp_current (which is -KL[current, base])
    """
    # Compute logprobs at all data items
    full_sequence_inputs_D = [
        datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
        for datum in data_D
    ]
    base_logprobs_D = await asyncio.gather(
        *[
            base_sampling_client.compute_logprobs_async(sequence_input)
            for sequence_input in full_sequence_inputs_D
        ]
    )
    # compute the logprob differences, zeroed out when the mask == 0
    sampled_logprobs_D = [datum.loss_fn_inputs["logprobs"].to_torch() for datum in data_D]
    float_masks = [datum.loss_fn_inputs["mask"].to_torch().float() for datum in data_D]
    logprob_diffs = [
        (sampled_logprobs - torch.tensor(base_logprobs[1:])) * mask
        for base_logprobs, sampled_logprobs, mask in safezip(
            base_logprobs_D, sampled_logprobs_D, float_masks
        )
    ]
    avg_logp_diff = sum([diff.sum() for diff in logprob_diffs]) / sum(
        [mask.sum() for mask in float_masks]
    )
    for i, datum in enumerate(data_D):
        kl_advantages = kl_penalty_coef * float_masks[i] * (avg_logp_diff - logprob_diffs[i])
        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )
    return {"kl_policy_base": float(avg_logp_diff)}


def compute_advantages(trajectory_groups_P: List[TrajectoryGroup], adv_estimator: str, adv_estimator_beta: float, adv_estimator_mu: float = 5/1.503163635, adv_estimator_sigma: float = 0.000001) -> List[torch.Tensor]:
    """Compute advantages for each trajectory, centered within groups."""
    advantages_P: list[torch.Tensor] = []

    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        # Center advantages within the group
        if adv_estimator == "mean_baseline":
            advantages_G = rewards_G - rewards_G.mean()
        elif adv_estimator == "entropic":
            beta = adv_estimator_beta
            s_safe = rewards_G - rewards_G.max(dim=-1, keepdim=True)[0]
            e = torch.exp(beta * s_safe)
            k = e.shape[0]
            if k == 1:
                Z = e
            else:
                Z = (e.sum() - e) / (k - 1)
            w = e / (Z + 1e-12)
            advantages_G = w - 1.0
        elif adv_estimator == "entropic_adaptive_beta":
            delta = np.log(2)
            beta_max = 1e6
            iters = 60
            eps = 1e-12

            r = rewards_G.float()
            k = r.shape[0]

            if k < 2:
                beta = r.new_tensor(0.0)
            else:
                logK = math.log(k)

                def kl_hat(beta_scalar: float) -> float:
                    # q_beta over samples: q ∝ exp(beta * r), KL(q||uniform)
                    b = r.new_tensor(beta_scalar)
                    logits = b * (r - r.max(dim=0, keepdim=True).values)      # stable
                    logq = logits - torch.logsumexp(logits, dim=0, keepdim=True)
                    q = torch.exp(logq)
                    kl = (q * (logq + logK)).sum(dim=0)
                    return float(kl.mean().item())

                lo, hi = 0.0, 1.0
                if kl_hat(hi) < delta:
                    while hi < beta_max and kl_hat(hi) < delta:
                        hi *= 2.0
                    if kl_hat(hi) < delta:
                        beta = r.new_tensor(hi)  # best effort
                    else:
                        beta = None
                else:
                    beta = None

                if beta is None:
                    for _ in range(iters):
                        mid = 0.5 * (lo + hi)
                        if kl_hat(mid) < delta:
                            lo = mid
                        else:
                            hi = mid
                    beta = r.new_tensor(hi)

            # LOO entropic advantages using solved beta
            e = torch.exp(beta * (r - r.max(dim=0, keepdim=True).values))

            if k == 1:
                Z = e
            else:
                Z = (e.sum(dim=0, keepdim=True) - e) / (k - 1)

            w = e / (Z + eps)
            advantages_G = w - 1.0
        else:
            raise ValueError(f"Invalid advantage estimator: {adv_estimator}")
        advantages_P.append(advantages_G)

    return advantages_P


@scope
async def enqueue_optim_step(
    training_client: tinker.TrainingClient,
    learning_rate: float,
) -> tinker.APIFuture[tinker.OptimStepResponse]:
    """Enqueue an optimizer step and return the future"""
    adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    optim_step_future = await training_client.optim_step_async(adam_params)
    return optim_step_future


@scope
async def consume_optim_step(
    optim_step_future: tinker.APIFuture[tinker.OptimStepResponse],
) -> tinker.OptimStepResponse:
    """Apply the accumulated gradients to update the model weights and return the result"""
    return await optim_step_future.result_async()


@scope
def remove_mask(datum: tinker.Datum) -> tinker.Datum:
    return tinker.Datum(
        model_input=datum.model_input,
        loss_fn_inputs={k: v for k, v in datum.loss_fn_inputs.items() if k != "mask"},
    )


@scope
async def enqueue_forward_backward(
    training_client: tinker.TrainingClient,
    batch_d: List[tinker.Datum],
    loss_fn: LossFnType,
) -> tinker.APIFuture[tinker.ForwardBackwardOutput]:
    """Enqueue a forward-backward pass for a minibatch of data and return the future"""
    fwd_bwd_future = await training_client.forward_backward_async(
        list(map(remove_mask, batch_d)), loss_fn=loss_fn
    )
    return fwd_bwd_future


@scope
async def consume_forward_backward(
    fwd_bwd_future: tinker.APIFuture[tinker.ForwardBackwardOutput],
) -> List[torch.Tensor]:
    """Consume the result of a forward-backward pass and return the training logprobs"""
    fwd_bwd_result = await fwd_bwd_future.result_async()

    # Extract training logprobs from loss_fn_outputs
    training_logprobs_D: list[torch.Tensor] = []
    for output in fwd_bwd_result.loss_fn_outputs:
        training_logprobs = output["logprobs"].to_torch()
        training_logprobs_D.append(training_logprobs)

    # We dont display fwd_bwd_result.metrics to avoid spam
    return training_logprobs_D


@scope
async def train_step(
    data_D: List[tinker.Datum],
    training_client: tinker.TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: LossFnType,
) -> List[torch.Tensor]:
    """Train the model on collected trajectories."""
    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    training_logprobs_D: list[torch.Tensor] = []

    if len(batches_md) == 0:
        return training_logprobs_D

    enqueued_futures: (
        tuple[
            tinker.APIFuture[tinker.ForwardBackwardOutput],
            tinker.APIFuture[tinker.OptimStepResponse],
        ]
        | None
    ) = (
        await enqueue_forward_backward(training_client, batches_md[0], loss_fn),
        await enqueue_optim_step(training_client, learning_rate),
    )

    for i in range(len(batches_md)):
        assert enqueued_futures is not None

        fwd_bwd_future, optim_step_future = enqueued_futures
        enqueued_futures = None

        # Enqueue the next forward-backward pass and optimizer step before consuming the current result
        if i != len(batches_md) - 1:
            assert enqueued_futures is None
            enqueued_futures = (
                await enqueue_forward_backward(training_client, batches_md[i + 1], loss_fn),
                await enqueue_optim_step(training_client, learning_rate),
            )

        training_logprobs = await consume_forward_backward(fwd_bwd_future)
        training_logprobs_D.extend(training_logprobs)

        await consume_optim_step(optim_step_future)

    assert enqueued_futures is None

    return training_logprobs_D


@scope
async def save_checkpoint_and_get_sampling_client(
    training_client: tinker.TrainingClient,
    i_batch: int,
    log_path: str,
    save_every: int,
    start_batch: int = 0,
) -> tuple[tinker.SamplingClient, dict[str, Any]]:
    metrics = {}
    with timed("save_checkpoint", metrics):
        if save_every > 0 and i_batch > start_batch and i_batch % save_every == 0:
            path_dict = await save_checkpoint_async(
                training_client=training_client,
                name=f"{i_batch:06d}",
                log_path=log_path,
                loop_state={"batch": i_batch},
                kind="both",
            )
            return training_client.create_sampling_client(path_dict["sampler_path"]), metrics
        else:
            return await training_client.save_weights_and_get_sampling_client_async(), metrics
