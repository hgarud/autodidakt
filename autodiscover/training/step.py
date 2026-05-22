"""RL training step primitives (backend-agnostic).

Public entrypoints:
- compute_advantages
- incorporate_kl_penalty
- train_step
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List

import numpy as np
import torch

from autodiscover.backends.protocol import SamplingClient, TrainingClient
from autodiscover.backends.types import TokenSequence, TrainingDatum
from autodiscover.trace import scope
from autodiscover.types import TrajectoryGroup
from autodiscover.utils import safezip, split_list

logger = logging.getLogger(__name__)


@scope
async def incorporate_kl_penalty(
    data_D: List[TrainingDatum],
    base_sampling_client: SamplingClient,
    kl_penalty_coef: float,
) -> Dict[str, float]:
    """KL against base. Adjusts ``datum.advantages`` in place by
    ``kl_coef * mask * (avg_logp_diff - logprob_diffs[i])``.
    """
    # Full sequence = input_tokens + [last target token].
    full_sequences = [
        TokenSequence(tokens=d.input_tokens + [d.target_tokens[-1]])
        for d in data_D
    ]
    base_logprobs_D = await base_sampling_client.compute_logprobs(full_sequences)

    sampled_logprobs_D = [torch.tensor(d.old_logprobs) for d in data_D]
    float_masks = [torch.tensor(d.mask, dtype=torch.float32) for d in data_D]

    logprob_diffs = [
        (sampled_lp - torch.tensor(base_lp[1:])) * mask
        for base_lp, sampled_lp, mask in safezip(
            base_logprobs_D, sampled_logprobs_D, float_masks,
        )
    ]
    total_mass = sum(m.sum() for m in float_masks)
    avg_logp_diff = sum(diff.sum() for diff in logprob_diffs) / total_mass

    for i, d in enumerate(data_D):
        kl_adv = (
            kl_penalty_coef
            * float_masks[i]
            * (avg_logp_diff - logprob_diffs[i])
        )
        new_adv = torch.tensor(d.advantages) + kl_adv
        d.advantages = new_adv.tolist()

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
async def train_step(
    data_D: List[TrainingDatum],
    training_client: TrainingClient,
    learning_rate: float,
    num_substeps: int,
    loss_fn: str,
) -> None:
    """Run ``num_substeps`` forward_backward + optim_step pairs over the
    batch. Backend may pipeline internally.
    """
    if not data_D:
        return
    batches_md = split_list(data_D, min(num_substeps, len(data_D)))
    for batch in batches_md:
        await training_client.forward_backward(batch, loss_fn)
        await training_client.optim_step(learning_rate=learning_rate)
