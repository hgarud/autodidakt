"""Data processing functions for RL training.

Converts trajectories into a backend-neutral list of TrainingDatum.
"""
from __future__ import annotations

import logging
from typing import List

from autodiscover.backends.types import TrainingDatum
from autodiscover.types import Trajectory, TrajectoryGroup
from autodiscover.utils import all_same, safezip

logger = logging.getLogger(__name__)


def _rightshift_leftshift(tokens: list[int]) -> tuple[list[int], list[int]]:
    """Build (input_tokens, target_tokens) from a flat token sequence."""
    if len(tokens) < 2:
        raise ValueError("need at least 2 tokens for input/target split")
    return tokens[:-1], tokens[1:]


def _is_prefix(seq1: list[int], seq2: list[int]) -> bool:
    return len(seq1) <= len(seq2) and seq2[: len(seq1)] == seq1


def trajectory_to_data(
    traj: Trajectory, traj_advantage: float,
) -> list[TrainingDatum]:
    """Return one or more TrainingDatum corresponding to the trajectory.

    Merging rule (unchanged from before): if each successive observation
    is a prefix-extension of the previous accumulator, merge into a single
    Datum. Otherwise start a new Datum.
    """

    class _Acc:
        full_sequence: list[int] = []
        sampled_logprobs: list[float] = []
        advantages: list[float] = []
        mask: list[float] = []

        @classmethod
        def clear(cls) -> None:
            cls.full_sequence = []
            cls.sampled_logprobs = []
            cls.advantages = []
            cls.mask = []

    def _flush() -> TrainingDatum:
        all_tokens = list(_Acc.full_sequence)
        input_tokens, target_tokens = _rightshift_leftshift(all_tokens)
        # First entries are the prompt; logprobs/adv/mask are aligned to
        # the full sequence and need a left-shift to match target_tokens.
        old_logprobs = _Acc.sampled_logprobs[1:]
        advantages = _Acc.advantages[1:]
        mask = _Acc.mask[1:]
        assert (
            len(input_tokens) == len(target_tokens) == len(old_logprobs)
            == len(advantages) == len(mask)
        )
        return TrainingDatum(
            input_tokens=input_tokens,
            target_tokens=target_tokens,
            old_logprobs=old_logprobs,
            advantages=advantages,
            mask=mask,
        )

    data: list[TrainingDatum] = []
    for transition in traj.transitions:
        ob_tokens = transition.ob.tokens  # TokenSequence -> list[int]
        ac = transition.ac

        if not _Acc.full_sequence:
            delta = ob_tokens
        elif _is_prefix(_Acc.full_sequence, ob_tokens):
            delta = ob_tokens[len(_Acc.full_sequence):]
        else:
            data.append(_flush())
            _Acc.clear()
            delta = ob_tokens

        delta_len = len(delta)
        _Acc.full_sequence.extend(delta)
        _Acc.full_sequence.extend(ac.tokens)
        _Acc.sampled_logprobs.extend([0.0] * delta_len + ac.logprobs)
        _Acc.advantages.extend(
            [0.0] * delta_len + [traj_advantage] * len(ac.tokens),
        )
        _Acc.mask.extend([0.0] * delta_len + ac.mask)

    if _Acc.full_sequence:
        data.append(_flush())

    return data


def assemble_training_data(
    trajectory_groups_P: List[TrajectoryGroup],
    advantages_P,  # List[torch.Tensor] or list[list[float]] — iterable per-group
) -> tuple[List[TrainingDatum], List[dict[str, int]]]:
    """Convert trajectories to training data."""
    data_D: list[TrainingDatum] = []
    metadata_D: list[dict[str, int]] = []

    for i_group, (traj_group, advantages_G) in enumerate(
        safezip(trajectory_groups_P, advantages_P),
    ):
        for i_traj, (traj, traj_advantage) in enumerate(
            safezip(traj_group.trajectories_G, advantages_G),
        ):
            new_data = trajectory_to_data(traj, float(traj_advantage))
            data_D.extend(new_data)
            metadata_D.extend(
                [dict(group_idx=i_group, traj_idx=i_traj) for _ in new_data],
            )
    return data_D, metadata_D


def remove_constant_reward_groups(
    trajectory_groups_P: List[TrajectoryGroup],
) -> List[TrajectoryGroup]:
    new_groups: list[TrajectoryGroup] = []
    for group in trajectory_groups_P:
        if not all_same(group.get_total_rewards()):
            new_groups.append(group)
    if not new_groups:
        logger.warning("All rewards are uniform. There will be no gradient.")
        return trajectory_groups_P[0:1]
    return new_groups
