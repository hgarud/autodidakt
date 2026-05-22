"""
Basic interfaces and types for reinforcement learning.
"""

from dataclasses import dataclass, field
from typing import TypeAlias

from autodiscover.backends.types import TokenSequence
from autodiscover.completers import StopCondition, TokensWithLogprobs
from autodiscover.utils import safezip

__all__ = [
    "Action", "Observation", "Logprobs", "Metrics",
    "StopCondition", "TokensWithLogprobs",
    "Transition", "Trajectory", "TrajectoryGroup",
]

Action: TypeAlias = list[int]
Observation: TypeAlias = TokenSequence
Logprobs: TypeAlias = list[float]
Metrics: TypeAlias = dict[str, float | int]


@dataclass
class Transition:
    ob: Observation
    ac: TokensWithLogprobs
    reward: float
    episode_done: bool
    metrics: Metrics = field(default_factory=dict)


@dataclass(frozen=True)
class Trajectory:
    """
    A sequence of observations and actions, resulting from running a single agent in a single
    environment.
    """

    transitions: list[Transition]
    final_ob: Observation


@dataclass
class TrajectoryGroup:
    """
    A group of trajectories, resulting from instantiating a group of environments using an
    EnvGroupBuilder, doing a rollout for each environment, and computing the rewards.
    """

    trajectories_G: list[Trajectory]
    final_rewards_G: list[float]  # computed by the EnvGroupBuilder, looking at whole group
    metrics_G: list[Metrics]

    def get_total_rewards(self) -> list[float]:
        """
        Get the total reward (i.e., the return) of each trajectory (episode) in the group.
        The total reward is the sum of the per-timestep rewards plus the final group reward
        computed by the EnvGroupBuilder.
        """
        return [
            sum(transition.reward for transition in trajectory.transitions) + final_reward
            for trajectory, final_reward in safezip(self.trajectories_G, self.final_rewards_G)
        ]
