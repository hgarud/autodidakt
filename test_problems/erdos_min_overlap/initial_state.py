"""Initial state factory for Erdős min-overlap.

Self-contained: depends only on numpy. ``evaluator.get_reward`` reads
``state.construction`` and nothing else, so the state container is a thin
dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class InitialState:
    construction: list[float] = field(default_factory=list)


def make_initial_state(problem_type: str = "default") -> InitialState:
    """Build a fresh initial state.

    Randomizes ``n_points`` in [40, 100] and constructs h ≈ 0.5 +
    zero-mean perturbation. ``problem_type`` is accepted for API
    compatibility and currently unused.
    """
    del problem_type
    rng = np.random.default_rng()
    n_points = int(rng.integers(40, 100))
    construction = np.ones(n_points) * 0.5
    perturbation = rng.uniform(-0.4, 0.4, n_points)
    perturbation = perturbation - np.mean(perturbation)
    construction = construction + perturbation
    return InitialState(construction=list(construction))
