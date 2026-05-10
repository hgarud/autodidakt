"""Local reward evaluator for the Erdős min-overlap problem.

This is the lightweight per-problem evaluator the subagent calls via
``eval.py``. It executes the candidate's ``run()`` in-process and returns
a reward dict matching the contract documented in this directory's
README.md.

Contract: ``get_reward(code: str, state: State) -> dict`` with at minimum
``reward``, ``correctness``, ``raw_score``, ``msg`` keys. (Plus
``result_construction`` and ``stdout`` when available.)
"""
from __future__ import annotations

import contextlib
import io
import traceback
from typing import Any

import numpy as np


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        h_values = h_values * (target_sum / current_sum)
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result: tuple[np.ndarray, float, int]) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


def _failure(msg: str, *, stdout: str = "") -> dict[str, Any]:
    return {
        "reward": 0.0,
        "correctness": 0.0,
        "raw_score": float("inf"),
        "msg": msg,
        "stdout": stdout,
    }


def get_reward(code: str, state: Any) -> dict[str, Any]:
    """Execute ``code`` (must define ``run()``) and score the result.

    Returns a dict with keys ``reward, correctness, raw_score, msg`` plus
    ``result_construction`` and ``stdout`` when available. Any exception
    in user code is captured and surfaced as a zero-reward failure entry
    instead of propagating.
    """
    # Inject helpers and (optionally) the initial construction.
    globs: dict[str, Any] = {
        "__name__": "__candidate__",
        "np": np,
        "evaluate_erdos_solution": evaluate_erdos_solution,
    }
    construction = getattr(state, "construction", None)
    if construction is not None:
        globs["initial_h_values"] = np.array(list(construction))

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(compile(code, "<candidate>", "exec"), globs)
            run_fn = globs.get("run")
            if not callable(run_fn):
                return _failure("candidate defines no top-level run()", stdout=buf.getvalue())
            output = run_fn()
    except Exception:
        return _failure(
            "candidate raised: " + traceback.format_exc(limit=4),
            stdout=buf.getvalue(),
        )

    if not verify_erdos_solution(output):
        return _failure("invalid solution (failed verifier)", stdout=buf.getvalue())

    h_values, c5_bound, n_points = output
    c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
    return {
        "reward": float(1.0 / (1e-8 + c5_bound)),
        "correctness": 1.0,
        "raw_score": float(c5_bound),
        "msg": f"C5 bound: {c5_bound}",
        "result_construction": list(h_values) if hasattr(h_values, "__iter__") else h_values,
        "stdout": buf.getvalue(),
    }
