# Erdős minimum-overlap problem

## Goal

Find a discretized step function `h: [0, 2] → [0, 1]` (length `n_points`,
with `dx = 2.0 / n_points`) that **minimizes** the autocorrelation overlap

```
C5 = max(np.correlate(h, 1 - h, mode="full")) * dx
```

subject to `h[i] ∈ [0, 1]` and `sum(h) * dx == 1`.

- Best known C5: `≤ 0.380876` (Haugland 2010, FFT-accelerated descent).
- Target this run: `C5 ≤ 0.38080`.

## `run()` contract

A solution is a single top-level function:

```python
def run(seed: int = 42, budget_s: int = 1000, **kwargs) -> tuple[list[float], float, int]:
    ...
    return h_values, c5_bound, n_points
```

- `h_values`: list of `n_points` floats, each in `[0, 1]`, with
  `sum(h_values) * (2.0 / n_points) == 1`.
- `c5_bound`: float equal to
  `max(np.correlate(h_values, 1 - h_values, mode="full")) * (2.0 / n_points)`.
- `n_points`: int length of `h_values`.

Allowed libraries: `numpy`, `scipy`, `cvxpy`, `math`. No filesystem or
network IO inside `run()`. Resources: 2 CPUs, ≤ 1100 s wall, ≤ 1 GB
memory. All helpers must be top-level (no closures or lambdas).

## Reference techniques

- Haugland (2010): step-function descent with FFT-accelerated correlations.
- Swinnerton-Dyer's correlation bound.
- Projected gradient descent on the simplex `{h : 0 ≤ h ≤ 1, ∫h dx = 1}`.
- Simulated annealing on step-function representations.

## Evaluating a candidate

A subagent in a worktree can run:

```
uv run python eval.py --code-path path/to/candidate.py
```

`eval.py` prints exactly one JSON line on stdout, e.g.:

```json
{"reward": 0.71, "correctness": 1.0, "raw_score": 0.39, "msg": "ok",
 "result_construction": [0.0, 0.5, 1.0, ...], "stdout": "..."}
```

The orchestrator forwards this dict directly to
`POST $AUTODISCOVER_SERVER_URL/rollout/reward` as the `results` field.
