"""Subagent-facing evaluator entrypoint for the Erdős test problem.

Usage (inside a subagent's worktree, cwd = test_problems/erdos_min_overlap):
    uv run python eval.py --code-path path/to/candidate.py

Prints exactly one JSON line on stdout. Logging / Ray banners go to stderr.
"""
from __future__ import annotations

import argparse
import json
import sys

from evaluator import get_reward
from initial_state import make_initial_state


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--code-path", required=True)
    p.add_argument("--problem-type", default="default")
    args = p.parse_args(argv)

    code = open(args.code_path, "r").read()
    state = make_initial_state(args.problem_type)
    out = get_reward(code, state)
    # Emit exactly one JSON line on stdout for the orchestrator to parse.
    print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
