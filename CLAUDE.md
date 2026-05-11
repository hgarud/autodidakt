# autodidakt-claude

A live-Claude-Code-orchestrated test-time RL system. A long-running Claude
Code session is the orchestrator + implementer; a separately-launched FastAPI
trainer server (`autodiscover.cli.trainer_server`) owns the policy.

The orchestration contract lives in `.claude/commands/discover.md` — that
file is the source of truth. This document gives the high-level picture; for
any details about the workflow, defer to `discover.md`.

## Roles

- **Live Claude Code session (orchestrator)**: invoked as
  `/discover <high_level_task_goal>`. Validates the trainer server, asks
  clarifying questions, gathers context from cwd + the web, posts the
  context to `/rollout/begin`, **APFS-clones cwd into one scratch dir per
  plan** under `/tmp/discover/<run_id>/<plan_id>`, fans out one subagent
  per plan in a single message, collects each subagent's results JSON,
  posts back to `/rollout/reward`, `rm -rf`s the run's scratch tree, and
  writes a `./results.csv` mirror.
- **Subagents (implementers)**: `subagent_type: general-purpose`. Each
  receives one plan plus an assigned `WORKDIR=/tmp/discover/<run>/<plan_id>`
  (an APFS clone of cwd), `cd`s there, implements the plan, runs the
  problem's local evaluator inside `$WORKDIR`, returns a single JSON
  results dict, and is destroyed. Subagents do not use git worktrees.
- **Trainer server (out-of-band)**: a problem-agnostic FastAPI process
  hosting a Tinker LoRA. Endpoints: `/status`, `/rollout/begin`,
  `/rollout/reward`, `/best`. The orchestrator talks to whatever URL is in
  `$AUTODISCOVER_SERVER_URL`. The server is *not* launched by `/discover`.

## Wire schema

```
POST /rollout/begin   { iter_idx: int, context: str }
                   -> { plans: [ { plan_id, plan_text, iter_idx } ] }

POST /rollout/reward  { plan_id, results: dict }
                   -> { accepted, group_complete, batch_complete }

GET  /status       -> { ok, iter, batches_trained, outstanding,
                        last_reward_mean, done }

GET  /best?k=5     -> { rows: [ { plan_id, iter_idx, reward, results } ] }
```

The server reads `results["reward"]` (float) for the RL step and persists
the full results dict for `/best`.

## Per-run flow (operator)

1. **Boot the trainer server**:
   ```
   uv run python -m autodiscover.cli.trainer_server \
       --group-size 64 --groups-per-batch 8 --num-epochs 50
   ```
   The CLI prints `AUTODISCOVER_SERVER_URL=http://host:port`.

2. **Export the URL** in the orchestrator's shell:
   ```
   export AUTODISCOVER_SERVER_URL=http://host:port
   ```

3. **`cd` into a problem directory** containing whatever materials the
   orchestrator should treat as context (problem statement, baseline code,
   prior results, an evaluator script). The orchestrator gathers context
   freely; there is no required filename layout.

4. **Run `/discover <goal>`** in the live Claude Code session and follow
   the workflow defined in `.claude/commands/discover.md`.

## Conventions

- `./results.csv` (cwd-relative) is a thin local mirror of rewards. The
  canonical store is server-side and is what `/best` returns.
- Subagents run in APFS-cloned scratch dirs under `/tmp/discover/` (no git
  worktrees) and never contact `$AUTODISCOVER_SERVER_URL` directly — only
  the live session does. Claude Code's sandbox (`.claude/settings.json`)
  enforces that subagent writes stay inside `/tmp/discover/`.
- Each `/rollout/begin` returns G plans. The orchestrator emits all G
  Agent calls in a single assistant message (so they run in parallel).
- One full run completes when `/status` returns `done: true`.

## Costs

At paper config (G=64, P=8, num_epochs=50) one run is ~512 subagent
invocations per training batch. Tinker compute is ~$500 per gpt-oss-120b
run.
