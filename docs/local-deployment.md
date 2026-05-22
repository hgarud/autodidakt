# Local (air-gapped) deployment runbook

> **Tested against:** `mlx-lm==0.31.3`, `mlx>=0.20`, Python 3.12, macOS
> 14.5+. If you upgrade `mlx-lm`, re-validate Section 5 (sampler launch)
> and Section 6 (cold-start memory numbers) before shipping.

This document walks an on-site operator from a cold machine to a
running `/discover` session, with **no network egress** other than the
Claude Code session's own connection to the Anthropic API (see Section
7). The reader is assumed to be a senior engineer who has never opened
this repo before.

---

## 1. Hardware floor

- **Apple Silicon, M3 Max / M4 Max / M4 Ultra.** Intel and pre-M3 chips
  are not supported.
- **128 GB unified memory minimum** for the default `gpt-oss-120b` build.
  - Resident set during a run is ~120 GB (sampler ~60 GB + trainer ~60
    GB + OS).
  - 64 GB machines **cannot** train `gpt-oss-120b`. If 64 GB is all you
    have, downsize the model in *every* `--mlx-model` argument:
    ```
    --mlx-model mlx-community/gpt-oss-20b-MXFP4 --lora-rank 8
    ```
- **~250 GB free disk** (model weights ~60 GB + adapter snapshots ~5
  GB/run + logs).

---

## 2. Pre-stage model weights (no internet at target)

On an **internet-connected** staging machine with the same
Hugging Face cache layout you intend to use on the target:

```bash
huggingface-cli download mlx-community/gpt-oss-120b-MXFP4 \
    --local-dir ~/models/gpt-oss-120b-MXFP4
# (or mlx-community/gpt-oss-20b-MXFP4 on a 64 GB machine)
```

Copy `~/models/` to the target host (encrypted USB, sneakernet, etc.)
preserving directory structure.

On the **target**:

```bash
export HF_HOME=~/models
# Persist this in your shell profile so every terminal sees it:
echo 'export HF_HOME=~/models' >> ~/.zshrc
```

Verify `mlx_lm` resolves the model locally without network:

```bash
uv run python - <<'PY'
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
from mlx_lm.utils import load
m, tok = load("mlx-community/gpt-oss-120b-MXFP4")
print("ok:", type(m).__name__)
PY
```

If that errors with a network call, fix the cache layout before
proceeding. The model directory must contain `config.json`,
`tokenizer.json`, and `model*.safetensors`.

---

## 3. Install

In the project root:

```bash
uv sync --extra mlx
```

Do **not** install `--extra tinker` — that extra pulls a hosted-service
SDK and is unnecessary (and undesirable) for local deployment. If you
previously installed it, recreate the venv:

```bash
rm -rf .venv && uv sync --extra mlx
```

---

## 4. Run layout

Create a working directory outside the repo:

```bash
mkdir -p ~/discover-run/{adapters,logs}
```

Final layout:

```
~/discover-run/
├── adapters/                       # LoRA snapshots
│   ├── current.safetensors         # symlink — always points at latest
│   └── iter_*.safetensors          # per-batch snapshots
├── logs/
│   ├── trainer.log
│   ├── archive.csv                 # server-side reward archive
│   └── _trace.json                 # internal training trace
└── problem-A/                      # one cwd per problem
    ├── problem.md                  # natural-language statement
    ├── baseline.py                 # starting code (optional)
    ├── evaluator.py                # local scorer; run inside $WORKDIR
    └── results.csv                 # local mirror of archive.csv
```

The `current.safetensors` symlink is maintained by the trainer after
each batch. The sampler watches that path.

---

## 5. Boot order (three terminals)

Boot in this exact order. Wait for each step to print its readiness
banner before starting the next.

### Terminal 1 — Sampler

Use Path A if your `mlx-lm` version exposes `/v1/load_lora_adapter`
natively (check with `mlx_lm.server --help | grep -i adapter`). Use
Path B otherwise. (See `docs/impl-steps/15-mlx-adapter-reload.md` for
the decision tree.)

**Path A — native `mlx_lm.server`:**

```bash
mlx_lm.server \
    --model mlx-community/gpt-oss-120b-MXFP4 \
    --adapter-path ~/discover-run/adapters/current.safetensors \
    --host 127.0.0.1 --port 8081
```

**Path B — wrapper with reload endpoint:**

```bash
uv run python -m autodiscover.backends.mlx.lm_server \
    --model mlx-community/gpt-oss-120b-MXFP4 \
    --adapter-path ~/discover-run/adapters/current.safetensors \
    --host 127.0.0.1 --port 8081
```

Sampler is ready when it logs `Uvicorn running on http://127.0.0.1:8081`
and `curl -s http://127.0.0.1:8081/v1/models` returns JSON.

### Terminal 2 — Trainer

```bash
uv run python -m autodiscover.cli.trainer_server \
    --backend mlx_local \
    --mlx-sampler-url http://127.0.0.1:8081 \
    --mlx-adapter-dir ~/discover-run/adapters \
    --mlx-model mlx-community/gpt-oss-120b-MXFP4 \
    --log-dir ~/discover-run/logs \
    --group-size 64 --groups-per-batch 8 --num-epochs 50
```

On startup the trainer prints, exactly once:

```
AUTODISCOVER_SERVER_URL=http://127.0.0.1:<port>
```

Copy that line — Terminal 3 needs it. The trainer is ready when
`curl -s http://127.0.0.1:<port>/status` returns `{"ok": true, ...}`.

### Terminal 3 — Orchestrator (Claude Code)

```bash
cd ~/discover-run/problem-A
export AUTODISCOVER_SERVER_URL=http://127.0.0.1:<port>  # from Terminal 2
claude
```

Then inside the Claude Code session:

```
/discover <high-level task goal in one sentence>
```

The session prompts for clarifications, gathers context from cwd, then
fans out subagents. Watch `./results.csv` in another shell to see
rewards land.

---

## 6. Cold-start gotchas

- **Sampler boot is slow.** The first `mlx_lm.server` boot loads
  ~60 GB of weights into unified memory; expect **30–90 seconds**
  before the "Uvicorn running" line. Do not Ctrl-C — it isn't hung.
- **Trainer boot is also slow.** It separately loads ~60 GB. Until it
  finishes, `/status` refuses connections. Do **not** run `/discover`
  until both Terminal 1 and Terminal 2 are responsive.
- **Memory pressure.** Total resident set ~120 GB on a 128 GB machine.
  Activity Monitor's "Memory Pressure" gauge will sit in **yellow** —
  expected. If it goes **red**, swap is active and you will see 10–50×
  slowdown. Mitigations:
    - Lower `--group-size` to 32 (default 64).
    - Lower `--groups-per-batch` to 4 (default 8).
    - Downsize to `gpt-oss-20b` (Section 1).
- **Don't run other large processes.** Browser tabs are fine. A second
  LLM, a Docker VM, or a JetBrains IDE indexing a monorepo are not.

---

## 7. Air-gap proof

1. Confirm the sandbox allowlist (from Step 16) is in place:

   ```bash
   jq '.sandbox' .claude/settings.json
   ```

   `network.allowOnly` (or equivalent in your Claude Code version) must
   list **only** `127.0.0.1` / `localhost`. No `api.anthropic.com`, no
   `huggingface.co`, no `*.amazonaws.com`.

2. While `/discover` is running an iteration, in another terminal:

   ```bash
   lsof -i -P -n | grep -vE '127\.0\.0\.1|::1|localhost' | grep -v COMMAND
   ```

   Output must be empty *modulo* the Claude Code parent process's
   connection to your Anthropic API endpoint (see point 4 below).

3. Confirm the trainer is not calling out:

   ```bash
   lsof -p $(pgrep -f trainer_server) -i -P -n | grep -v 127.0.0.1
   ```

   Empty.

4. **Important caveat — the Claude API itself.** Claude Code's
   connection from this machine to `api.anthropic.com` is **not**
   proxied through the sandbox configured here. The sandbox controls
   subagent/tool egress, not Claude Code's own LLM call. If the
   customer's air-gap requirement includes the orchestrator model:

   - Run the Claude API via a **privately-deployed inference endpoint**
     (AWS Bedrock with private VPC + PrivateLink, or GCP Vertex AI
     equivalent).
   - Configure Claude Code to use it via `ANTHROPIC_BEDROCK_BASE_URL`
     / `CLAUDE_CODE_USE_BEDROCK=1` (or the Vertex equivalent) before
     launching `claude` in Terminal 3.
   - The trainer / sampler / subagents are unaffected: they never call
     the Anthropic API, only the local sampler.

   If the customer accepts the Anthropic-API egress as out-of-scope
   (the typical case), say so explicitly in the deployment sign-off.

---

## 8. Operating during a run

- `~/discover-run/problem-A/results.csv` is a thin local mirror of
  rewards. It is rewritten on every reward submission.
- `~/discover-run/logs/archive.csv` is the **canonical** archive
  (server-side, fsynced per batch).
- Inspect top plans without leaving the host:

  ```bash
  curl -s http://127.0.0.1:<port>/best?k=5 | jq
  ```

- Adapter snapshots accumulate at `~/discover-run/adapters/iter_*.safetensors`.
  Each is ~50 MB at `lora_rank=32`. Prune old ones manually if disk
  pressure mounts — they are not required for resume (see Section 9).
- The trainer terminates when `/status` returns `done: true` (after
  `--num-epochs` batches). The sampler keeps running so you can do
  inference against the final adapter.

---

## 9. Restart / resume

- **Trainer killed mid-run.** `archive.csv` is fsynced after every
  batch. Re-launch with the **same** `--log-dir` and
  `--mlx-adapter-dir` and the trainer resumes from the last completed
  batch. In-flight rewards from the killed batch are lost.
- **Sampler killed mid-run.** Re-launch with the same `--adapter-path`.
  The trainer's next `POST /reload_adapter` will fail once and retry; no
  state loss.
- **Both killed.** Bring up Terminal 1 then Terminal 2 in the original
  order. The orchestrator session in Terminal 3 can stay open — it will
  reconnect on the next `/rollout/begin`.

---

## 10. Tearing down

```bash
# In each of Terminals 1, 2, 3: Ctrl-C, wait for graceful shutdown
# (trainer prints "saved final adapter" then exits).

# Disk cleanup:
rm -rf /tmp/discover                              # subagent scratch dirs
rm -rf ~/discover-run/adapters/iter_*.safetensors # keep current.safetensors

# To wipe everything except the model cache:
rm -rf ~/discover-run
```

The model cache under `$HF_HOME` is **not** touched — re-use it for the
next run.

---

## 11. Quick sanity checks (run after every fresh install)

```bash
# 1. mlx-lm version matches what this runbook was tested against
uv run python -c "import mlx_lm; print(mlx_lm.__version__)"  # expect 0.31.3

# 2. Sampler responds
curl -s http://127.0.0.1:8081/v1/models | jq '.data[0].id'

# 3. Trainer responds
curl -s "$AUTODISCOVER_SERVER_URL/status" | jq

# 4. Sandbox allowlist
jq '.sandbox' .claude/settings.json

# 5. No outbound sockets to the public internet
lsof -i -P -n | grep -vE '127\.0\.0\.1|::1' | grep ESTABLISHED || echo "clean"
```

If any of those fail, do not proceed to `/discover` — debug first.
