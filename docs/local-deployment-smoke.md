# Local MLX deployment — end-to-end smoke checklist

This is a manual gate that must pass before declaring the local MLX
backend "deployment ready" on a new machine. It complements the
automated tests in ``autodiscover/backends/test_mlx_*`` (which only
exercise tiny models / mocked HTTP).

Pre-reqs:
- ``[mlx]`` extra installed (``uv sync --extra mlx``).
- ``gpt-oss-120b-MXFP4`` weights pre-staged per
  ``docs/local-deployment.md`` §2.
- A tiny *test problem* under ``./smoke-problem/`` containing whatever
  context you want to feed the orchestrator. A fast-evaluator script
  (e.g. ``eval.sh`` that prints ``reward=-len(plan_text)``) keeps the
  loop snappy.
- Two terminals (one for the sampler, one for the trainer + orchestrator).
- Default loopback ports free: ``8081`` (sampler), ``8123`` (trainer).

Mark each step as you go: replace ``[ ]`` with ``[x]`` on success, or
``[!]`` plus a short note on failure. **The smoke is only "passed" when
every step is ``[x]``.**

---

## 1. Pre-stage the model (one-time)

- [ ] Confirm ``HF_HOME`` is set to a directory that already contains
      ``mlx-community/gpt-oss-120b-MXFP4`` (no network access during the
      smoke). See ``docs/local-deployment.md`` §2.

## 2. Launch the sampler

- [ ] In terminal A:
      ```
      uv run mlx_lm.server \
          --model mlx-community/gpt-oss-120b-MXFP4 \
          --adapter-path ./adapters/current.safetensors \
          --host 127.0.0.1 --port 8081
      ```
      (Adapter file does not need to exist for the very first launch —
      pass ``--adapter-path`` only after the first save_checkpoint
      writes ``adapters/current.safetensors``. Otherwise launch without
      ``--adapter-path``.)

## 3. Smoke the sampler

- [ ] ``curl http://127.0.0.1:8081/v1/completions \
        -H 'content-type: application/json' \
        -d '{"prompt":"hello","max_tokens":4,"temperature":0,"logprobs":true}'``
      returns a 200 with non-empty ``choices[0].text``.

## 4. Launch the trainer with the MLX backend

- [ ] In terminal B:
      ```
      uv run python -m autodiscover.cli.trainer_server \
          --backend mlx_local \
          --mlx-sampler-url http://127.0.0.1:8081 \
          --mlx-adapter-dir ./adapters \
          --mlx-model mlx-community/gpt-oss-120b-MXFP4 \
          --num-epochs 1 --group-size 4 --groups-per-batch 2
      ```
- [ ] Capture the printed ``AUTODISCOVER_SERVER_URL`` (e.g.
      ``http://127.0.0.1:8123``) and ``export`` it in a third
      terminal where the orchestrator lives.

## 5. Run /discover on a tiny test problem

- [ ] ``cd ./smoke-problem/``
- [ ] In a Claude Code session: ``/discover <one-line goal>``
- [ ] Watch ``GET $AUTODISCOVER_SERVER_URL/status`` periodically.

## 6. Wait for completion

- [ ] ``curl $AUTODISCOVER_SERVER_URL/status`` returns ``done: true``.
- [ ] The trainer terminal shows the loss line for batch 1.

## 7. Verify rewards persisted

- [ ] ``./tinker_log/archive.csv`` has exactly 1 + (G * P) =
      1 + 4 * 2 = **5** rewarded rows (1 seed + 4 sampled children for
      this G=4, P=2 single-batch config).
      ```
      wc -l ./tinker_log/archive.csv  # header + 5 rows = 6
      ```

## 8. Verify adapter checkpoint

- [ ] ``./adapters/current.safetensors`` exists.
- [ ] Its mtime is within the last minute (``stat -f '%m %N' ./adapters/current.safetensors``).
- [ ] ``./adapters/iter_000001.safetensors`` (or similar) exists as
      the versioned snapshot.

## 9. Verify sampler picked up the new adapter

- [ ] Start a *second* ``/discover`` against the same problem.
- [ ] Compare the first-token of the new completion to the first run —
      they should differ (different policy ⇒ different first token at
      temperature > 0 with the same prompt). If they match, the sampler
      may still be serving the pre-adapter base model; check the
      sampler's logs for an ``adapter reloaded`` line.

## 10. Verify zero non-loopback traffic during the run

- [ ] While step 5 is running:
      ```
      lsof -i -nP | grep -v '127\.0\.0\.1' | grep -v '::1' | grep -v LISTEN
      ```
      should list nothing tied to ``python`` / ``mlx_lm`` / the trainer
      process. (The Claude Code session itself does reach
      ``api.anthropic.com``; ignore those rows but verify no MLX /
      trainer / autodiscover process appears.)

---

## Tinker regression (optional — only if you have Tinker access)

- [ ] Re-run the same smoke problem with ``--backend tinker``.
- [ ] Compare reward distribution against the MLX run. A large gap
      (>2x mean-reward difference at G=4, P=2) is a red flag — open
      a bug. Otherwise the two backends are considered equivalent for
      the smoke set.

If Tinker access is unavailable, document the comparison the customer
should run in their internal CI before any future change that could
affect the training loop (loss math, advantage estimation, batch
construction).
