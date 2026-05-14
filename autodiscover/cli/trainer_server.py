"""FastAPI trainer server.

Boot:
    uv run python -m autodiscover.cli.trainer_server \\
        --server-host 127.0.0.1 --server-port 8123 \\
        --group-size 64 --groups-per-batch 8 --num-epochs 50

The server is problem-agnostic: it does not read any problem files at
startup. Per-iteration context arrives over the wire on
``POST /rollout/begin`` (``BeginRequest.context``).

On boot:
- Connects to Tinker, creates training + sampling clients.
- Prints ``AUTODISCOVER_SERVER_URL=http://host:port`` to stdout so the
  operator can paste it into their environment.
- Starts the TrainingLoop coroutine in the background.

The server keeps an in-memory result store of all accepted rewards (in
RolloutStore), which is what ``GET /best`` returns.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import socket
import sys
from pathlib import Path

import tinker
import uvicorn
from fastapi import FastAPI, HTTPException, Query

from autodiscover.sampling.completers import TokensWithLogprobs
from autodiscover.tokenizer import get_tokenizer
from autodiscover.renderers import get_renderer

from autodiscover.config import AutoDiscoverConfig
from autodiscover.server.api import (
    BeginRequest, BeginResponse, BestResponse, BestRow, PlanOut,
    RewardItemResult, RewardRequest, RewardResponse, StatusResponse,
)
from autodiscover.server.sampling import sample_plans
from autodiscover.server.state import OutstandingPlan, RolloutStore, make_plan_id
from autodiscover.server.training import TrainingLoop

logger = logging.getLogger("autodiscover.trainer_server")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server-host", default="127.0.0.1")
    p.add_argument("--server-port", type=int, default=0)
    p.add_argument("--log-dir", default="./tinker_log",
                   help="Where TrainingLoop writes checkpoints + optional debug traces.")
    # training overrides (subset of AutoDiscoverConfig)
    p.add_argument("--group-size", type=int)
    p.add_argument("--groups-per-batch", type=int)
    p.add_argument("--num-epochs", type=int)
    p.add_argument("--lr", type=float, dest="learning_rate")
    p.add_argument("--lora-rank", type=int)
    p.add_argument("--kl-penalty-coef", type=float)
    p.add_argument("--temperature", type=float)
    p.add_argument("--phase1-max-tokens", type=int)
    p.add_argument("--save-every", type=int)
    p.add_argument("--model-name", type=str)
    p.add_argument("--renderer-name", type=str)
    p.add_argument("--wandb-project", type=str, default=None)
    return p.parse_args(argv)


def build_cfg(args: argparse.Namespace) -> AutoDiscoverConfig:
    cfg = AutoDiscoverConfig()
    cfg.server_host = args.server_host
    cfg.server_port = args.server_port
    for fld in (
        "group_size", "groups_per_batch", "num_epochs", "learning_rate", "lora_rank",
        "kl_penalty_coef", "temperature", "phase1_max_tokens", "save_every",
        "model_name", "renderer_name", "wandb_project",
    ):
        v = getattr(args, fld, None)
        if v is not None:
            setattr(cfg, fld, v)
    return cfg


async def _make_tinker_clients(cfg: AutoDiscoverConfig):
    service_client = tinker.ServiceClient(base_url=None)
    base_sampling_client = service_client.create_sampling_client(base_model=cfg.model_name)
    training_client = await service_client.create_lora_training_client_async(
        cfg.model_name, rank=cfg.lora_rank,
    )
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    return service_client, training_client, base_sampling_client, sampling_client


def build_app(
    *,
    cfg: AutoDiscoverConfig,
    store: RolloutStore,
    training_loop: TrainingLoop,
    renderer,
    tokenizer,
    stop_condition,
    archive_csv_path: Path,
) -> FastAPI:
    app = FastAPI(title="autodiscover trainer", version="3")
    state: dict = {
        "last_metrics": {},
        "last_iter_idx": -1,
    }

    def _persist_archive() -> None:
        archive_csv_path.write_text(store.archive.to_csv())

    @app.get("/status", response_model=StatusResponse)
    async def status() -> StatusResponse:
        m = state["last_metrics"]
        return StatusResponse(
            ok=True,
            iter=state.get("last_iter_idx", -1) + 1,
            batches_trained=training_loop.batches_trained,
            outstanding=store.outstanding,
            last_reward_mean=m.get("reward_mean"),
            done=training_loop.done,
        )

    @app.post("/rollout/begin", response_model=BeginResponse)
    async def begin(req: BeginRequest) -> BeginResponse:
        # 1. Merge incoming archive (orchestrator's local results.csv) into
        #    the server-side archive. Server values win on conflict; this
        #    is how a fresh server is seeded from the orchestrator's
        #    persistent local file across restarts.
        if req.archive_csv:
            store.archive.merge_csv(req.archive_csv)

        # 2. Cold-start guard: iter 0 with no rewarded rows means the
        #    orchestrator skipped the seed-bootstrap step. Fail loudly
        #    rather than silently inventing an empty seed.
        rewarded_rows = [r for r in store.archive.rows() if r.reward is not None]
        if not rewarded_rows:
            raise HTTPException(
                400,
                "archive is empty; the orchestrator must seed ./results.csv "
                "with a baseline row before the first /rollout/begin "
                "(see .claude/commands/discover.md step 2.a).",
            )

        # 3. PUCT-select a parent. Lineage blocking (per-batch) ensures
        #    diversity across the P groups that fill one training batch.
        parent_pid = store.archive.puct_select()
        parent_row = store.archive.get(parent_pid)
        store.archive.block_lineage(parent_pid)

        # 4. Sample G children conditioned on the parent.
        sampling_client = await training_loop.current_sampling_client()
        prompt_input, plans = await sample_plans(
            sampling_client=sampling_client,
            tokenizer=tokenizer,
            renderer=renderer,
            context=req.context,
            G=cfg.group_size,
            phase1_max_tokens=cfg.phase1_max_tokens,
            temperature=cfg.temperature,
            stop_condition=stop_condition,
            parent_plan_text=parent_row.plan_text,
            parent_reward=parent_row.reward,
        )

        # 5. Register plans with the store + archive (one /rollout/begin == one group).
        group_idx = await store.started_groups_for_iter(req.iter_idx)
        out: list[PlanOut] = []
        for traj_idx, sp in enumerate(plans):
            plan_id = make_plan_id(req.iter_idx, group_idx, traj_idx)
            await store.register_plan(OutstandingPlan(
                plan_id=plan_id,
                iter_idx=req.iter_idx,
                group_idx=group_idx,
                traj_idx=traj_idx,
                prompt_input=prompt_input,
                action=TokensWithLogprobs(
                    tokens=sp.tokens, maybe_logprobs=sp.logprobs, maybe_mask=None,
                ),
                plan_text=sp.plan_text,
                parent_id=parent_pid,
            ))
            out.append(PlanOut(
                plan_id=plan_id, plan_text=sp.plan_text, iter_idx=req.iter_idx,
            ))
        # 6. Record the expansion: T += 1, n(parent) and ancestors += 1.
        store.archive.record_expansion(
            parent_id=parent_pid, iter_idx=req.iter_idx, group_idx=group_idx,
        )
        _persist_archive()

        state["last_iter_idx"] = req.iter_idx
        return BeginResponse(
            plans=out,
            parent_plan_id=parent_pid,
            archive_csv=store.archive.to_csv(),
        )

    @app.post("/rollout/reward", response_model=RewardResponse)
    async def reward(req: RewardRequest) -> RewardResponse:
        items = [(r.plan_id, r.results) for r in req.rewards]
        try:
            per_item, groups_completed, batches_completed = (
                await store.submit_rewards_batch(items)
            )
        finally:
            # One fsync per batch — keep on-disk consistent with in-memory
            # even if some items inside the batch errored.
            _persist_archive()
        return RewardResponse(
            results=[RewardItemResult(**r) for r in per_item],
            group_complete=groups_completed > 0,
            batch_complete=batches_completed > 0,
            groups_completed=groups_completed,
            batches_completed=batches_completed,
        )

    @app.get("/best", response_model=BestResponse)
    async def best(k: int = Query(5, ge=1, le=100)) -> BestResponse:
        rows = await store.top_k(k=k, maximize=True)
        return BestResponse(rows=[BestRow(**r) for r in rows])

    # Wire training_loop metrics callback to the in-app status mirror.
    training_loop._on_metrics = lambda m: state.__setitem__("last_metrics", m)
    return app


async def amain(args: argparse.Namespace) -> int:
    cfg = build_cfg(args)
    log_dir = Path(args.log_dir or "./tinker_log")
    log_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = get_tokenizer(cfg.model_name)
    renderer = get_renderer(cfg.renderer_name, tokenizer)
    # Stop on the renderer's natural EOS (e.g. <|return|> for gpt-oss harmony).
    # We deliberately do *not* stop on "</plan>": with a reasoning-channel model
    # the CoT often paraphrases the system prompt and emits "</plan>" inside
    # the analysis channel, which would halt sampling before the final channel
    # ever opens. "</plan>" is enforced post-hoc in _strip_harmony_markup.
    stop_condition = renderer.get_stop_sequences()

    _, training_client, base_sampling, sampling = await _make_tinker_clients(cfg)

    store = RolloutStore(group_size=cfg.group_size, groups_per_batch=cfg.groups_per_batch)
    # Persistent archive lives under log_dir so it survives restarts and
    # an operator can inspect the tree across iterations. The orchestrator
    # also keeps a mirror in its own ./results.csv via the bidirectional
    # merge in /rollout/begin.
    archive_csv_path = log_dir / "archive.csv"
    if archive_csv_path.exists():
        store.archive.merge_csv(archive_csv_path.read_text())
        logger.info("loaded archive from %s (%d rows)",
                    archive_csv_path, len(store.archive))
    loop = TrainingLoop(
        store=store,
        training_client=training_client,
        base_sampling_client=base_sampling,
        sampling_client=sampling,
        log_dir=log_dir,
        learning_rate=cfg.learning_rate,
        kl_penalty_coef=cfg.kl_penalty_coef,
        adv_estimator=cfg.adv_estimator,
        adv_estimator_beta=cfg.adv_estimator_beta,
        loss_fn=cfg.loss_fn,
        num_substeps=cfg.num_substeps,
        save_every=cfg.save_every,
        num_epochs=cfg.num_epochs,
    )

    app = build_app(
        cfg=cfg, store=store, training_loop=loop,
        renderer=renderer, tokenizer=tokenizer, stop_condition=stop_condition,
        archive_csv_path=archive_csv_path,
    )

    # Bind first so server-port=0 produces a known port BEFORE uvicorn starts
    # listening — otherwise the orchestrator could race uvicorn's startup.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((cfg.server_host, cfg.server_port))
    host, port = sock.getsockname()
    print(f"AUTODISCOVER_SERVER_URL=http://{host}:{port}", flush=True)

    config = uvicorn.Config(app=app, fd=sock.fileno(), log_level="info")
    server = uvicorn.Server(config)

    train_task = asyncio.create_task(loop.run())
    try:
        await server.serve()
    finally:
        loop.stop()
        try:
            await asyncio.wait_for(train_task, timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("training loop did not stop within 10s; cancelling")
            train_task.cancel()
            try:
                await train_task
            except (asyncio.CancelledError, Exception):
                pass

    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
