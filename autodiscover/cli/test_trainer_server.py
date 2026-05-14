"""Integration test for /rollout/begin and /rollout/reward.

Drives the FastAPI app with a stubbed Tinker sampler so we can exercise
the PUCT archive wiring without booting a real training loop.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import tinker
from fastapi.testclient import TestClient

from autodiscover.cli import trainer_server
from autodiscover.config import AutoDiscoverConfig
from autodiscover.server.sampling import SampledPlan
from autodiscover.server.state import RolloutStore


def _make_app(tmp_path: Path, *, group_size: int, groups_per_batch: int):
    cfg = AutoDiscoverConfig()
    cfg.group_size = group_size
    cfg.groups_per_batch = groups_per_batch
    cfg.phase1_max_tokens = 16
    cfg.temperature = 1.0

    store = RolloutStore(group_size=group_size, groups_per_batch=groups_per_batch)

    fake_loop = AsyncMock()
    fake_loop.batches_trained = 0
    fake_loop.done = False
    fake_loop.current_sampling_client = AsyncMock(return_value=object())
    fake_loop._on_metrics = lambda m: None

    archive_csv_path = tmp_path / "archive.csv"
    app = trainer_server.build_app(
        cfg=cfg,
        store=store,
        training_loop=fake_loop,
        renderer=None,
        tokenizer=None,
        stop_condition=["</plan>"],
        archive_csv_path=archive_csv_path,
    )
    return app, store, archive_csv_path


def _stub_sample_plans(plan_texts: list[str]):
    """Return a coroutine that mimics autodiscover.server.sampling.sample_plans."""
    async def _fake(**kwargs):
        plans = [
            SampledPlan(
                tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3],
                plan_text=t, parse_success=1.0,
            )
            for t in plan_texts
        ]
        return tinker.ModelInput.empty(), plans
    return _fake


def test_cold_start_without_seed_returns_400(tmp_path: Path):
    app, store, _ = _make_app(tmp_path, group_size=2, groups_per_batch=1)
    client = TestClient(app)
    resp = client.post("/rollout/begin", json={"iter_idx": 0, "context": "ctx"})
    assert resp.status_code == 400
    assert "step 2.a" in resp.json()["detail"]


def test_seeded_begin_picks_seed_and_returns_archive_csv(tmp_path: Path):
    app, store, archive_path = _make_app(tmp_path, group_size=2, groups_per_batch=1)
    seed_csv = (
        "plan_id,iter_idx,parent_id,reward,plan_text\n"
        "seed,-1,,0.0,baseline plan body\n"
    )
    with patch("autodiscover.cli.trainer_server.sample_plans",
               new=_stub_sample_plans(["plan A", "plan B"])):
        client = TestClient(app)
        resp = client.post(
            "/rollout/begin",
            json={"iter_idx": 0, "context": "ctx", "archive_csv": seed_csv},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["parent_plan_id"] == "seed"
    assert len(body["plans"]) == 2
    assert body["plans"][0]["plan_id"] == "0000_00_00"
    # Archive CSV now has seed + 2 freshly-minted children (rewards empty).
    assert "0000_00_00" in body["archive_csv"]
    assert "0000_00_01" in body["archive_csv"]
    # On-disk persistence happened.
    assert archive_path.exists()
    assert "seed" in archive_path.read_text()
    # Counters updated: T=1, n(seed)=1.
    assert store.archive.T == 1
    assert store.archive.n("seed") == 1


def test_full_iteration_loop_updates_counters_and_picks_child(tmp_path: Path):
    app, store, _ = _make_app(tmp_path, group_size=2, groups_per_batch=1)
    seed_csv = (
        "plan_id,iter_idx,parent_id,reward,plan_text\n"
        "seed,-1,,0.0,baseline\n"
    )
    client = TestClient(app)

    # Iter 0: PUCT picks seed; submit two rewards.
    with patch("autodiscover.cli.trainer_server.sample_plans",
               new=_stub_sample_plans(["A", "B"])):
        r0 = client.post("/rollout/begin",
                         json={"iter_idx": 0, "context": "ctx", "archive_csv": seed_csv})
    assert r0.status_code == 200
    csv_after_iter0 = r0.json()["archive_csv"]
    plan_ids_iter0 = [p["plan_id"] for p in r0.json()["plans"]]

    rw = client.post("/rollout/reward", json={"rewards": [
        {"plan_id": plan_ids_iter0[0], "results": {"reward": 0.4}},
        {"plan_id": plan_ids_iter0[1], "results": {"reward": 0.9}},
    ]})
    assert rw.status_code == 200
    body = rw.json()
    assert [r["accepted"] for r in body["results"]] == [True, True]
    assert body["groups_completed"] == 1
    assert body["batches_completed"] == 1
    assert body["group_complete"] is True
    assert body["batch_complete"] is True
    # m(seed) should now reflect the best child.
    assert store.archive.m("seed") == 0.9

    # Iter 1: orchestrator uploads its current archive_csv (which is the
    # server's last response). PUCT should now have a non-seed candidate
    # available and may pick it. Either way, the call must succeed.
    with patch("autodiscover.cli.trainer_server.sample_plans",
               new=_stub_sample_plans(["C", "D"])):
        r1 = client.post("/rollout/begin",
                         json={"iter_idx": 1, "context": "ctx", "archive_csv": csv_after_iter0})
    assert r1.status_code == 200
    body = r1.json()
    # Two expansions total now: iter 0 and iter 1.
    assert store.archive.T == 2
    assert body["parent_plan_id"] in {"seed", *plan_ids_iter0}


def test_reward_rejects_duplicate_plan_id_in_batch(tmp_path: Path):
    app, _, _ = _make_app(tmp_path, group_size=2, groups_per_batch=1)
    client = TestClient(app)
    resp = client.post("/rollout/reward", json={"rewards": [
        {"plan_id": "0000_00_00", "results": {"reward": 0.1}},
        {"plan_id": "0000_00_00", "results": {"reward": 0.2}},
    ]})
    assert resp.status_code == 422
    assert "duplicate plan_id" in resp.text


def test_reward_unknown_plan_id_is_per_item_not_404(tmp_path: Path):
    app, store, _ = _make_app(tmp_path, group_size=2, groups_per_batch=1)
    seed_csv = (
        "plan_id,iter_idx,parent_id,reward,plan_text\n"
        "seed,-1,,0.0,baseline\n"
    )
    client = TestClient(app)
    with patch("autodiscover.cli.trainer_server.sample_plans",
               new=_stub_sample_plans(["A", "B"])):
        r0 = client.post("/rollout/begin",
                         json={"iter_idx": 0, "context": "ctx", "archive_csv": seed_csv})
    assert r0.status_code == 200
    plan_ids = [p["plan_id"] for p in r0.json()["plans"]]

    rw = client.post("/rollout/reward", json={"rewards": [
        {"plan_id": plan_ids[0], "results": {"reward": 0.3}},
        {"plan_id": "not_a_real_id", "results": {"reward": 0.4}},
        {"plan_id": plan_ids[1], "results": {"reward": 0.5}},
    ]})
    assert rw.status_code == 200
    body = rw.json()
    accepted = [r["accepted"] for r in body["results"]]
    assert accepted == [True, False, True]
    assert "unknown plan_id" in body["results"][1]["error"]
    # Group still closed by the two valid items.
    assert body["groups_completed"] == 1
