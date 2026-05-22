"""Unit tests for autodiscover.server.state.RolloutStore."""
from __future__ import annotations

import asyncio

import pytest

from autodiscover.backends.types import TokenSequence
from autodiscover.server.state import (
    OutstandingPlan,
    RolloutStore,
    make_plan_id,
)
from autodiscover.types import TokensWithLogprobs


def _make_plan(iter_idx: int, group_idx: int, traj_idx: int) -> OutstandingPlan:
    return OutstandingPlan(
        plan_id=make_plan_id(iter_idx, group_idx, traj_idx),
        iter_idx=iter_idx,
        group_idx=group_idx,
        traj_idx=traj_idx,
        prompt_input=TokenSequence(tokens=[]),
        action=TokensWithLogprobs(
            tokens=[traj_idx + 1, group_idx + 10],
            maybe_logprobs=[-0.1, -0.2],
            maybe_mask=None,
        ),
        plan_text=f"plan-{iter_idx}-{group_idx}-{traj_idx}",
    )


@pytest.mark.asyncio
async def test_out_of_order_rewards_complete_groups_and_batch():
    store = RolloutStore(group_size=2, groups_per_batch=2)

    # 4 plans across 2 groups (G=2), iter=0, groups 0 and 1.
    plans = [
        _make_plan(0, 0, 0),
        _make_plan(0, 0, 1),
        _make_plan(0, 1, 0),
        _make_plan(0, 1, 1),
    ]
    for p in plans:
        await store.register_plan(p)

    assert store.outstanding == 4

    # Batch 1 interleaves the two groups so neither completes.
    per_item, gc, bc = await store.submit_rewards_batch([
        (plans[0].plan_id, {"reward": 1.0}),  # g0 t0
        (plans[2].plan_id, {"reward": 3.0}),  # g1 t0
    ])
    assert all(r["accepted"] for r in per_item)
    assert (gc, bc) == (0, 0)
    assert store.outstanding == 2

    # Batch 2 closes both groups; with P=2 the batch is flushed.
    per_item, gc, bc = await store.submit_rewards_batch([
        (plans[1].plan_id, {"reward": 2.0}),  # g0 t1 -> group 0 done
        (plans[3].plan_id, {"reward": 4.0}),  # g1 t1 -> group 1 done, batch flushes
    ])
    assert all(r["accepted"] for r in per_item)
    assert (gc, bc) == (2, 1)
    assert store.outstanding == 0

    # next_batch should return immediately with 2 groups, each with 2 trajectories
    batch = await asyncio.wait_for(store.next_batch(), timeout=1.0)
    assert len(batch) == 2
    for tg in batch:
        assert len(tg.trajectories_G) == 2
        for traj in tg.trajectories_G:
            assert len(traj.transitions) == 1
        assert tg.final_rewards_G == [0.0, 0.0]

    flat_rewards = sorted(
        traj.transitions[0].reward
        for tg in batch
        for traj in tg.trajectories_G
    )
    assert flat_rewards == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.asyncio
async def test_top_k_round_trips_results_dict():
    store = RolloutStore(group_size=1, groups_per_batch=1)
    plan = _make_plan(0, 0, 0)
    await store.register_plan(plan)

    payload = {"reward": 0.7, "msg": "ok", "extra": [1, 2]}
    await store.submit_rewards_batch([(plan.plan_id, payload)])

    rows = await store.top_k(1, maximize=True)
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {"plan_id", "iter_idx", "reward", "results"}
    assert row["plan_id"] == plan.plan_id
    assert row["iter_idx"] == 0
    assert row["reward"] == 0.7
    assert row["results"] == {"reward": 0.7, "msg": "ok", "extra": [1, 2]}


@pytest.mark.asyncio
async def test_submit_reward_defaults_missing_or_bad_reward_to_zero():
    store = RolloutStore(group_size=2, groups_per_batch=1)
    p0 = _make_plan(0, 0, 0)
    p1 = _make_plan(0, 0, 1)
    await store.register_plan(p0)
    await store.register_plan(p1)

    per_item, _, _ = await store.submit_rewards_batch([
        (p0.plan_id, {"msg": "no reward field"}),
        (p1.plan_id, {"reward": "not-a-number"}),
    ])
    assert all(r["accepted"] for r in per_item)

    rows = await store.top_k(2, maximize=True)
    assert [r["reward"] for r in rows] == [0.0, 0.0]


@pytest.mark.asyncio
async def test_unknown_plan_id_in_batch_is_skipped_not_fatal():
    store = RolloutStore(group_size=2, groups_per_batch=1)
    p0 = _make_plan(0, 0, 0)
    p1 = _make_plan(0, 0, 1)
    await store.register_plan(p0)
    await store.register_plan(p1)

    per_item, gc, bc = await store.submit_rewards_batch([
        (p0.plan_id, {"reward": 0.5}),
        ("does_not_exist", {"reward": 9.9}),
        (p1.plan_id, {"reward": 0.6}),
    ])
    # Order is preserved across accepted + rejected.
    assert per_item[0]["accepted"] is True
    assert per_item[1]["accepted"] is False
    assert "unknown plan_id" in per_item[1]["error"]
    assert per_item[2]["accepted"] is True
    # The good two complete the only group -> 1 group, batch flush (P=1).
    assert (gc, bc) == (1, 1)
    assert store.outstanding == 0
