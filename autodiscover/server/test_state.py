"""Unit tests for autodiscover.server.state.RolloutStore."""
from __future__ import annotations

import asyncio

import pytest
import tinker

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
        prompt_input=tinker.ModelInput.empty(),
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

    # Reward order interleaves the two groups so neither completes first
    # until two same-group rewards arrive.
    rewards = {
        plans[0].plan_id: 1.0,  # g0 t0
        plans[2].plan_id: 3.0,  # g1 t0
        plans[1].plan_id: 2.0,  # g0 t1 -> g0 complete, batch not yet
        plans[3].plan_id: 4.0,  # g1 t1 -> g1 complete, batch complete
    }

    # 1st reward: neither complete
    gc, bc = await store.submit_reward(
        plans[0].plan_id, {"reward": rewards[plans[0].plan_id], "r": 1.0}
    )
    assert (gc, bc) == (False, False)
    assert store.outstanding == 3

    # 2nd reward (different group): neither complete
    gc, bc = await store.submit_reward(
        plans[2].plan_id, {"reward": rewards[plans[2].plan_id], "r": 3.0}
    )
    assert (gc, bc) == (False, False)
    assert store.outstanding == 2

    # 3rd reward: completes group 0, batch not yet (only 1 of 2 groups done)
    gc, bc = await store.submit_reward(
        plans[1].plan_id, {"reward": rewards[plans[1].plan_id], "r": 2.0}
    )
    assert (gc, bc) == (True, False)
    assert store.outstanding == 1

    # 4th reward: completes group 1, fills batch (P=2)
    gc, bc = await store.submit_reward(
        plans[3].plan_id, {"reward": rewards[plans[3].plan_id], "r": 4.0}
    )
    assert (gc, bc) == (True, True)
    assert store.outstanding == 0

    # next_batch should return immediately with 2 groups, each with 2 trajectories
    batch = await asyncio.wait_for(store.next_batch(), timeout=1.0)
    assert len(batch) == 2
    for tg in batch:
        assert len(tg.trajectories_G) == 2
        for traj in tg.trajectories_G:
            assert len(traj.transitions) == 1
        assert tg.final_rewards_G == [0.0, 0.0]

    # Rewards from group 0 should appear in some trajectory of one TG, group 1 in the other.
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
    await store.submit_reward(plan.plan_id, payload)

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

    await store.submit_reward(p0.plan_id, {"msg": "no reward field"})
    await store.submit_reward(p1.plan_id, {"reward": "not-a-number"})

    rows = await store.top_k(2, maximize=True)
    assert [r["reward"] for r in rows] == [0.0, 0.0]
