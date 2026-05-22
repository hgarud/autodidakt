"""In-memory store mapping plan_ids to outstanding rollouts and feeding
completed TrajectoryGroups to the training batch queue.

Concurrency model: single-process asyncio. All public methods are async;
internal mutation is guarded by an asyncio.Lock to avoid races between the
sampler coroutine (which inserts) and the reward handler (which finalizes).
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from autodiscover.backends.types import TokenSequence
from autodiscover.types import Trajectory, TrajectoryGroup, Transition, TokensWithLogprobs

from autodiscover.server.archive import Archive, ArchiveEntry


@dataclass
class OutstandingPlan:
    plan_id: str
    iter_idx: int
    group_idx: int
    traj_idx: int
    prompt_input: TokenSequence
    action: TokensWithLogprobs
    plan_text: str
    # Parent plan_id chosen by PUCT for the group this plan belongs to.
    # Defaults to "" (seed) so the existing unit tests, which exercise the
    # store directly without going through /rollout/begin, keep working.
    parent_id: str = ""


@dataclass
class _GroupAccum:
    iter_idx: int
    group_idx: int
    target_G: int
    trajectories: list[Trajectory] = field(default_factory=list)
    final_rewards: list[float] = field(default_factory=list)
    metrics: list[dict] = field(default_factory=list)


def make_plan_id(iter_idx: int, group_idx: int, traj_idx: int) -> str:
    return f"{iter_idx:04d}_{group_idx:02d}_{traj_idx:02d}"


class RolloutStore:
    def __init__(self, *, group_size: int, groups_per_batch: int):
        self._G = group_size
        self._P = groups_per_batch
        self._lock = asyncio.Lock()

        # Outstanding plans (sampled, awaiting reward)
        self._outstanding: dict[str, OutstandingPlan] = {}

        # In-flight groups (some plans rewarded, group not yet full)
        # Key: (iter_idx, group_idx)
        self._groups: dict[tuple[int, int], _GroupAccum] = {}

        # Completed groups awaiting batch fill
        self._pending_groups: list[TrajectoryGroup] = []

        # Asyncio queue: each item is a list[TrajectoryGroup] of length P (one batch)
        self._batches: asyncio.Queue[list[TrajectoryGroup]] = asyncio.Queue()

        # Counter of groups started per iter_idx — used by /rollout/begin to
        # mint a fresh group_idx for each call (one group per /rollout/begin).
        self._started_groups: dict[int, int] = {}

        # Canonical result store: every accepted reward, in submission order.
        # Read by /best — keeps the buffer server-side rather than depending
        # on the orchestrator's filesystem.
        self._results: list[dict] = []

        # PUCT archive: tree of (plan_id, parent_id, reward, plan_text) used
        # by /rollout/begin to pick a parent. Maintained alongside _results
        # because the archive carries minimal columns (paper-faithful) while
        # /best wants the full results dict.
        self.archive = Archive()

    @property
    def outstanding(self) -> int:
        return len(self._outstanding)

    async def started_groups_for_iter(self, iter_idx: int) -> int:
        """Return how many distinct groups have been started for this iter.

        Increments by one per new (iter_idx, group_idx) seen in register_plan.
        The trainer server uses this to allocate the next group_idx for a
        fresh /rollout/begin call.
        """
        async with self._lock:
            return self._started_groups.get(iter_idx, 0)

    async def register_plan(self, plan: OutstandingPlan) -> None:
        async with self._lock:
            self._outstanding[plan.plan_id] = plan
            key = (plan.iter_idx, plan.group_idx)
            if key not in self._groups:
                self._groups[key] = _GroupAccum(
                    iter_idx=plan.iter_idx,
                    group_idx=plan.group_idx,
                    target_G=self._G,
                )
                self._started_groups[plan.iter_idx] = (
                    self._started_groups.get(plan.iter_idx, 0) + 1
                )
            # Mirror into the PUCT archive (reward not yet known).
            self.archive.add_row(ArchiveEntry(
                plan_id=plan.plan_id,
                iter_idx=plan.iter_idx,
                parent_id=plan.parent_id,
                reward=None,
                plan_text=plan.plan_text,
            ))

    async def submit_rewards_batch(
        self, items: list[tuple[str, dict]],
    ) -> tuple[list[dict], int, int]:
        """Submit a batch of (plan_id, results) pairs atomically.

        Returns ``(per_item_results, groups_completed, batches_completed)``
        where each per-item entry is shaped ``{plan_id, accepted, error}``.
        Unknown plan_ids are surfaced as ``accepted=False`` with an error
        string — they do not abort the batch. ``results`` is the free-form
        payload from the orchestrator's subagent; the server reads
        ``results["reward"]`` (float; defaults to 0.0 if missing or non-numeric)
        for the RL step. Items are processed in submission order; archive cap
        maintenance is order-sensitive across multiple group completions.
        """
        per_item: list[dict] = []
        groups_completed = 0
        batches_completed = 0
        async with self._lock:
            for plan_id, results in items:
                try:
                    plan = self._outstanding.pop(plan_id)
                except KeyError:
                    per_item.append({
                        "plan_id": plan_id,
                        "accepted": False,
                        "error": f"unknown plan_id {plan_id}",
                    })
                    continue
                try:
                    try:
                        reward = float(results.get("reward", 0.0))
                    except (TypeError, ValueError):
                        reward = 0.0
                    self._results.append({
                        "plan_id": plan_id,
                        "iter_idx": plan.iter_idx,
                        "reward": reward,
                        "results": dict(results),
                    })
                    self.archive.set_reward(plan_id, reward)
                    transition = Transition(
                        ob=plan.prompt_input,
                        ac=plan.action,
                        reward=reward,
                        episode_done=True,
                        metrics=results,
                    )
                    traj = Trajectory(
                        transitions=[transition],
                        final_ob=TokenSequence(tokens=[]),
                    )
                    key = (plan.iter_idx, plan.group_idx)
                    accum = self._groups[key]
                    accum.trajectories.append(traj)
                    accum.final_rewards.append(0.0)
                    accum.metrics.append(results)

                    if len(accum.trajectories) == accum.target_G:
                        tg = TrajectoryGroup(
                            trajectories_G=accum.trajectories,
                            final_rewards_G=accum.final_rewards,
                            metrics_G=accum.metrics,
                        )
                        self._pending_groups.append(tg)
                        del self._groups[key]
                        # Paper §A.2 archive maintenance per completed group.
                        if plan.parent_id:
                            self.archive.enforce_child_cap(plan.parent_id)
                        self.archive.enforce_global_cap()
                        groups_completed += 1
                        if len(self._pending_groups) >= self._P:
                            batch = self._pending_groups[: self._P]
                            self._pending_groups = self._pending_groups[self._P :]
                            await self._batches.put(batch)
                            batches_completed += 1
                            # Lineage block is per-batch (paper §A.2 (iv)).
                            self.archive.clear_blocked()
                    per_item.append({
                        "plan_id": plan_id, "accepted": True, "error": None,
                    })
                except Exception as e:
                    per_item.append({
                        "plan_id": plan_id, "accepted": False, "error": str(e),
                    })
        return per_item, groups_completed, batches_completed

    async def next_batch(self) -> list[TrajectoryGroup]:
        return await self._batches.get()

    async def drain_pending(self) -> list[list[TrajectoryGroup]]:
        """Flush any non-full pending groups as undersized batches (used at shutdown)."""
        async with self._lock:
            out: list[list[TrajectoryGroup]] = []
            if self._pending_groups:
                out.append(list(self._pending_groups))
                self._pending_groups.clear()
            return out

    async def top_k(self, k: int, *, maximize: bool = True) -> list[dict]:
        """Return the top-k accepted rewards, sorted by reward.

        Each row is shaped for ``BestRow``:
        ``{plan_id, iter_idx, reward, results}``.
        """
        async with self._lock:
            rows = sorted(
                self._results, key=lambda r: r["reward"], reverse=maximize
            )
            return [dict(r) for r in rows[:k]]
