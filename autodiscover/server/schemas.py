"""Wire schemas for the trainer FastAPI server.

The orchestrator (Claude Code under /discover) and the trainer server
(autodiscover.cli.trainer_server) both import this module. Keep it free of
heavy deps so it can be imported without Tinker / Ray.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


# ---------- /rollout/begin ----------
class BeginRequest(BaseModel):
    iter_idx: int = Field(..., ge=0)
    context: str = Field(..., description="Per-iteration context the orchestrator "
                                          "gathered. Becomes the user-message body.")
    archive_csv: str | None = Field(
        default=None,
        description="The orchestrator's local results.csv contents. The server "
                    "merges these rows into its archive (server values win on "
                    "conflict). On a cold start with iter_idx=0 the orchestrator "
                    "must include at least the seed row; see discover.md step 2.a.",
    )


class PlanOut(BaseModel):
    plan_id: str         # f"{iter:04d}_{group:02d}_{traj:02d}" (server-minted; opaque to caller)
    plan_text: str
    iter_idx: int


class BeginResponse(BaseModel):
    plans: list[PlanOut]
    parent_plan_id: str = Field(
        ...,
        description="plan_id of the PUCT-selected parent the new plans were "
                    "conditioned on. 'seed' on the very first expansion.",
    )
    archive_csv: str = Field(
        ...,
        description="The canonical archive CSV after merge + new-row insertion. "
                    "The orchestrator overwrites its local results.csv with this.",
    )


# ---------- /rollout/reward ----------
class RewardItem(BaseModel):
    plan_id: str
    # Free-form dict from the orchestrator's subagent. The server reads
    # results["reward"] (float) for the RL step and persists the rest verbatim.
    results: dict[str, Any] = Field(default_factory=dict)


class RewardRequest(BaseModel):
    rewards: list[RewardItem] = Field(..., min_length=1)

    @model_validator(mode="after")
    def _no_duplicate_plan_ids(self) -> "RewardRequest":
        ids = [r.plan_id for r in self.rewards]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate plan_id in batch")
        return self


class RewardItemResult(BaseModel):
    plan_id: str
    accepted: bool
    error: str | None = None


class RewardResponse(BaseModel):
    results: list[RewardItemResult]
    group_complete: bool
    batch_complete: bool
    groups_completed: int
    batches_completed: int


# ---------- /status ----------
class StatusResponse(BaseModel):
    ok: bool = True
    iter: int                       # current iteration index (last seen on /rollout/begin)
    batches_trained: int
    outstanding: int                # plan_ids awaiting reward
    last_reward_mean: float | None = None
    done: bool = False              # batches_trained >= num_epochs


# ---------- /best ----------
class BestRow(BaseModel):
    plan_id: str
    iter_idx: int
    reward: float
    results: dict[str, Any]         # full payload as posted by the orchestrator


class BestResponse(BaseModel):
    rows: list[BestRow]
