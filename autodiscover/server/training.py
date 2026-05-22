"""Background training loop for the trainer server.

One coroutine, one batch at a time. The loop is intentionally simple — there is
no overlap with sampling because the rotating sampling_client is reused by the
/rollout/begin handler and we want a clean barrier between iterations.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from autodiscover.backends.protocol import SamplingClient, TrainingClient
from autodiscover.training.data import assemble_training_data
from autodiscover.training.step import (
    compute_advantages,
    incorporate_kl_penalty,
    train_step,
)

from autodiscover.server.state import RolloutStore

logger = logging.getLogger(__name__)


class TrainingLoop:
    """Owns the training_client + base sampling_client.

    The current rollout sampling_client is exposed via `current_sampling_client()`
    so the FastAPI handler can read the latest after each train step.
    """

    def __init__(
        self,
        *,
        store: RolloutStore,
        training_client: TrainingClient,
        base_sampling_client: SamplingClient,
        sampling_client: SamplingClient,
        log_dir: Path,
        learning_rate: float,
        kl_penalty_coef: float,
        adv_estimator: str,
        adv_estimator_beta: float,
        loss_fn: str,
        num_substeps: int,
        num_epochs: int,
        on_metrics: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self._store = store
        self._training_client = training_client
        self._base_sampling_client = base_sampling_client
        self._sampling_client = sampling_client
        self._log_dir = log_dir
        self._lr = learning_rate
        self._kl_coef = kl_penalty_coef
        self._adv_estimator = adv_estimator
        self._adv_estimator_beta = adv_estimator_beta
        self._loss_fn = loss_fn
        self._num_substeps = num_substeps
        self._num_epochs = num_epochs
        self._on_metrics = on_metrics or (lambda _: None)

        self._batches_trained = 0
        self._stop = asyncio.Event()
        self._lock = asyncio.Lock()  # guards _sampling_client read/swap

    async def current_sampling_client(self) -> SamplingClient:
        async with self._lock:
            return self._sampling_client

    @property
    def batches_trained(self) -> int:
        return self._batches_trained

    @property
    def done(self) -> bool:
        return self._batches_trained >= self._num_epochs

    def stop(self) -> None:
        self._stop.set()

    async def run(self) -> None:
        while not self._stop.is_set() and not self.done:
            # Use wait_for to allow stop() to interrupt.
            try:
                batch = await asyncio.wait_for(self._store.next_batch(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            await self._train_one(batch)

    async def _train_one(self, batch: list) -> None:
        i = self._batches_trained
        try:
            advantages = compute_advantages(
                batch, self._adv_estimator, self._adv_estimator_beta,
            )
            data, _meta = assemble_training_data(batch, advantages)

            kl_metrics: dict[str, Any] = {}
            if self._kl_coef > 0:
                kl_metrics = await incorporate_kl_penalty(
                    data, self._base_sampling_client, self._kl_coef,
                )

            await train_step(
                data,
                self._training_client,
                self._lr,
                self._num_substeps,
                self._loss_fn,
            )
            self._maybe_dump_trace(i, advantages, data, kl_metrics)

            new_client = await self._training_client.get_post_training_sampling_client()
            async with self._lock:
                self._sampling_client = new_client

            # Telemetry only — not used in the loss.
            total_trajs = sum(len(g.trajectories_G) for g in batch)
            reward_sum = sum(
                t.transitions[0].reward
                for g in batch
                for t in g.trajectories_G
            )
            reward_mean = reward_sum / max(1, total_trajs)

            self._on_metrics({
                "batch": i,
                "reward_mean": reward_mean,
                **kl_metrics,
            })
        except Exception:
            logger.exception("training step %d failed", i)
            raise
        finally:
            self._batches_trained += 1

    def _maybe_dump_trace(
        self,
        i_batch: int,
        advantages: Any,
        data: Any,
        kl_metrics: dict[str, Any],
    ) -> None:
        """When AUTODISCOVER_DEBUG_TRACE=1, dump (advantages,
        data_input_tokens, kl_metrics).
        """
        if os.environ.get("AUTODISCOVER_DEBUG_TRACE") != "1":
            return
        try:
            adv_payload = (
                [[list(map(float, group_adv)) for group_adv in advantages]]
                if advantages else []
            )
            tokens_payload = [list(map(int, d.input_tokens)) for d in data]
            trace_path = self._log_dir / "_trace.json"
            existing = []
            if trace_path.exists():
                try:
                    existing = json.loads(trace_path.read_text())
                except Exception:
                    existing = []
            existing.append({
                "i_batch": i_batch,
                "advantages": adv_payload,
                "data_input_tokens": tokens_payload,
                "kl_metrics": {
                    k: float(v) for k, v in (kl_metrics or {}).items()
                    if isinstance(v, (int, float))
                },
            })
            trace_path.write_text(json.dumps(existing))
        except Exception:
            logger.exception("debug trace dump failed at batch %d", i_batch)
