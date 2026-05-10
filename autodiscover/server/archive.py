"""PUCT-based archive of past plans + rewards.

Implements the parent-selection rule from
``important_papers/ttt-discover/appendix_a_training_details.md`` §A.2:

    score(s) = Q(s) + c * scale * P(s) * sqrt(1+T) / (1+n(s))

where P(s) is a rank-based prior over the archive, Q(s) is the max child
reward (or R(s) if s has not yet been expanded), n(s) is the visitation
count (with ancestor backprop), T is the total expansions, and
scale = R_max - R_min over the current archive.

The archive is a tree: each entry has a ``parent_id`` (``""`` for seeds).
PUCT counters (``n``, ``m``, ``T``) are reconstructed from the rows on
load, so the persisted CSV can stay minimal (plan_id, iter_idx, parent_id,
reward, plan_text). After load, counters are maintained incrementally as
new groups are minted and rewarded.

Lineage blocking: while a batch is in flight on the trainer side, the
archive blocks the full ancestor+descendant lineage of every parent that
has already been picked for the batch, so the next /rollout/begin in the
same batch picks a parent from a different subtree (paper §A.2 (iv)).
"""
from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass


SEED_PARENT_ID = ""              # parent_id of every seed row
SEED_RESERVED_ID = "seed"        # reserved plan_id for the bootstrap seed
DEFAULT_C = 1.0                  # exploration coefficient (paper hyperparameters)
CHILD_CAP = 2                    # top-2 children per parent (paper §A.2)
GLOBAL_CAP = 1000                # top-1000 archive size (paper §A.2)

CSV_FIELDS = ("plan_id", "iter_idx", "parent_id", "reward", "plan_text")
_CHILD_PLAN_ID_RE = re.compile(r"^(\d+)_(\d+)_(\d+)$")


@dataclass
class ArchiveEntry:
    plan_id: str
    iter_idx: int
    parent_id: str           # "" for seed
    reward: float | None     # None until /rollout/reward populates it
    plan_text: str


def _parse_plan_id(plan_id: str) -> tuple[int, int, int] | None:
    """Return (iter_idx, group_idx, traj_idx) for child plan_ids, else None.

    Seed and any non-conforming ids return None.
    """
    m = _CHILD_PLAN_ID_RE.match(plan_id)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


class Archive:
    def __init__(self, *, c: float = DEFAULT_C):
        self._c = c
        self._rows: dict[str, ArchiveEntry] = {}
        # Derived counters — kept in sync with self._rows.
        self._n: dict[str, int] = {}                 # visitation count (incl. ancestor backprop)
        self._m: dict[str, float] = {}               # max child reward observed
        self._T: int = 0                             # total expansions
        # (iter_idx, group_idx) -> parent_id for every group already minted.
        self._group_parent: dict[tuple[int, int], str] = {}
        # Lineage block set for the in-flight batch (cleared at batch boundary).
        self._blocked: set[str] = set()

    # ---------- Mutation ----------

    def add_row(self, entry: ArchiveEntry) -> None:
        """Insert a new row. Idempotent: existing rows are not overwritten."""
        if entry.plan_id in self._rows:
            return
        self._rows[entry.plan_id] = entry

    def set_reward(self, plan_id: str, reward: float) -> None:
        row = self._rows.get(plan_id)
        if row is None:
            return
        row.reward = reward
        if row.parent_id:
            prev = self._m.get(row.parent_id, float("-inf"))
            if reward > prev:
                self._m[row.parent_id] = reward

    def record_expansion(self, *, parent_id: str, iter_idx: int, group_idx: int) -> None:
        """Register that a group expanded ``parent_id``.

        Increments T and n(a) for parent + all ancestors. Idempotent on the
        same (iter_idx, group_idx) key.
        """
        key = (iter_idx, group_idx)
        if key in self._group_parent:
            return
        self._group_parent[key] = parent_id
        self._T += 1
        for a in [parent_id] + self.ancestors(parent_id):
            self._n[a] = self._n.get(a, 0) + 1

    # ---------- Tree queries ----------

    def has(self, plan_id: str) -> bool:
        return plan_id in self._rows

    def get(self, plan_id: str) -> ArchiveEntry:
        return self._rows[plan_id]

    def ancestors(self, plan_id: str) -> list[str]:
        chain: list[str] = []
        cur = plan_id
        seen: set[str] = set()
        while True:
            row = self._rows.get(cur)
            if row is None or not row.parent_id or row.parent_id in seen:
                break
            chain.append(row.parent_id)
            seen.add(row.parent_id)
            cur = row.parent_id
        return chain

    def descendants(self, plan_id: str) -> set[str]:
        children_of: dict[str, list[str]] = {}
        for r in self._rows.values():
            if r.parent_id:
                children_of.setdefault(r.parent_id, []).append(r.plan_id)
        out: set[str] = set()
        stack: list[str] = list(children_of.get(plan_id, []))
        while stack:
            cur = stack.pop()
            if cur in out:
                continue
            out.add(cur)
            stack.extend(children_of.get(cur, []))
        return out

    def lineage(self, plan_id: str) -> set[str]:
        return {plan_id} | set(self.ancestors(plan_id)) | self.descendants(plan_id)

    # ---------- Lineage blocking (per-batch) ----------

    def block_lineage(self, plan_id: str) -> None:
        self._blocked.update(self.lineage(plan_id))

    def clear_blocked(self) -> None:
        self._blocked.clear()

    # ---------- PUCT selection ----------

    def puct_select(self) -> str:
        """Return the plan_id with the highest PUCT score among unblocked rows.

        Only rewarded rows are eligible (a non-rewarded row has no Q to score
        against and is by construction a freshly-minted child, not a parent
        candidate). If all parents are blocked, lineage filtering is dropped
        as a fallback so the batch can still progress.
        """
        eligible = [
            r for r in self._rows.values()
            if r.reward is not None and r.plan_id not in self._blocked
        ]
        if not eligible:
            eligible = [r for r in self._rows.values() if r.reward is not None]
        if not eligible:
            raise ValueError("archive has no rewarded rows; cannot run PUCT")

        rewards = [r.reward for r in eligible]
        scale = max(rewards) - min(rewards)
        if scale <= 0.0:
            scale = 1.0
        eligible_sorted = sorted(eligible, key=lambda r: r.reward, reverse=True)
        N = len(eligible_sorted)
        # Sum_{i=0..N-1}(N - i) = N*(N+1)/2
        denom = N * (N + 1) / 2.0
        rank_of = {r.plan_id: i for i, r in enumerate(eligible_sorted)}

        best_pid: str | None = None
        best_score = float("-inf")
        for r in eligible:
            rank = rank_of[r.plan_id]
            P = (N - rank) / denom
            n = self._n.get(r.plan_id, 0)
            Q = self._m[r.plan_id] if (n > 0 and r.plan_id in self._m) else r.reward
            score = Q + self._c * scale * P * math.sqrt(1 + self._T) / (1 + n)
            if score > best_score:
                best_score = score
                best_pid = r.plan_id
        assert best_pid is not None
        return best_pid

    # ---------- Archive maintenance ----------

    def enforce_child_cap(self, parent_id: str, *, cap: int = CHILD_CAP) -> set[str]:
        """Keep only the top-``cap`` rewarded children of ``parent_id``.

        Returns the set of removed plan_ids. Children with reward=None are
        always kept (in flight; not eligible for the cap yet).
        """
        rewarded_children = [
            r for r in self._rows.values()
            if r.parent_id == parent_id and r.reward is not None
        ]
        if len(rewarded_children) <= cap:
            return set()
        rewarded_children.sort(key=lambda r: r.reward, reverse=True)
        to_remove = rewarded_children[cap:]
        removed = {r.plan_id for r in to_remove}
        for pid in removed:
            self._rows.pop(pid, None)
            self._n.pop(pid, None)
            self._m.pop(pid, None)
        return removed

    def enforce_global_cap(self, *, cap: int = GLOBAL_CAP) -> set[str]:
        """Cap the archive at ``cap`` rewarded rows, always retaining seeds.

        Seeds are rows with ``parent_id == ""``. Unrewarded rows (in flight)
        are never evicted.
        """
        seeds = {pid for pid, r in self._rows.items() if r.parent_id == SEED_PARENT_ID}
        rewarded = [r for r in self._rows.values()
                    if r.reward is not None and r.plan_id not in seeds]
        if len(rewarded) + len(seeds) <= cap:
            return set()
        rewarded.sort(key=lambda r: r.reward, reverse=True)
        budget = max(cap - len(seeds), 0)
        keep = set(seeds) | {r.plan_id for r in rewarded[:budget]}
        # Always keep unrewarded (in-flight) rows too.
        keep |= {pid for pid, r in self._rows.items() if r.reward is None}
        removed: set[str] = set()
        for pid in list(self._rows.keys()):
            if pid in keep:
                continue
            removed.add(pid)
            self._rows.pop(pid, None)
            self._n.pop(pid, None)
            self._m.pop(pid, None)
        return removed

    # ---------- CSV (de)serialization ----------

    def to_csv(self) -> str:
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        # Stable order: by iter_idx, then plan_id.
        rows = sorted(
            self._rows.values(),
            key=lambda r: (r.iter_idx, r.plan_id),
        )
        for r in rows:
            writer.writerow({
                "plan_id": r.plan_id,
                "iter_idx": r.iter_idx,
                "parent_id": r.parent_id,
                "reward": "" if r.reward is None else repr(r.reward),
                "plan_text": r.plan_text,
            })
        return buf.getvalue()

    def merge_csv(self, csv_text: str) -> None:
        """Merge incoming rows. Server values win on conflict (existing rows kept)."""
        if not csv_text or not csv_text.strip():
            return
        reader = csv.DictReader(io.StringIO(csv_text))
        for row in reader:
            pid = (row.get("plan_id") or "").strip()
            if not pid or pid in self._rows:
                continue
            try:
                iter_idx = int(row.get("iter_idx", "").strip())
            except (TypeError, ValueError):
                continue
            parent_id = (row.get("parent_id") or "").strip()
            raw_reward = (row.get("reward") or "").strip()
            reward: float | None
            if not raw_reward:
                reward = None
            else:
                try:
                    reward = float(raw_reward)
                except ValueError:
                    reward = None
            plan_text = row.get("plan_text") or ""
            self._rows[pid] = ArchiveEntry(
                plan_id=pid,
                iter_idx=iter_idx,
                parent_id=parent_id,
                reward=reward,
                plan_text=plan_text,
            )
        self._reconstruct_counters()

    def _reconstruct_counters(self) -> None:
        """Recompute n, m, T, group_parent from the current rows.

        Called after a merge — incremental updates from record_expansion /
        set_reward only cover events the *server* has seen, not events
        loaded from a pre-existing CSV.
        """
        self._n.clear()
        self._m.clear()
        self._group_parent.clear()
        self._T = 0
        # All G children in a group share the same parent — picking any
        # row for a (iter, group) is enough to recover the parent.
        for r in self._rows.values():
            parsed = _parse_plan_id(r.plan_id)
            if parsed is None:
                continue
            iter_idx, group_idx, _ = parsed
            self._group_parent.setdefault((iter_idx, group_idx), r.parent_id)
        self._T = len(self._group_parent)
        for r in self._rows.values():
            if r.reward is None or not r.parent_id:
                continue
            prev = self._m.get(r.parent_id, float("-inf"))
            if r.reward > prev:
                self._m[r.parent_id] = r.reward
        for (_, _), parent_id in self._group_parent.items():
            for a in [parent_id] + self.ancestors(parent_id):
                self._n[a] = self._n.get(a, 0) + 1

    # ---------- Introspection (for tests + /best) ----------

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def T(self) -> int:
        return self._T

    def n(self, plan_id: str) -> int:
        return self._n.get(plan_id, 0)

    def m(self, plan_id: str) -> float | None:
        return self._m.get(plan_id)

    def rows(self) -> list[ArchiveEntry]:
        return list(self._rows.values())

    def top_k(self, k: int) -> list[ArchiveEntry]:
        rewarded = [r for r in self._rows.values() if r.reward is not None]
        rewarded.sort(key=lambda r: r.reward, reverse=True)
        return rewarded[:k]
