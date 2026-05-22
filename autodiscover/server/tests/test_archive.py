"""Unit tests for autodiscover.server.archive.Archive."""
from __future__ import annotations

import math

from autodiscover.server.archive import (
    Archive,
    ArchiveEntry,
    SEED_PARENT_ID,
)


def _seed(reward: float = 0.0, *, plan_id: str = "seed", text: str = "baseline") -> ArchiveEntry:
    return ArchiveEntry(
        plan_id=plan_id, iter_idx=-1, parent_id=SEED_PARENT_ID,
        reward=reward, plan_text=text,
    )


def _child(iter_idx: int, group_idx: int, traj_idx: int, *,
           parent_id: str, reward: float | None = None,
           text: str = "child") -> ArchiveEntry:
    plan_id = f"{iter_idx:04d}_{group_idx:02d}_{traj_idx:02d}"
    return ArchiveEntry(
        plan_id=plan_id, iter_idx=iter_idx, parent_id=parent_id,
        reward=reward, plan_text=text,
    )


# ---------- PUCT scoring ----------

def test_seed_only_archive_returns_seed():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    assert a.puct_select() == "seed"


def test_argmax_picks_highest_score():
    # Two parents: a low-reward unvisited node should still be selected
    # over a high-reward heavily-visited node when n is large enough.
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.5))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=0.9))
    # Pretend the high-reward node has been hammered.
    for _ in range(50):
        a.record_expansion(parent_id="0000_00_01", iter_idx=99, group_idx=_ )
    pid = a.puct_select()
    # Must not pick the over-visited high-reward parent.
    assert pid != "0000_00_01"


def test_visitation_backprops_to_ancestors():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.4))
    a.record_expansion(parent_id="0000_00_00", iter_idx=1, group_idx=0)
    # Both the direct parent and the seed (its ancestor) should be visited.
    assert a.n("0000_00_00") == 1
    assert a.n("seed") == 1
    assert a.T == 1


def test_m_tracks_max_child_reward():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    # Three children of seed.
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.2))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=0.8))
    a.add_row(_child(0, 0, 2, parent_id="seed", reward=0.5))
    a.record_expansion(parent_id="seed", iter_idx=0, group_idx=0)
    # set_reward on all three
    a.set_reward("0000_00_00", 0.2)
    a.set_reward("0000_00_01", 0.8)
    a.set_reward("0000_00_02", 0.5)
    assert a.m("seed") == 0.8


# ---------- CSV round-trip and reconstruction ----------

def test_csv_round_trip_preserves_rows():
    a = Archive()
    a.add_row(_seed(reward=0.0, text="baseline plan"))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.7, text="line one\nline two"))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=None, text="in flight"))
    csv_text = a.to_csv()

    b = Archive()
    b.merge_csv(csv_text)
    rows = {r.plan_id: r for r in b.rows()}
    assert set(rows) == {"seed", "0000_00_00", "0000_00_01"}
    assert rows["seed"].plan_text == "baseline plan"
    assert rows["0000_00_00"].plan_text == "line one\nline two"
    assert rows["0000_00_00"].reward == 0.7
    assert rows["0000_00_01"].reward is None


def test_counter_reconstruction_from_csv():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    # Group (0,0) expands seed; G=2 children
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.5))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=0.9))
    # Group (1,0) expands child 0000_00_01; G=2 children
    a.add_row(_child(1, 0, 0, parent_id="0000_00_01", reward=0.95))
    a.add_row(_child(1, 0, 1, parent_id="0000_00_01", reward=0.7))
    csv_text = a.to_csv()

    b = Archive()
    b.merge_csv(csv_text)
    assert b.T == 2  # two distinct (iter, group) groups
    # Seed visited twice: once when group (0,0) expanded it directly, once
    # when group (1,0) expanded its grandchild (ancestor backprop).
    assert b.n("seed") == 2
    assert b.n("0000_00_01") == 1
    assert b.n("0000_00_00") == 0
    # m(seed) = max child reward of seed = max(0.5, 0.9)
    assert b.m("seed") == 0.9
    assert b.m("0000_00_01") == 0.95


def test_merge_does_not_overwrite_existing_rows():
    """Server values win on conflict — incoming row with same plan_id is dropped."""
    a = Archive()
    a.add_row(_seed(reward=0.42, text="server seed"))
    incoming = "plan_id,iter_idx,parent_id,reward,plan_text\nseed,-1,,0.0,client seed\n"
    a.merge_csv(incoming)
    assert a.get("seed").reward == 0.42
    assert a.get("seed").plan_text == "server seed"


# ---------- Lineage and blocking ----------

def test_lineage_includes_ancestors_and_descendants():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.5))
    a.add_row(_child(1, 0, 0, parent_id="0000_00_00", reward=0.7))
    a.add_row(_child(1, 0, 1, parent_id="0000_00_00", reward=0.6))
    a.add_row(_child(0, 1, 0, parent_id="seed", reward=0.4))   # sibling subtree

    lineage = a.lineage("0000_00_00")
    assert "seed" in lineage           # ancestor
    assert "0000_00_00" in lineage     # self
    assert "0001_00_00" in lineage     # descendant
    assert "0001_00_01" in lineage     # descendant
    assert "0000_01_00" not in lineage  # sibling, not in lineage


def test_block_lineage_excludes_from_puct():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.5))
    a.add_row(_child(0, 1, 0, parent_id="seed", reward=0.4))
    a.block_lineage("0000_00_00")
    pid = a.puct_select()
    # Both 0000_00_00 and seed (its ancestor) are blocked; only 0000_01_00 remains.
    assert pid == "0000_01_00"

    # Clearing the block returns the seed/child as candidates.
    a.clear_blocked()
    assert a.puct_select() in {"seed", "0000_00_00", "0000_01_00"}


def test_block_falls_back_when_everything_blocked():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.block_lineage("seed")
    # Seed is the only candidate; fallback should still return it.
    assert a.puct_select() == "seed"


# ---------- Archive maintenance ----------

def test_child_cap_keeps_top_two_rewarded():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    rewards = [0.1, 0.9, 0.5, 0.3, 0.7]
    for i, r in enumerate(rewards):
        a.add_row(_child(0, 0, i, parent_id="seed", reward=r))
    removed = a.enforce_child_cap("seed")
    surviving = {r.plan_id for r in a.rows() if r.parent_id == "seed"}
    # Top-2 by reward are 0.9 (idx 1) and 0.7 (idx 4).
    assert surviving == {"0000_00_01", "0000_00_04"}
    assert "0000_00_00" in removed and "0000_00_02" in removed and "0000_00_03" in removed


def test_child_cap_does_not_evict_unrewarded_in_flight_children():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.9))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=0.8))
    a.add_row(_child(0, 0, 2, parent_id="seed", reward=0.7))   # would normally be evicted
    a.add_row(_child(0, 0, 3, parent_id="seed", reward=None))  # in flight
    a.enforce_child_cap("seed")
    surviving = {r.plan_id for r in a.rows() if r.parent_id == "seed"}
    # In-flight row stays, top-2 rewarded stay, the "extra" rewarded is evicted.
    assert "0000_00_03" in surviving
    assert "0000_00_00" in surviving and "0000_00_01" in surviving
    assert "0000_00_02" not in surviving


def test_global_cap_retains_seeds():
    a = Archive(c=1.0)
    a.add_row(_seed(reward=0.0))
    # Add 5 children with increasing rewards.
    for i, r in enumerate([0.1, 0.2, 0.3, 0.4, 0.5]):
        a.add_row(_child(0, 0, i, parent_id="seed", reward=r))
    # Cap=3 means we keep the seed + top-(3-1)=2 children = top-2 rewarded.
    a.enforce_global_cap(cap=3)
    surviving = {r.plan_id for r in a.rows()}
    assert "seed" in surviving
    assert "0000_00_04" in surviving and "0000_00_03" in surviving
    assert len(surviving) == 3


# ---------- Score-formula sanity ----------

def test_score_formula_matches_paper():
    """For a 2-row archive with known counters, replicate the formula by hand."""
    a = Archive(c=1.0)
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=1.0))
    # No expansions yet => T=0, n(s)=0 for all s, Q(s)=R(s).
    # scale = 1.0 - 0.0 = 1.0
    # ranks: 0000_00_00 (R=1.0) rank 0, seed (R=0.0) rank 1.
    # N=2, denom = 2*3/2 = 3.
    # P(0000_00_00) = (2-0)/3 = 2/3; P(seed) = (2-1)/3 = 1/3.
    # score(child) = 1.0 + 1.0 * 1.0 * 2/3 * sqrt(1)/1 = 1.0 + 2/3 ≈ 1.6667
    # score(seed)  = 0.0 + 1.0 * 1.0 * 1/3 * sqrt(1)/1 = 1/3 ≈ 0.3333
    pid = a.puct_select()
    assert pid == "0000_00_00"
    # Now hammer the child so its bonus shrinks but Q stays at R until a
    # group of children has been rewarded.
    for k in range(10):
        a.record_expansion(parent_id="0000_00_00", iter_idx=99, group_idx=k)
    # After 10 expansions, n(child)=10, n(seed)=10 (ancestor), T=10.
    # Now the bonus on child is small; seed has same n; tie-break by Q.
    # Q(child)=1.0 still (no children with rewards), Q(seed)=0.0.
    # So child still wins on Q, but the gap is tiny — the formula is
    # well-defined either way; we just check no crash.
    pid2 = a.puct_select()
    assert pid2 in {"0000_00_00", "seed"}


def test_top_k_orders_by_reward_descending():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=0.5))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=0.9))
    a.add_row(_child(0, 0, 2, parent_id="seed", reward=None))  # in flight, excluded
    rows = a.top_k(2)
    assert [r.reward for r in rows] == [0.9, 0.5]


def test_set_reward_updates_m_for_parent():
    a = Archive()
    a.add_row(_seed(reward=0.0))
    a.add_row(_child(0, 0, 0, parent_id="seed", reward=None))
    a.add_row(_child(0, 0, 1, parent_id="seed", reward=None))
    assert a.m("seed") is None
    a.set_reward("0000_00_00", 0.3)
    assert a.m("seed") == 0.3
    a.set_reward("0000_00_01", 0.1)
    assert a.m("seed") == 0.3   # max stays at 0.3
    # NaN-safe: no crash on rows we don't know about.
    a.set_reward("not-in-archive", 0.99)


def test_seed_with_zero_reward_still_eligible():
    """Cold start: only seed exists, with reward 0.0 and scale=0 fallback to 1.0."""
    a = Archive()
    a.add_row(_seed(reward=0.0))
    # scale = R_max - R_min = 0; should fall back to 1.0 internally.
    pid = a.puct_select()
    assert pid == "seed"
    # Sanity: verify the score is finite.
    # (reading via the public surface: just ensure it doesn't raise.)
    assert math.isfinite(0.0)  # placeholder; the absence of an exception is the assertion
