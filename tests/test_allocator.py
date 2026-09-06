import pytest

from promptvault.allocator import allocate_tiers
from tests.conftest import make_turn


def test_every_turn_gets_exactly_one_tier():
    turns = [make_turn(i, f"turn number {i}") for i in range(1, 6)]
    scores = {t.turn_id: 0.5 for t in turns}
    tiers = allocate_tiers(turns, scores, token_budget=1000)
    assert set(tiers.keys()) == {t.turn_id for t in turns}


def test_recency_floor_kept_regardless_of_low_relevance():
    """The most recent turns must be KEEP_RECENT even if they score
    zero relevance and the budget is tight enough that they'd
    otherwise lose to a higher-relevance older turn."""
    turns = [
        make_turn(1, "highly relevant content about refunds", role="user"),
        make_turn(2, "ok", role="user"),  # recent, low relevance
        make_turn(3, "continue", role="user"),  # most recent, low relevance
    ]
    scores = {1: 0.9, 2: 0.0, 3: 0.0}
    tiers = allocate_tiers(turns, scores, token_budget=1000, recency_floor=2)

    assert tiers[2] == "KEEP_RECENT"
    assert tiers[3] == "KEEP_RECENT"
    assert tiers[1] == "KEEP_FULL"


def test_recency_floor_zero_disables_recency_tier():
    turns = [make_turn(i, f"content {i}") for i in range(1, 4)]
    scores = {t.turn_id: 0.5 for t in turns}
    tiers = allocate_tiers(turns, scores, token_budget=1000, recency_floor=0)
    assert "KEEP_RECENT" not in tiers.values()


def test_higher_relevance_wins_keep_full_under_tight_budget():
    """When the budget can't fit every older turn at full size, the
    higher-relevance ones should be preferred for KEEP_FULL."""
    high = make_turn(1, "word " * 50)  # ~50 tokens-ish, high relevance
    low = make_turn(2, "word " * 50)  # same size, low relevance
    turns = [high, low]
    scores = {1: 0.9, 2: 0.1}

    # Budget fits exactly one turn at full size.
    budget = high.token_count
    tiers = allocate_tiers(turns, scores, token_budget=budget, recency_floor=0)

    assert tiers[1] == "KEEP_FULL"
    assert tiers[2] in ("COMPRESS", "DROP")


def test_turn_falls_back_to_compress_when_full_does_not_fit():
    turn = make_turn(1, "word " * 100)
    scores = {1: 0.9}
    # Budget too small for full text but big enough for the estimated
    # compressed size (compression_ratio=0.3 -> ~30 tokens).
    budget = int(turn.token_count * 0.3) + 5
    tiers = allocate_tiers(
        [turn], scores, token_budget=budget, recency_floor=0, compression_ratio=0.3
    )
    assert tiers[1] == "COMPRESS"


def test_turn_dropped_when_nothing_fits():
    turn = make_turn(1, "word " * 100)
    scores = {1: 0.9}
    tiers = allocate_tiers([turn], scores, token_budget=0, recency_floor=0)
    assert tiers[1] == "DROP"


def test_missing_score_defaults_to_zero_not_crash():
    turns = [make_turn(1, "content"), make_turn(2, "other content")]
    scores = {1: 0.8}  # turn 2 has no score entry
    tiers = allocate_tiers(turns, scores, token_budget=1000, recency_floor=0)
    assert set(tiers.keys()) == {1, 2}


def test_generous_budget_keeps_everything_full():
    turns = [make_turn(i, f"short turn {i}") for i in range(1, 5)]
    scores = {t.turn_id: 0.5 for t in turns}
    tiers = allocate_tiers(turns, scores, token_budget=10_000, recency_floor=0)
    assert all(tier == "KEEP_FULL" for tier in tiers.values())


@pytest.mark.parametrize("bad_recency_floor", [-1, -5])
def test_negative_recency_floor_raises(bad_recency_floor):
    with pytest.raises(ValueError):
        allocate_tiers([], {}, token_budget=100, recency_floor=bad_recency_floor)


@pytest.mark.parametrize("bad_ratio", [0, -0.1, 1.5])
def test_invalid_compression_ratio_raises(bad_ratio):
    with pytest.raises(ValueError):
        allocate_tiers([], {}, token_budget=100, compression_ratio=bad_ratio)


def test_empty_turns_returns_empty_dict():
    assert allocate_tiers([], {}, token_budget=100) == {}

def test_recency_floor_capped_when_it_would_starve_budget():
    """If the intended recency-floor turns alone would exceed
    max_recency_fraction of the budget, the oldest of them should be
    bumped back into the relevance-ranked pool rather than starving
    every other tier."""
    # Three "recent" turns, each ~50 tokens-ish, total ~150+.
    recent_turns = [make_turn(i, "word " * 50) for i in (8, 9, 10)]
    old_high_relevance = make_turn(1, "critical refund policy details")
    turns = [old_high_relevance, *recent_turns]
    scores = {1: 0.95, 8: 0.0, 9: 0.0, 10: 0.0}

    # Budget tight enough that all 3 recent turns together would blow
    # past 50% of it, but the single old high-relevance turn is small.
    budget = 120
    tiers = allocate_tiers(
        turns, scores, token_budget=budget, recency_floor=3, max_recency_fraction=0.5
    )

    # Not all three "recent" turns can be KEEP_RECENT under the cap.
    recent_kept = [tid for tid in (8, 9, 10) if tiers[tid] == "KEEP_RECENT"]
    assert len(recent_kept) < 3

    # The high-relevance older turn should still get a shot at budget
    # (not starved to DROP by the recency floor alone).
    assert tiers[1] in ("KEEP_FULL", "COMPRESS")


def test_recency_floor_fits_normally_when_within_cap():
    """Sanity check: when recency turns comfortably fit under the cap,
    behavior is unchanged from the uncapped case."""
    turns = [make_turn(i, f"short turn {i}") for i in range(1, 4)]
    scores = {t.turn_id: 0.5 for t in turns}
    tiers = allocate_tiers(
        turns, scores, token_budget=10_000, recency_floor=2, max_recency_fraction=0.5
    )
    assert tiers[2] == "KEEP_RECENT"
    assert tiers[3] == "KEEP_RECENT"


@pytest.mark.parametrize("bad_fraction", [0, -0.1, 1.5])
def test_invalid_max_recency_fraction_raises(bad_fraction):
    with pytest.raises(ValueError):
        allocate_tiers([], {}, token_budget=100, max_recency_fraction=bad_fraction)