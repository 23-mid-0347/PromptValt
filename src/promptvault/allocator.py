"""Token-budget allocator: assigns each turn to a tier based on
relevance score and a hard token budget (plan §2).

Tiers (4-tier design with a recency floor, per project decision):
  - KEEP_RECENT: the most recent N turns, kept verbatim regardless of
    relevance score. Exempt from scoring because very recent turns
    (e.g. "ok continue", "yes do that") are often critical for
    conversational coherence even though they'd score near-zero on
    any relevance metric.
  - KEEP_FULL: kept verbatim, selected by relevance score.
  - COMPRESS: kept but will be summarized/pruned by compressor.py to
    a smaller token footprint.
  - DROP: excluded entirely.

ALGORITHM CHOICE: greedy knapsack over relevance rank, not fixed
percentile buckets (e.g. "top 20% = KEEP_FULL"). Fixed percentiles
break under a real token budget because turns have wildly different
lengths — a fixed "top 20%" could blow the budget on a few long turns,
or leave most of it unused on a batch of short ones. Instead: carve
out the recency floor first, sort everything else by relevance
descending, then greedily spend the remaining budget top-down —
highest-relevance turns get KEEP_FULL first; once a turn no longer
fits at full size, try it at COMPRESS's estimated size; once nothing
fits, DROP the rest. This directly respects the hard token_budget
constraint, which is the actual deliverable per plan §2.

DESIGN NOTE (flagging, not blocking): `compression_ratio` is a single
global estimate (e.g. "compressed turns end up at ~30% of original
size") used only to decide *allocation* here. The real post-compression
size is whatever compressor.py actually produces later, which will
vary per turn depending on content. If the real ratio comes out worse
than estimated on average, the final assembled output could slightly
overshoot token_budget; if better, it'll come in under. Treat
token_budget here as a planning target, not a post-compression
hard guarantee — that guarantee (if you want one) would need to live
in assembler.py as a final trim/re-check pass instead.

EDGE CASE (resolved): what if the recency-floor turns alone exceed
token_budget? Left unhandled, that silently starves every other tier —
every older turn gets DROP regardless of relevance, since the budget
goes negative before relevance-based allocation even starts. That's a
realistic scenario (a small budget with a couple of verbose recent
turns), not just a theoretical one, so it's capped structurally via
`max_recency_fraction`: the recency floor may claim at most that
fraction of token_budget. If the intended recency-floor turns would
exceed the cap, the oldest of that group are trimmed back into the
normal relevance-ranked pool instead of being dropped outright — they
compete for KEEP_FULL/COMPRESS/DROP like any other older turn. This
guarantees the pipeline never fully starves the relevance-driven tiers
just because recent turns happened to be long.
"""

from __future__ import annotations

import math
from typing import Literal

from promptvault.types import Turn

Tier = Literal["KEEP_RECENT", "KEEP_FULL", "COMPRESS", "DROP"]


def allocate_tiers(
    turns: list[Turn],
    scores: dict[int, float],
    token_budget: int,
    recency_floor: int = 3,
    compression_ratio: float = 0.3,
    max_recency_fraction: float = 0.5,
) -> dict[int, Tier]:
    """Assign each turn a tier under a hard token budget.

    Args:
        turns: all candidate turns (any order; sorted internally by
            turn_id to determine chronological recency).
        scores: turn_id -> relevance score, as produced by
            RelevanceScorer.score(). Turns inside the recency floor
            don't need a score (they're exempt), but any turn outside
            the floor missing a score is treated as score 0.0 rather
            than raising, so a caller who forgot to score a turn gets
            a safely low-priority allocation instead of a crash.
        token_budget: ceiling on total *planned* output tokens across
            all kept turns (COMPRESS turns count at their estimated
            compressed size — see module docstring's design note).
        recency_floor: number of most recent turns (by turn_id) that
            are, by default, kept verbatim exempt from relevance
            scoring. Pass 0 to disable and fall back to pure
            relevance-ranked allocation. Actual count kept may be
            lower than this if `max_recency_fraction` caps it (see
            module docstring's "EDGE CASE" note).
        compression_ratio: estimated fraction of a turn's original
            token count it will occupy after compression, in (0, 1].
        max_recency_fraction: maximum fraction of token_budget the
            recency floor is allowed to claim, in (0, 1]. If the
            intended recency-floor turns would exceed this, the
            oldest of them are trimmed back into the relevance-ranked
            pool instead — the recency floor never fully consumes the
            budget on its own.

    Returns:
        {turn_id: tier}. Every input turn appears exactly once.

    Raises:
        ValueError: if recency_floor is negative, or compression_ratio
            / max_recency_fraction are outside (0, 1].
    """
    if recency_floor < 0:
        raise ValueError("recency_floor must be >= 0")
    if not 0 < compression_ratio <= 1:
        raise ValueError("compression_ratio must be in (0, 1]")
    if not 0 < max_recency_fraction <= 1:
        raise ValueError("max_recency_fraction must be in (0, 1]")

    ordered = sorted(turns, key=lambda t: t.turn_id)
    candidate_recent = ordered[-recency_floor:] if recency_floor else []

    # Cap the recency floor's budget share: walk from the most recent
    # backwards, only admitting a turn into KEEP_RECENT if doing so
    # keeps the running recency cost within max_recency_fraction of
    # the total budget. Anything that doesn't make the cut falls back
    # into the normal relevance-ranked pool below.
    recency_cap = token_budget * max_recency_fraction
    recent_ids: set[int] = set()
    recency_cost = 0
    for t in reversed(candidate_recent):
        if recency_cost + t.token_count <= recency_cap:
            recent_ids.add(t.turn_id)
            recency_cost += t.token_count
        else:
            break  # older turns in this group are even less likely to fit

    tiers: dict[int, Tier] = {}
    remaining_budget = token_budget

    for t in ordered:
        if t.turn_id in recent_ids:
            tiers[t.turn_id] = "KEEP_RECENT"
            remaining_budget -= t.token_count

    older = [t for t in ordered if t.turn_id not in recent_ids]
    older.sort(key=lambda t: scores.get(t.turn_id, 0.0), reverse=True)

    for t in older:
        if remaining_budget >= t.token_count:
            tiers[t.turn_id] = "KEEP_FULL"
            remaining_budget -= t.token_count
            continue

        compressed_cost = math.ceil(t.token_count * compression_ratio)
        if remaining_budget >= compressed_cost:
            tiers[t.turn_id] = "COMPRESS"
            remaining_budget -= compressed_cost
            continue

        tiers[t.turn_id] = "DROP"

    return tiers