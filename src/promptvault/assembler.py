"""Chronological reassembly (plan §2, step 5).

Takes the allocator's tier decisions and the compressor's techniques,
applies each turn's assigned treatment, and reassembles everything
back into a single chronologically-ordered message list — the actual
`compressed.messages` the plan's API returns (§10).

DESIGN DECISION (§12.6, resolved): when turns were merged by
`semantic_dedup` before reaching this module, only the survivor's
original text appears in the final `messages` list — the merge is
NOT noted inline (no "(3 similar messages consolidated)" marker) and
the other merged turns' text is NOT concatenated in. Reasoning: if
turns were similar enough to pass the dedup similarity threshold,
they're near-paraphrases by construction, so concatenating adds
tokens back for near-zero new information, and an inline note spends
tokens on metadata the LLM doesn't need to answer questions. The fact
that a merge happened is preserved for anyone auditing the
compression via `CompressionResult.merged_groups`, which this module
just passes through unchanged — the final message list itself stays
silent about it.
"""

from __future__ import annotations

import math
from collections.abc import Callable

from promptvault.allocator import Tier
from promptvault.compressor import EntropyPruner, compress_turn
from promptvault.scorer import RelevanceScorer
from promptvault.tokenizer import count_tokens
from promptvault.types import CompressionResult, Turn


def assemble(
    turns: list[Turn],
    tiers: dict[int, Tier],
    query: str,
    scorer: RelevanceScorer,
    entropy_pruner: EntropyPruner,
    merged_groups: list[list[int]] | None = None,
    compression_ratio: float = 0.3,
    token_counter: Callable[[str], int] = count_tokens,
) -> CompressionResult:
    """Apply each turn's tier and reassemble into a CompressionResult.

    Per-tier treatment:
      - KEEP_RECENT / KEEP_FULL: kept verbatim.
      - COMPRESS: shrunk via `compress_turn` to a target size of
        `turn.token_count * compression_ratio` (rounded up) — the
        SAME estimate `allocate_tiers` used when planning the budget,
        so the actual output size stays consistent with what was
        planned rather than drifting from it. If `compress_turn`
        produces nothing usable (e.g. the target rounds down to
        something with no content to extract), the turn is treated as
        effectively dropped rather than emitting an empty message.
      - DROP: excluded, recorded in `dropped_turn_ids`.

    Surviving turns are sorted by `turn_id` before building the final
    `messages` list, guaranteeing chronological order regardless of
    the input order or processing order above.

    Args:
        turns: candidate turns (any order; typically post-dedup output
            from `semantic_dedup` if dedup is used upstream).
        tiers: turn_id -> Tier, from `allocate_tiers()`. Every turn in
            `turns` must have an entry here.
        query: needed to drive extractive selection for COMPRESS-tier
            turns.
        scorer, entropy_pruner: passed through to `compress_turn` for
            COMPRESS-tier turns.
        merged_groups: passthrough from `semantic_dedup`, stored as-is
            on the result for audit purposes (see module docstring's
            design note — the messages list itself doesn't reflect
            this).
        compression_ratio: must match the value passed to
            `allocate_tiers()` for the two stages' token accounting to
            stay consistent.
        token_counter: injectable for tests to avoid a real tiktoken
            dependency; defaults to the real `count_tokens`.

    Returns:
        CompressionResult with `messages` in chronological order.

    Raises:
        ValueError: if any turn is missing a tier assignment, or has
            an unrecognized tier value.
    """
    ordered = sorted(turns, key=lambda t: t.turn_id)
    original_tokens = sum(t.token_count for t in ordered)

    dropped_turn_ids: list[int] = []
    assembled: list[tuple[str, str]] = []  # (role, content), in turn_id order
    compressed_tokens = 0
    tier_assignment: dict[int, Tier] = {}

    for t in ordered:
        tier = tiers.get(t.turn_id)
        if tier is None:
            raise ValueError(f"turn_id {t.turn_id} has no tier assignment")
        tier_assignment[t.turn_id] = tier

        if tier in ("KEEP_RECENT", "KEEP_FULL"):
            assembled.append((t.role, t.content))
            compressed_tokens += t.token_count

        elif tier == "COMPRESS":
            target = max(1, math.ceil(t.token_count * compression_ratio))
            compressed_text = compress_turn(
                t.content,
                query,
                target,
                scorer,
                entropy_pruner,
                token_counter=token_counter,
            )
            if compressed_text:
                assembled.append((t.role, compressed_text))
                compressed_tokens += token_counter(compressed_text)
            else:
                dropped_turn_ids.append(t.turn_id)

        elif tier == "DROP":
            dropped_turn_ids.append(t.turn_id)

        else:
            raise ValueError(f"unrecognized tier {tier!r} for turn_id {t.turn_id}")

    messages = [{"role": role, "content": content} for role, content in assembled]

    # original_tokens == 0 only happens with an empty input conversation
    # (or all-empty-content turns) — define the ratio as 0.0 in that
    # edge case rather than raising a ZeroDivisionError, since there's
    # nothing meaningful to report a "compression ratio" for anyway.
    ratio = (compressed_tokens / original_tokens) if original_tokens else 0.0

    return CompressionResult(
        messages=messages,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=ratio,
        tier_assignment=tier_assignment,
        dropped_turn_ids=dropped_turn_ids,
        merged_groups=merged_groups if merged_groups is not None else [],
    )