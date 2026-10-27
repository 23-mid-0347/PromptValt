"""Metrics for the PromptVault benchmark.

All functions are pure (no model calls) so they can be unit-tested and
reused by every method, including PromptVault and the baselines.
"""

from __future__ import annotations

import re
from collections.abc import Callable

Message = dict
TokenCounter = Callable[[str], int]

_WS = re.compile(r"\s+")
PART_SEPARATOR = " | "  # multi_evidence gold answers: "Priya | April 9"


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace so trivial formatting
    differences never count as information loss."""
    return _WS.sub(" ", text.lower()).strip()


def gold_parts(gold_answer: str) -> list[str]:
    return [p.strip() for p in gold_answer.split(PART_SEPARATOR) if p.strip()]


def messages_text(messages: list[Message]) -> str:
    return normalize("\n".join(m["content"] for m in messages))


def count_messages_tokens(messages: list[Message], count: TokenCounter) -> int:
    return sum(count(m["content"]) for m in messages)


def token_reduction_pct(original_tokens: int, compressed_tokens: int) -> float:
    """100 * (1 - compressed/original). 0.0 for an empty original."""
    if original_tokens <= 0:
        return 0.0
    return 100.0 * (1.0 - compressed_tokens / original_tokens)


def compression_ratio(original_tokens: int, compressed_tokens: int) -> float:
    """original / compressed (>1 means smaller). Returns inf when the
    output is empty but the input was not, and 0.0 when both are empty."""
    if compressed_tokens <= 0:
        return float("inf") if original_tokens > 0 else 0.0
    return original_tokens / compressed_tokens


def evidence_recall(messages: list[Message], gold_answer: str) -> float:
    """Fraction of gold-answer parts still present (verbatim, after
    normalization) anywhere in the output messages.

    This checks the TEXT that actually reaches the model, not tier
    labels: a turn marked COMPRESS may have lost the fact, and a turn
    merged away by dedup is fine if its survivor still contains it.
    """
    parts = gold_parts(gold_answer)
    if not parts:
        return 0.0
    haystack = messages_text(messages)
    found = sum(1 for p in parts if normalize(p) in haystack)
    return found / len(parts)


def answer_present(messages: list[Message], gold_answer: str) -> float:
    """Deterministic answer-quality proxy: 1.0 only if EVERY gold part
    survives, else 0.0. Stricter than evidence_recall for
    multi_evidence cases. This is a retention proxy, not a model's
    answer accuracy; real QA accuracy needs a target LLM."""
    return 1.0 if evidence_recall(messages, gold_answer) == 1.0 else 0.0