"""Baseline context-selection methods for the PromptVault benchmark.

Every baseline has the same shape so the runner can treat them (and
PromptVault) interchangeably:

    method(history, query, budget) -> list[{"role", "content"}]

`budget` is the EFFECTIVE token ceiling for the returned messages
(already safety-margin adjusted by the runner). `query` is accepted but
ignored by these baselines; it is part of the signature only so they
match PromptVault's call shape.
"""

from __future__ import annotations

from collections.abc import Callable

Message = dict
TokenCounter = Callable[[str], int]


def full_context(history: list[Message], query: str, budget: int, count: TokenCounter) -> list[Message]:
    """No compression. The accuracy ceiling (and cost floor is worst).
    Ignores `budget` entirely; the runner records it once per case."""
    return [dict(m) for m in history]


def last_n(history: list[Message], query: str, budget: int, count: TokenCounter) -> list[Message]:
    """Naive truncation: keep the most recent turns that fit the budget.

    Walks backwards from the newest turn and stops at the first turn
    that does not fit, so the result is always one contiguous recent
    block (never skips a long turn to grab an older short one).
    """
    kept: list[Message] = []
    used = 0
    for msg in reversed(history):
        cost = count(msg["content"])
        if used + cost > budget:
            break
        kept.append(dict(msg))
        used += cost
    kept.reverse()
    return kept


def head_tail(
    history: list[Message],
    query: str,
    budget: int,
    count: TokenCounter,
    head_turns: int = 2,
) -> list[Message]:
    """Sliding-window style baseline (StreamingLLM-flavoured): keep the
    first `head_turns` turns (conversation setup) plus the most recent
    turns that fit what is left of the budget.

    Distinct from `last_n` because it also protects the start of the
    conversation, which is where many real chats put their key facts.
    """
    head: list[Message] = []
    used = 0
    for msg in history[:head_turns]:
        cost = count(msg["content"])
        if used + cost > budget:
            break
        head.append(dict(msg))
        used += cost

    tail: list[Message] = []
    for msg in reversed(history[len(head):]):
        cost = count(msg["content"])
        if used + cost > budget:
            break
        tail.append(dict(msg))
        used += cost
    tail.reverse()
    return head + tail


BASELINES: dict[str, Callable[..., list[Message]]] = {
    "full_context": full_context,
    "last_n": last_n,
    "head_tail": head_tail,
}