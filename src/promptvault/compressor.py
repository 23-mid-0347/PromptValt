"""Compression techniques for the COMPRESS tier (plan §2.2).

Three distinct techniques, each mapped to a compression-theory analogue:

1. `semantic_dedup` — redundancy elimination. Classical lossless
   compression (LZ77/78) replaces repeated byte sequences with
   references; here, near-duplicate turns (via embedding similarity)
   collapse into one, since a repeated fact adds near-zero new
   information.

2. `EntropyPruner` / `prune_by_information` — entropy-based token
   pruning, LLMLingua-style. A genuine information-theory technique:
   each word's self-information I(word) = -log P(word | context) is
   computed via a small reference LM (distilgpt2). Low-information
   (highly predictable) words are dropped first; high-information
   words are kept. Applied within a turn's text, not at the
   whole-turn level.

3. `extractive_compress` — lossy compression with a rate-distortion
   tradeoff, same principle as JPEG/MP3: deliberately drop
   lower-relevance sentences to fit a token budget, keeping the
   sentences most relevant to the query (reuses RelevanceScorer at
   sentence granularity instead of turn granularity).

Each technique is split into a pure, LM/embedder-free core function
plus a thin wrapper around the real model, so the core logic is fully
unit-testable without downloading anything (see tests/test_compressor.py),
while the real distilgpt2 / sentence-transformers path gets its own
integration tests (tests/test_compressor_integration.py).
"""

from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np

from promptvault.scorer import RelevanceScorer
from promptvault.tokenizer import count_tokens
from promptvault.types import Turn

# ---------------------------------------------------------------------------
# 1. Semantic deduplication
# ---------------------------------------------------------------------------


def semantic_dedup(
    turns: list[Turn],
    embedder: object | None = None,
    similarity_threshold: float = 0.92,
) -> tuple[list[Turn], list[list[int]]]:
    """Merge near-duplicate turns into their first (chronologically
    earliest) occurrence.

    Args:
        turns: candidate turns, any order.
        embedder: optional object with `.encode(list[str]) -> array`,
            for tests / custom models. Defaults to lazily loading the
            same sentence-transformers model the scorer uses.
        similarity_threshold: cosine similarity above which two turns
            are considered near-duplicates, in (0, 1].

    Returns:
        (deduped_turns, merged_groups) — deduped_turns keeps one Turn
        per group (the earliest by turn_id); merged_groups lists the
        turn_ids collapsed into each group that had more than one
        member (for CompressionResult.merged_groups per plan §10).
        Groups of size 1 (no duplicate found) are not included in
        merged_groups.
    """
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if not turns:
        return [], []

    ordered = sorted(turns, key=lambda t: t.turn_id)

    # Reuse embeddings already cached by a prior RelevanceScorer.score()
    # call where available (Turn.embedding); only compute new ones for
    # turns that don't have one yet, mirroring scorer.py's caching.
    missing = [t for t in ordered if t.embedding is None]
    if missing:
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            embedder = SentenceTransformer("all-MiniLM-L6-v2")
        computed = embedder.encode([t.content for t in missing], convert_to_numpy=True)
        for t, vec in zip(missing, computed):
            t.embedding = vec.tolist()

    vectors = np.array([t.embedding for t in ordered], dtype=np.float32)
    norms = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    sim_matrix = norms @ norms.T

    kept: list[Turn] = []
    merged_groups: list[list[int]] = []
    absorbed: set[int] = set()

    for i, t in enumerate(ordered):
        if t.turn_id in absorbed:
            continue
        group = [t.turn_id]
        for j in range(i + 1, len(ordered)):
            other = ordered[j]
            if other.turn_id in absorbed:
                continue
            if sim_matrix[i, j] >= similarity_threshold:
                group.append(other.turn_id)
                absorbed.add(other.turn_id)
        kept.append(t)
        if len(group) > 1:
            merged_groups.append(group)

    return kept, merged_groups


# ---------------------------------------------------------------------------
# 2. Entropy-based token pruning (LLMLingua-style)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _split_words_with_spans(text: str) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in _WORD_RE.finditer(text)]


def prune_by_information(
    word_scores: list[tuple[str, float]],
    keep_fraction: float = 0.5,
) -> str:
    """Pure pruning logic: given precomputed (word, self_information)
    pairs in original order, keep the highest-information
    `keep_fraction` of words and rejoin them in their original order.

    This is deliberately separated from `EntropyPruner` so the
    selection logic is testable without a language model — only
    `EntropyPruner.word_information` needs the real model.

    Args:
        word_scores: [(word, self_information), ...] in original
            left-to-right order.
        keep_fraction: fraction of words to retain, in (0, 1].

    Returns:
        The kept words rejoined with single spaces. Word order is
        preserved (this drops low-information words in place, it
        doesn't reorder by importance) — output may read choppily,
        which is expected for this technique (same tradeoff as
        LLMLingua's own output).
    """
    if not 0 < keep_fraction <= 1:
        raise ValueError("keep_fraction must be in (0, 1]")
    if not word_scores:
        return ""

    n_keep = max(1, round(len(word_scores) * keep_fraction))
    ranked = sorted(range(len(word_scores)), key=lambda i: word_scores[i][1], reverse=True)
    keep_idx = set(ranked[:n_keep])
    kept = [w for i, (w, _score) in enumerate(word_scores) if i in keep_idx]
    return " ".join(kept)


class EntropyPruner:
    """Wraps a small causal LM (default distilgpt2) to compute
    word-level self-information and prune low-information words.

    The model is lazy-loaded on first use, so constructing an
    EntropyPruner never requires network by itself (same pattern as
    RelevanceScorer).
    """

    def __init__(self, model_name: str = "distilgpt2"):
        self._model_name = model_name
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForCausalLM.from_pretrained(self._model_name)
            self._model.eval()

    def word_information(self, text: str) -> list[tuple[str, float]]:
        """Compute self-information I(word) = -log P(word | context)
        for each word in `text`, in original order.

        Subword tokens are mapped back to their owning word via
        character offsets, and a word's information is the sum of its
        constituent tokens' self-information. The first word has no
        preceding context to condition on, so it's assigned a fixed
        neutral score (see inline note) rather than 0.0, so it isn't
        automatically the first thing pruned.
        """
        import torch

        self._ensure_loaded()
        if not text.strip():
            return []

        word_spans = _split_words_with_spans(text)
        enc = self._tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        input_ids = enc["input_ids"]
        offsets = enc["offset_mapping"][0].tolist()

        with torch.no_grad():
            logits = self._model(input_ids).logits[0]
        log_probs = torch.log_softmax(logits, dim=-1)

        word_scores = [0.0] * len(word_spans)
        word_has_info = [False] * len(word_spans)

        # Token i's self-information is predicted from position i-1's
        # logits (standard causal-LM next-token setup). Token 0 has no
        # preceding context, so it's skipped here and handled below.
        for i in range(1, input_ids.shape[1]):
            tok_start, tok_end = offsets[i]
            if tok_start == tok_end:
                continue  # special token with no text span
            token_id = input_ids[0, i].item()
            info = -log_probs[i - 1, token_id].item()
            for wi, (ws, we, _w) in enumerate(word_spans):
                if tok_start < we and tok_end > ws:
                    word_scores[wi] += info
                    word_has_info[wi] = True
                    break

        # First word (and any word whose only token was position 0)
        # has no computed information. Rather than defaulting to 0.0
        # (which would make it the first candidate for pruning purely
        # because it happens to be first), give it the median of the
        # words that *do* have a score, so it competes on a neutral
        # footing.
        scored = [s for s, has in zip(word_scores, word_has_info) if has]
        neutral = float(np.median(scored)) if scored else 0.0
        for wi, has in enumerate(word_has_info):
            if not has:
                word_scores[wi] = neutral

        return [(w, word_scores[i]) for i, (_ws, _we, w) in enumerate(word_spans)]

    def prune(self, text: str, keep_fraction: float = 0.5) -> str:
        """Convenience: compute word information with the real model,
        then prune. See `prune_by_information` for the selection
        logic."""
        return prune_by_information(self.word_information(text), keep_fraction)


# ---------------------------------------------------------------------------
# 3. Extractive summarization
# ---------------------------------------------------------------------------

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on '.', '!', '?' boundaries.

    Deliberately simple (no abbreviation/decimal-number handling) —
    good enough for conversational text, and a wrong split here just
    costs a slightly awkward sentence boundary, not a correctness bug.
    """
    text = text.strip()
    if not text:
        return []
    parts = [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
    return parts or [text]


def extractive_compress(
    text: str,
    query: str,
    target_tokens: int,
    scorer: RelevanceScorer,
    token_counter: Callable[[str], int] = count_tokens,
) -> str:
    """Select the sentences most relevant to `query`, up to
    `target_tokens`, and rejoin them in their original order.

    Reuses RelevanceScorer at sentence granularity instead of turn
    granularity — each sentence is treated as a pseudo-Turn so the
    same hybrid BM25+embedding scoring applies.

    Args:
        text: the turn's full content.
        query: the current query to score relevance against.
        target_tokens: token budget for the compressed output.
        scorer: a RelevanceScorer instance (reused from the pipeline,
            so embeddings already cached on turns aren't wasted).
        token_counter: injectable for tests to avoid a real tiktoken
            dependency; defaults to the real `count_tokens`.

    Returns:
        Selected sentences joined with a space, in original order.
        Empty string if `text` has no sentences or target_tokens <= 0.
    """
    sentences = split_sentences(text)
    if not sentences or target_tokens <= 0:
        return ""

    pseudo_turns = [
        Turn(turn_id=i, role="user", content=s, token_count=token_counter(s))
        for i, s in enumerate(sentences)
    ]
    scores = scorer.score(pseudo_turns, query)

    ranked = sorted(pseudo_turns, key=lambda t: scores[t.turn_id], reverse=True)
    selected_ids: set[int] = set()
    used_tokens = 0
    for t in ranked:
        # Stop at the first sentence that doesn't fit, rather than
        # skipping it and checking lower-ranked (smaller) ones next.
        # Continuing past a miss would let a strictly less-relevant
        # sentence jump ahead of a more-relevant one purely because it
        # happened to be smaller — bin-packing optimality, not
        # relevance-driven selection, which is the whole point of this
        # function. (Found via a real end-to-end test: an unrelated
        # "weather" sentence was sneaking into refund-focused output
        # this way.)
        if used_tokens + t.token_count > target_tokens:
            break
        selected_ids.add(t.turn_id)
        used_tokens += t.token_count

    kept = [t.content for t in pseudo_turns if t.turn_id in selected_ids]
    return " ".join(kept)
# ---------------------------------------------------------------------------
# Composition: combining extractive selection + entropy pruning for a
# single COMPRESS-tier turn (design decided 2026-09: sequential,
# conditional — see module note below)
# ---------------------------------------------------------------------------


def compress_turn(
    text: str,
    query: str,
    target_tokens: int,
    scorer: RelevanceScorer,
    entropy_pruner: EntropyPruner,
    extractive_overshoot: float = 1.5,
    token_counter: Callable[[str], int] = count_tokens,
) -> str:
    """Compress a single COMPRESS-tier turn's text to fit
    `target_tokens`, combining both techniques above in sequence.

    DESIGN DECISION: sequential, not alternative techniques, and the
    second stage is conditional rather than unconditional.

    Extractive selection decides WHAT survives (which whole sentences
    are relevant); entropy pruning decides HOW DENSELY surviving text
    is phrased (which individual words within kept sentences are safe
    to drop as predictable filler). These are complementary axes, not
    competing solutions to the same problem, so they compose rather
    than being chosen between.

    The order can't be reversed: entropy pruning needs coherent
    sentence context to compute meaningful self-information, so
    running it before sentence selection would corrupt both the LM's
    context and `split_sentences`'s regex for the extractive stage.

    Stage 1 runs with `extractive_overshoot` slack above the real
    budget (default 1.5x) so it has enough surviving material to
    leave for stage 2, rather than being forced to cut whole sentences
    that stage 2 could have trimmed more cheaply at the word level.
    Stage 2 only runs if stage 1's output still exceeds
    `target_tokens` — forcing a second pruning pass on text that
    already fits would discard real information for no compression
    benefit, which is the whole reason this is conditional rather
    than always-on.

    Args:
        text: the turn's full original content.
        query: the current query to score relevance against.
        target_tokens: the real, final token budget for this turn.
        scorer: RelevanceScorer instance for stage 1.
        entropy_pruner: EntropyPruner instance for stage 2 (only
            invoked if needed).
        extractive_overshoot: multiplier applied to target_tokens for
            stage 1's intermediate budget, in [1.0, +inf). 1.0 disables
            the overshoot margin (stage 1 targets the real budget
            directly, leaving stage 2 little room to work with).
        token_counter: injectable for tests; defaults to real
            `count_tokens`.

    Returns:
        Compressed text, empty string if target_tokens <= 0 or there's
        no content to select from.
    """
    if target_tokens <= 0:
        return ""

    intermediate_budget = max(1, int(target_tokens * extractive_overshoot))
    selected = extractive_compress(text, query, intermediate_budget, scorer, token_counter)
    if not selected:
        return ""

    current_tokens = token_counter(selected)
    if current_tokens <= target_tokens:
        # Stage 1 alone already fits — skip stage 2 entirely (see
        # "conditional" rationale above).
        return selected

    keep_fraction = min(1.0, max(0.01, target_tokens / current_tokens))
    return entropy_pruner.prune(selected, keep_fraction=keep_fraction)