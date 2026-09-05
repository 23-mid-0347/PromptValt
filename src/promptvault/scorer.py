"""Hybrid relevance scorer: cosine similarity (semantic) + BM25 (lexical).

Per plan §2.1, this is a zero-LLM-cost algorithm — both signals run
locally, offline, in microseconds-to-milliseconds on CPU. No API key,
no network call, no per-use cost. This module can be run as many times
as needed during benchmark sweeps without any marginal cost.
"""

from __future__ import annotations

import re

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from promptvault.types import Turn

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Simple lowercase alphanumeric tokenizer for BM25.

    BM25 doesn't need a sophisticated tokenizer — it just needs
    consistent term boundaries so overlapping-word counts are stable.
    """
    return _TOKEN_RE.findall(text.lower())


def _min_max_normalize(scores: np.ndarray) -> np.ndarray:
    """Scale scores to [0, 1] within the current batch.

    DESIGN NOTE (flagging for visibility, not blocking on it):
    BM25 raw scores are unbounded and their range depends on corpus
    size and query term rarity, while cosine similarity from a
    normalized sentence-embedding model is already roughly in [0, 1].
    Combining them with fixed weights (semantic_weight=0.6,
    lexical_weight=0.4 per §10) only makes sense if both are on a
    comparable scale first — otherwise BM25's larger dynamic range
    could dominate the sum regardless of the configured weights.

    Per-query, per-batch min-max normalization is used here (rather
    than a fixed sigmoid squash) because it's the simplest thing that
    keeps the weights meaningful, and BM25's absolute scale doesn't
    carry cross-query meaning anyway (a "good" BM25 score for one
    query/turn-set isn't comparable to another). The tradeoff: if
    every turn in a batch is roughly equally (ir)relevant, the min-max
    spread becomes noisy (small real differences get stretched to
    the full [0, 1] range). This is an acceptable v1 approximation —
    flag if you want a different normalization scheme.
    """
    lo, hi = scores.min(), scores.max()
    if hi - lo < 1e-12:
        # All turns equally (ir)relevant to this query — no signal to
        # rank on, so contribute a neutral flat score rather than
        # dividing by zero.
        return np.full_like(scores, 0.5)
    return (scores - lo) / (hi - lo)


class RelevanceScorer:
    """Combines BM25 lexical overlap with local embedding similarity.

    relevance = (semantic_weight * embedding_similarity)
              + (lexical_weight * bm25_score)

    Both components are min-max normalized per-query before combining
    (see `_min_max_normalize` for why).
    """

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        semantic_weight: float = 0.6,
        lexical_weight: float = 0.4,
        embedder: object | None = None,
    ):
        # `embedder` is dependency-injected rather than always built
        # here for two reasons: (1) it lets tests substitute a fast,
        # offline stand-in instead of downloading the real ~80MB model
        # every run, and (2) it lets advanced users swap in a
        # different encode(list[str]) -> np.ndarray implementation
        # without subclassing. Real usage just omits it and gets the
        # SentenceTransformer default, lazy-loaded on first use so
        # constructing a RelevanceScorer never requires network by
        # itself.
        self._embedding_model_name = embedding_model
        self._embedder = embedder
        self.semantic_weight = semantic_weight
        self.lexical_weight = lexical_weight

    def _get_embedder(self):
        if self._embedder is None:
            self._embedder = SentenceTransformer(self._embedding_model_name)
        return self._embedder

    def _get_embeddings(self, turns: list[Turn]) -> np.ndarray:
        """Return embeddings for all turns, computing + caching on the
        Turn object for any turn that doesn't have one yet (per §10:
        `embedding` is "computed lazily, cached")."""
        missing_idx = [i for i, t in enumerate(turns) if t.embedding is None]
        if missing_idx:
            texts = [turns[i].content for i in missing_idx]
            computed = self._get_embedder().encode(texts, convert_to_numpy=True)
            for i, vec in zip(missing_idx, computed):
                turns[i].embedding = vec.tolist()
        return np.array([t.embedding for t in turns], dtype=np.float32)

    def score(self, turns: list[Turn], query: str) -> dict[int, float]:
        """Score every turn's relevance to `query`. Returns
        {turn_id: combined_relevance_score}, scores in [0, 1]."""
        if not turns:
            return {}

        # --- Lexical (BM25) ---
        tokenized_corpus = [_tokenize(t.content) for t in turns]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_raw = np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)
        bm25_norm = _min_max_normalize(bm25_raw)

        # --- Semantic (local embeddings) ---
        turn_embeddings = self._get_embeddings(turns)
        query_embedding = self._get_embedder().encode([query], convert_to_numpy=True)[0]
        sem_raw = _cosine_similarity_batch(query_embedding, turn_embeddings)
        sem_norm = _min_max_normalize(sem_raw)

        combined = (
            self.semantic_weight * sem_norm + self.lexical_weight * bm25_norm
        )
        return {t.turn_id: float(s) for t, s in zip(turns, combined)}


def _cosine_similarity_batch(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Cosine similarity of one query vector against a matrix of turn
    embeddings, vectorized (no per-turn Python loop)."""
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    return matrix_norms @ query_norm
