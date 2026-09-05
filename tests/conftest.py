import numpy as np
import pytest

from promptvault.scorer import _TOKEN_RE, RelevanceScorer
from promptvault.tokenizer import count_tokens
from promptvault.types import Turn


class FakeEmbedder:
    """Deterministic, offline stand-in for a real embedding model.

    Encodes each text as a hashed bag-of-words vector, so cosine
    similarity tracks word overlap rather than true meaning. This
    exists purely so unit tests can exercise the scorer's
    weighting/normalization/caching *logic* without downloading the
    real ~80MB sentence-transformers model or requiring network.

    It deliberately CANNOT validate the "paraphrase with zero shared
    words still scores high" claim from plan §2.1 — that requires
    real semantic understanding. That specific behavior is verified
    in test_scorer_integration.py against the real model instead.
    """

    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode(self, texts, convert_to_numpy: bool = True):
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in _TOKEN_RE.findall(text.lower()):
                vectors[i, hash(word) % self.dim] += 1.0
        return vectors


@pytest.fixture
def scorer() -> RelevanceScorer:
    """Fast, offline scorer for unit tests — real BM25, fake embedder."""
    return RelevanceScorer(embedder=FakeEmbedder())


@pytest.fixture(scope="session")
def real_scorer() -> RelevanceScorer:
    """Scorer backed by the real sentence-transformers model.

    Session-scoped so the model downloads/loads once per test run.
    Skips (rather than fails) if the model can't be reached — e.g. in
    network-restricted sandboxes — so this doesn't block unrelated
    unit tests from running.
    """
    try:
        s = RelevanceScorer()
        s._get_embedder().encode(["warmup"], convert_to_numpy=True)
        return s
    except Exception as e:  # noqa: BLE001 - network failure types vary by library
        pytest.skip(f"real embedding model unavailable in this environment: {e}")


def make_turn(turn_id: int, content: str, role: str = "user") -> Turn:
    try:
        tokens = count_tokens(content)
    except Exception:  # noqa: BLE001 - network failure types vary by library
        # tiktoken downloads its BPE file on first use; some sandboxed
        # CI/dev environments block that host. token_count isn't used
        # by the scorer itself, so fall back to a naive proxy rather
        # than failing every scorer test over an unrelated dependency.
        tokens = len(content.split())
    return Turn(turn_id=turn_id, role=role, content=content, token_count=tokens)
