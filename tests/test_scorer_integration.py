"""Integration tests requiring the real sentence-transformers model.

These validate the actual claim from plan §2.1 — that semantic
similarity catches paraphrases with no shared vocabulary — which a
fake/hashed embedder (used in test_scorer.py) can't demonstrate by
construction. They're split into their own file so CI/dev machines
with real network access still get this coverage, while sandboxed
environments without access to huggingface.co can skip just these
without losing the rest of the suite (see conftest.real_scorer).
"""

import pytest

from tests.conftest import make_turn

pytestmark = pytest.mark.integration


def test_semantic_paraphrase_is_caught_with_no_shared_words(real_scorer):
    """A paraphrased query with zero word overlap should still score
    the semantically related turn higher than an unrelated one."""
    turns = [
        make_turn(1, "I would like to get a refund for this purchase."),
        make_turn(2, "The train departs from platform nine at noon."),
    ]
    scores = real_scorer.score(turns, "I need my money back")
    assert scores[1] > scores[2]


def test_hybrid_beats_pure_lexical_on_paraphrase(real_scorer):
    """DyCP's documented failure mode in reverse (plan §2.1/§4): pure
    lexical (BM25) alone scores a word-for-word-different paraphrase
    near the floor. The semantic component should pull the combined
    score up meaningfully above that BM25-only floor."""
    from promptvault.scorer import BM25Okapi, _tokenize

    turns = [make_turn(1, "I would like to get a refund for this purchase.")]
    query = "I need my money back"

    combined = real_scorer.score(turns, query)[1]

    bm25 = BM25Okapi([_tokenize(turns[0].content)])
    bm25_only = bm25.get_scores(_tokenize(query))[0]

    assert combined > 0.0
    assert bm25_only < 1.0  # essentially no lexical overlap
