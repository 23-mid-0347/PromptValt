"""Integration tests requiring real models: distilgpt2 (entropy
pruning) and sentence-transformers (dedup's real embedder path).
Split out from test_compressor.py so sandboxed environments without
network access to huggingface.co can skip just these (see
conftest.real_scorer for the same pattern applied to the scorer)."""

import pytest

from promptvault.compressor import EntropyPruner, semantic_dedup
from tests.conftest import make_turn

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def entropy_pruner() -> EntropyPruner:
    pruner = EntropyPruner()
    try:
        pruner.word_information("warmup")
    except Exception as e:  # noqa: BLE001 - network failure types vary by library
        pytest.skip(f"distilgpt2 unavailable in this environment: {e}")
    return pruner


def test_entropy_pruner_keeps_content_words_over_function_words(entropy_pruner):
    """Core claim: predictable function words ('the', 'is', 'a') carry
    low self-information and should be pruned before high-information
    content words ('refund', 'approved') when the budget is tight."""
    text = "The customer refund request was approved by the manager yesterday"
    result = entropy_pruner.prune(text, keep_fraction=0.5)

    assert "refund" in result or "approved" in result


def test_entropy_pruner_output_shorter_than_input(entropy_pruner):
    text = "This is a fairly long sentence with many predictable connector words in it"
    result = entropy_pruner.prune(text, keep_fraction=0.4)
    assert len(result.split()) < len(text.split())


def test_semantic_dedup_with_real_embedder_catches_paraphrased_duplicate(real_scorer):
    """The real embedder should recognize a paraphrase (not just exact
    text match) as a near-duplicate, unlike the FakeEmbedder used in
    unit tests."""
    t1 = make_turn(1, "The refund was processed and the money is on its way back to you.")
    t2 = make_turn(2, "Your refund has been processed; the money is being returned to you.")
    t3 = make_turn(3, "The train departs from platform nine at noon.")

    kept, _groups = semantic_dedup(
        [t1, t2, t3], embedder=real_scorer._get_embedder(), similarity_threshold=0.85
    )
    kept_ids = {t.turn_id for t in kept}
    assert 3 in kept_ids  # unrelated turn always survives
    assert len(kept) == 2  # t1/t2 collapsed into one