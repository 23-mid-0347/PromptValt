import pytest

from promptvault.compressor import (
    compress_turn,
    extractive_compress,
    prune_by_information,
    semantic_dedup,
    split_sentences,
)
from promptvault.scorer import RelevanceScorer
from tests.conftest import FakeEmbedder, make_turn


# ---------------------------------------------------------------------------
# semantic_dedup
# ---------------------------------------------------------------------------


def test_dedup_empty_list():
    assert semantic_dedup([], embedder=FakeEmbedder()) == ([], [])


def test_dedup_merges_near_identical_turns():
    t1 = make_turn(1, "the invoice number is INV-9981")
    t2 = make_turn(2, "the invoice number is INV-9981")  # exact duplicate content
    t3 = make_turn(3, "completely different unrelated content about hiking")

    kept, groups = semantic_dedup([t1, t2, t3], embedder=FakeEmbedder(), similarity_threshold=0.99)

    kept_ids = {t.turn_id for t in kept}
    assert 1 in kept_ids  # earliest of the duplicate pair survives
    assert 2 not in kept_ids  # absorbed into turn 1's group
    assert 3 in kept_ids
    assert [1, 2] in groups


def test_dedup_keeps_dissimilar_turns_separate():
    t1 = make_turn(1, "the weather today is sunny and warm")
    t2 = make_turn(2, "my flight was delayed by three hours")
    kept, groups = semantic_dedup([t1, t2], embedder=FakeEmbedder(), similarity_threshold=0.99)
    assert len(kept) == 2
    assert groups == []


def test_dedup_reuses_cached_embeddings():
    """If turns already have embeddings cached (e.g. from a prior
    scorer.score() call), semantic_dedup should not need to recompute
    them via the embedder."""
    t1 = make_turn(1, "hello world")
    t1.embedding = [1.0, 0.0, 0.0]
    t2 = make_turn(2, "hello world")
    t2.embedding = [1.0, 0.0, 0.0]

    class ExplodingEmbedder:
        def encode(self, texts, convert_to_numpy=True):
            raise AssertionError("should not be called when embeddings are cached")

    kept, groups = semantic_dedup([t1, t2], embedder=ExplodingEmbedder(), similarity_threshold=0.99)
    assert len(kept) == 1
    assert groups == [[1, 2]]


@pytest.mark.parametrize("bad_threshold", [0, -0.1, 1.5])
def test_dedup_invalid_threshold_raises(bad_threshold):
    with pytest.raises(ValueError):
        semantic_dedup([make_turn(1, "x")], embedder=FakeEmbedder(), similarity_threshold=bad_threshold)


# ---------------------------------------------------------------------------
# prune_by_information (pure logic, no LM)
# ---------------------------------------------------------------------------


def test_prune_keeps_highest_information_words():
    word_scores = [("the", 0.1), ("refund", 5.0), ("is", 0.2), ("approved", 4.0)]
    result = prune_by_information(word_scores, keep_fraction=0.5)
    # Should keep the 2 highest-info words: "refund" and "approved",
    # in their ORIGINAL order (not sorted by score).
    assert result == "refund approved"


def test_prune_preserves_original_order():
    word_scores = [("a", 1.0), ("b", 3.0), ("c", 2.0), ("d", 4.0)]
    result = prune_by_information(word_scores, keep_fraction=0.75)
    # Keep top 3 by score (b, c, d) but output in original left-to-right order.
    assert result == "b c d"


def test_prune_empty_input_returns_empty_string():
    assert prune_by_information([], keep_fraction=0.5) == ""


def test_prune_keep_fraction_one_keeps_everything():
    word_scores = [("a", 1.0), ("b", 2.0)]
    assert prune_by_information(word_scores, keep_fraction=1.0) == "a b"


def test_prune_always_keeps_at_least_one_word():
    word_scores = [("a", 1.0), ("b", 2.0), ("c", 3.0)]
    result = prune_by_information(word_scores, keep_fraction=0.01)
    assert len(result.split()) == 1


@pytest.mark.parametrize("bad_fraction", [0, -0.1, 1.5])
def test_prune_invalid_keep_fraction_raises(bad_fraction):
    with pytest.raises(ValueError):
        prune_by_information([("a", 1.0)], keep_fraction=bad_fraction)


# ---------------------------------------------------------------------------
# split_sentences
# ---------------------------------------------------------------------------


def test_split_sentences_basic():
    text = "First sentence. Second sentence! Third one?"
    assert split_sentences(text) == [
        "First sentence.",
        "Second sentence!",
        "Third one?",
    ]


def test_split_sentences_no_punctuation_returns_whole_text():
    assert split_sentences("just one clause with no ending") == [
        "just one clause with no ending"
    ]


def test_split_sentences_empty_string():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


# ---------------------------------------------------------------------------
# extractive_compress
# ---------------------------------------------------------------------------


def _fake_word_count(text: str) -> int:
    """Offline token-count stand-in for unit tests (avoids requiring
    a real tiktoken download)."""
    return len(text.split())


@pytest.fixture
def fake_scorer() -> RelevanceScorer:
    return RelevanceScorer(embedder=FakeEmbedder())


def test_extractive_compress_selects_relevant_sentences_within_budget(fake_scorer):
    text = (
        "The invoice number is INV-9981. "
        "I enjoy hiking on weekends. "
        "Your refund was processed yesterday."
    )
    query = "What is my invoice and refund status?"
    # Budget tight enough that only 2 of the 3 (5-word) sentences fit,
    # forcing the scorer to actually exclude one.
    result = extractive_compress(
        text, query, target_tokens=10, scorer=fake_scorer, token_counter=_fake_word_count
    )
    # Should exclude the unrelated hiking sentence in favor of the two
    # sentences relevant to the invoice/refund query.
    assert "hiking" not in result


def test_extractive_compress_respects_tight_budget(fake_scorer):
    text = "First relevant sentence about refunds. Second relevant sentence about refunds too."
    result = extractive_compress(
        text, "refund", target_tokens=5, scorer=fake_scorer, token_counter=_fake_word_count
    )
    assert _fake_word_count(result) <= 5


def test_extractive_compress_preserves_original_order(fake_scorer):
    text = "Alpha point one. Beta point two. Gamma point three."
    result = extractive_compress(
        text, "alpha beta gamma", target_tokens=100, scorer=fake_scorer, token_counter=_fake_word_count
    )
    # All three fit; original order should be preserved regardless of
    # relevance ranking.
    assert result.index("Alpha") < result.index("Beta") < result.index("Gamma")


def test_extractive_compress_empty_text_returns_empty(fake_scorer):
    assert extractive_compress("", "query", target_tokens=50, scorer=fake_scorer) == ""


def test_extractive_compress_zero_budget_returns_empty(fake_scorer):
    text = "Some sentence here."
    assert (
        extractive_compress(text, "query", target_tokens=0, scorer=fake_scorer) == ""
    )
def test_extractive_compress_does_not_let_smaller_low_relevance_sentence_jump_queue(fake_scorer):
    """Regression test for a real bug found via end-to-end testing:
    a highly-relevant-but-large sentence that doesn't fit the budget
    must not be skipped in favor of a smaller, less-relevant sentence
    later in the ranking. Once a ranked sentence doesn't fit, nothing
    lower-ranked should be considered either, even if it would fit."""
    text = (
        "Alpha is highly relevant to the query and quite long in wording. "
        "Zzz unrelated short filler."
    )
    query = "alpha relevant query wording"
    # Budget fits the short filler sentence alone, but not the long
    # relevant one — if the loop incorrectly skips ahead, "Zzz" would
    # appear in the output despite being irrelevant and lower-ranked.
    result = extractive_compress(
        text, query, target_tokens=4, scorer=fake_scorer, token_counter=_fake_word_count
    )
    assert "Zzz" not in result

# ---------------------------------------------------------------------------
# compress_turn (sequential composition of extractive + entropy pruning)
# ---------------------------------------------------------------------------


class ExplodingEntropyPruner:
    """Test double that fails the test if stage 2 runs when it
    shouldn't (extractive selection alone already fit the budget)."""

    def prune(self, text, keep_fraction=0.5):
        raise AssertionError(
            "entropy pruning should not run when extractive selection alone fits the budget"
        )


class RecordingEntropyPruner:
    """Test double that records calls and does a simple word-count
    trim, standing in for the real distilgpt2-backed EntropyPruner."""

    def __init__(self):
        self.calls: list[tuple[str, float]] = []

    def prune(self, text, keep_fraction=0.5):
        self.calls.append((text, keep_fraction))
        words = text.split()
        n_keep = max(1, round(len(words) * keep_fraction))
        return " ".join(words[:n_keep])


def test_compress_turn_skips_entropy_pruning_when_extractive_alone_fits(fake_scorer):
    text = "Short relevant sentence about refunds."
    result = compress_turn(
        text,
        "refund",
        target_tokens=50,  # generous — extractive alone easily fits
        scorer=fake_scorer,
        entropy_pruner=ExplodingEntropyPruner(),
        token_counter=_fake_word_count,
    )
    assert result  # didn't raise, and produced something


def test_compress_turn_invokes_entropy_pruning_when_still_over_budget(fake_scorer):
    text = (
        "The detailed refund policy explanation about processing times and methods. "
        "Refund amounts are calculated based on the original payment method used."
    )
    pruner = RecordingEntropyPruner()
    result = compress_turn(
        text,
        "refund policy",
        target_tokens=8,  # tight enough that stage 1 alone won't fit
        scorer=fake_scorer,
        entropy_pruner=pruner,
        token_counter=_fake_word_count,
    )
    assert len(pruner.calls) == 1
    _called_text, keep_fraction = pruner.calls[0]
    assert 0 < keep_fraction <= 1
    assert _fake_word_count(result) <= 8


def test_compress_turn_overshoot_gives_stage_one_more_material(fake_scorer):
    """With overshoot > 1.0, stage 1's intermediate budget should be
    larger than the final target — verified indirectly via the
    RecordingEntropyPruner seeing more input text than target_tokens
    would allow on its own."""
    text = "Alpha sentence about refunds. Beta sentence about refunds. Gamma sentence about refunds."
    pruner = RecordingEntropyPruner()
    compress_turn(
        text,
        "refunds",
        target_tokens=5,
        scorer=fake_scorer,
        entropy_pruner=pruner,
        extractive_overshoot=2.0,
        token_counter=_fake_word_count,
    )
    assert len(pruner.calls) == 1
    stage1_output = pruner.calls[0][0]
    # Stage 1's output (before pruning) should exceed the final
    # target — that's the overshoot margin doing its job.
    assert _fake_word_count(stage1_output) > 5


def test_compress_turn_zero_or_negative_budget_returns_empty(fake_scorer):
    text = "Some content."
    result = compress_turn(
        text, "query", target_tokens=0, scorer=fake_scorer, entropy_pruner=ExplodingEntropyPruner()
    )
    assert result == ""


def test_compress_turn_empty_text_returns_empty(fake_scorer):
    result = compress_turn(
        "", "query", target_tokens=50, scorer=fake_scorer, entropy_pruner=ExplodingEntropyPruner()
    )
    assert result == ""

def test_compress_turn_falls_back_to_entropy_pruning_when_extractive_finds_nothing(fake_scorer):
    """Regression test for a real gap found via assembler testing: a
    single atomic 'sentence' (no punctuation) too large to fit the
    overshoot budget left extractive selection with nothing to choose
    from, and compress_turn returned '' instead of falling back to
    entropy-pruning the original text directly."""
    text = "refund refund refund " + "irrelevant filler word " * 20  # no punctuation at all
    pruner = RecordingEntropyPruner()

    result = compress_turn(
        text,
        "refund",
        target_tokens=10,
        scorer=fake_scorer,
        entropy_pruner=pruner,
        token_counter=_fake_word_count,
    )

    assert result != ""
    assert len(pruner.calls) == 1
    pruned_text, _keep_fraction = pruner.calls[0]
    assert pruned_text == text  # fallback pruned the ORIGINAL text, not an empty selection