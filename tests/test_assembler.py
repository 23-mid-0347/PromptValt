import pytest

from promptvault.assembler import assemble
from promptvault.scorer import RelevanceScorer
from tests.conftest import FakeEmbedder, make_turn


def _fake_word_count(text: str) -> int:
    return len(text.split())


class FakeEntropyPruner:
    """Offline stand-in for EntropyPruner — trims to the first N words
    by keep_fraction, no model required."""

    def prune(self, text: str, keep_fraction: float = 0.5) -> str:
        words = text.split()
        n_keep = max(1, round(len(words) * keep_fraction))
        return " ".join(words[:n_keep])


@pytest.fixture
def fake_scorer() -> RelevanceScorer:
    return RelevanceScorer(embedder=FakeEmbedder())


@pytest.fixture
def fake_pruner() -> FakeEntropyPruner:
    return FakeEntropyPruner()


def test_assemble_empty_turns_returns_empty_result(fake_scorer, fake_pruner):
    result = assemble([], {}, "query", fake_scorer, fake_pruner)
    assert result.messages == []
    assert result.original_tokens == 0
    assert result.compressed_tokens == 0
    assert result.compression_ratio == 0.0
    assert result.dropped_turn_ids == []
    assert result.merged_groups == []


def test_assemble_raises_on_missing_tier(fake_scorer, fake_pruner):
    turns = [make_turn(1, "hello")]
    with pytest.raises(ValueError, match="no tier assignment"):
        assemble(turns, {}, "query", fake_scorer, fake_pruner)


def test_assemble_raises_on_unrecognized_tier(fake_scorer, fake_pruner):
    turns = [make_turn(1, "hello")]
    with pytest.raises(ValueError, match="unrecognized tier"):
        assemble(turns, {1: "NOT_A_REAL_TIER"}, "query", fake_scorer, fake_pruner)


def test_assemble_keep_tiers_included_verbatim(fake_scorer, fake_pruner):
    t1 = make_turn(1, "first message", role="user")
    t2 = make_turn(2, "second message", role="assistant")
    tiers = {1: "KEEP_FULL", 2: "KEEP_RECENT"}

    result = assemble([t1, t2], tiers, "query", fake_scorer, fake_pruner)

    assert result.messages == [
        {"role": "user", "content": "first message"},
        {"role": "assistant", "content": "second message"},
    ]
    assert result.compressed_tokens == t1.token_count + t2.token_count


def test_assemble_drop_tier_excluded_and_recorded(fake_scorer, fake_pruner):
    t1 = make_turn(1, "keep this")
    t2 = make_turn(2, "drop this")
    tiers = {1: "KEEP_FULL", 2: "DROP"}

    result = assemble([t1, t2], tiers, "query", fake_scorer, fake_pruner)

    assert len(result.messages) == 1
    assert result.messages[0]["content"] == "keep this"
    assert result.dropped_turn_ids == [2]


def test_assemble_compress_tier_produces_shorter_content(fake_scorer, fake_pruner):
    long_text = (
        "The refund request was submitted yesterday. "
        + "This sentence is filler and unrelated to the topic at hand. " * 5
        + "The refund was approved by the finance team today."
    )
    t1 = make_turn(1, long_text)
    tiers = {1: "COMPRESS"}

    result = assemble(
        [t1],
        tiers,
        "refund",
        fake_scorer,
        fake_pruner,
        compression_ratio=0.2,
        token_counter=_fake_word_count,
    )

    assert len(result.messages) == 1
    compressed_content = result.messages[0]["content"]
    assert len(compressed_content.split()) < len(long_text.split())


def test_assemble_messages_in_chronological_order_regardless_of_input_order(
    fake_scorer, fake_pruner
):
    t3 = make_turn(3, "third")
    t1 = make_turn(1, "first")
    t2 = make_turn(2, "second")
    tiers = {1: "KEEP_FULL", 2: "KEEP_FULL", 3: "KEEP_FULL"}

    # Deliberately out-of-order input.
    result = assemble([t3, t1, t2], tiers, "query", fake_scorer, fake_pruner)

    assert [m["content"] for m in result.messages] == ["first", "second", "third"]


def test_assemble_merged_groups_passthrough(fake_scorer, fake_pruner):
    t1 = make_turn(1, "hello")
    tiers = {1: "KEEP_FULL"}
    groups = [[1, 5, 7]]

    result = assemble(
        [t1], tiers, "query", fake_scorer, fake_pruner, merged_groups=groups
    )
    assert result.merged_groups == groups


def test_assemble_merged_groups_defaults_to_empty_list(fake_scorer, fake_pruner):
    t1 = make_turn(1, "hello")
    result = assemble([t1], {1: "KEEP_FULL"}, "query", fake_scorer, fake_pruner)
    assert result.merged_groups == []


def test_assemble_compression_ratio_reflects_actual_reduction(fake_scorer, fake_pruner):
    t1 = make_turn(1, "word " * 20)  # 20 words
    tiers = {1: "KEEP_FULL"}  # no compression at all — ratio should be 1.0

    result = assemble(
        [t1], tiers, "query", fake_scorer, fake_pruner, token_counter=_fake_word_count
    )
    assert result.compression_ratio == 1.0


def test_assemble_tier_assignment_recorded_for_every_turn(fake_scorer, fake_pruner):
    t1 = make_turn(1, "a")
    t2 = make_turn(2, "b")
    tiers = {1: "KEEP_FULL", 2: "DROP"}

    result = assemble([t1, t2], tiers, "query", fake_scorer, fake_pruner)
    assert result.tier_assignment == {1: "KEEP_FULL", 2: "DROP"}


def test_assemble_compress_tier_dropped_if_nothing_survives(fake_scorer, fake_pruner):
    """If compress_turn produces nothing usable (e.g. empty content),
    the turn should be recorded as dropped, not emitted as an empty
    message."""
    t1 = make_turn(1, "")  # empty content
    tiers = {1: "COMPRESS"}

    result = assemble(
        [t1], tiers, "query", fake_scorer, fake_pruner, token_counter=_fake_word_count
    )
    assert result.messages == []
    assert result.dropped_turn_ids == [1]