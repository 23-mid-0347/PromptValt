import pytest

from promptvault.scorer import RelevanceScorer
from promptvault.vault import PromptVault, _history_to_turns
from tests.conftest import FakeEmbedder


class FakeEntropyPruner:
    """Offline stand-in for EntropyPruner (same double used in
    test_assembler.py)."""

    def prune(self, text: str, keep_fraction: float = 0.5) -> str:
        words = text.split()
        n_keep = max(1, round(len(words) * keep_fraction))
        return " ".join(words[:n_keep])


def _fake_word_count(text: str) -> int:
    return len(text.split())


@pytest.fixture
def vault() -> PromptVault:
    """A PromptVault wired with fake, offline models — for testing
    orchestration logic, not model quality (that's covered by each
    module's own integration tests)."""
    return PromptVault(
        token_budget=1000,
        preserve_recent_n=1,
        _scorer=RelevanceScorer(embedder=FakeEmbedder()),
        _entropy_pruner=FakeEntropyPruner(),
        _token_counter=_fake_word_count,
    )


# ---------------------------------------------------------------------------
# _history_to_turns
# ---------------------------------------------------------------------------


def test_history_to_turns_assigns_turn_id_by_position():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
    ]
    turns = _history_to_turns(history, token_counter=_fake_word_count)
    assert [t.turn_id for t in turns] == [0, 1]
    assert [t.role for t in turns] == ["user", "assistant"]
    assert [t.content for t in turns] == ["first", "second"]


def test_history_to_turns_raises_on_missing_key():
    history = [{"role": "user"}]  # missing "content"
    with pytest.raises(ValueError, match="missing required key"):
        _history_to_turns(history, token_counter=_fake_word_count)


def test_history_to_turns_empty_history():
    assert _history_to_turns([], token_counter=_fake_word_count) == []


# ---------------------------------------------------------------------------
# PromptVault.__init__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_margin", [0, -0.1, 1.5])
def test_invalid_budget_safety_margin_raises(bad_margin):
    with pytest.raises(ValueError, match="budget_safety_margin"):
        PromptVault(budget_safety_margin=bad_margin)


def test_default_construction_does_not_require_network():
    """Constructing a PromptVault with all defaults should not itself
    trigger any model download — everything is lazy-loaded (matches
    RelevanceScorer/EntropyPruner's own lazy-loading contracts)."""
    PromptVault()  # should not raise / hang / attempt network access


# ---------------------------------------------------------------------------
# PromptVault.score
# ---------------------------------------------------------------------------


def test_score_returns_one_score_per_turn(vault):
    history = [
        {"role": "user", "content": "What's my refund status?"},
        {"role": "assistant", "content": "I enjoy hiking."},
    ]
    scores = vault.score(history, "refund")
    assert set(scores.keys()) == {0, 1}


def test_score_empty_history_returns_empty_dict(vault):
    assert vault.score([], "query") == {}


# ---------------------------------------------------------------------------
# PromptVault.compress
# ---------------------------------------------------------------------------


def test_compress_returns_messages_in_original_order(vault):
    history = [
        {"role": "user", "content": "first message about refunds"},
        {"role": "assistant", "content": "second message about refunds"},
        {"role": "user", "content": "third message, ok continue"},
    ]
    result = vault.compress(history, "refund status")
    original_order = [msg["content"] for msg in history]
    kept_contents = [m["content"] for m in result.messages]
    # Whatever survived should appear in the same relative order as
    # the original history, regardless of relevance ranking.
    assert kept_contents == [c for c in original_order if c in kept_contents]


def test_compress_respects_recency_floor(vault):
    """The most recent turn should survive as KEEP_RECENT even with
    zero lexical/semantic relevance to the query, given preserve_recent_n=1."""
    history = [
        {"role": "user", "content": "Tell me about refund policy details."},
        {"role": "user", "content": "ok"},  # most recent, irrelevant to query
    ]
    result = vault.compress(history, "refund policy")
    assert result.tier_assignment[1] == "KEEP_RECENT"


def test_compress_reports_dropped_turns(vault):
    small_vault = PromptVault(
        token_budget=1,  # effectively nothing fits
        preserve_recent_n=0,
        budget_safety_margin=1.0,
        _scorer=RelevanceScorer(embedder=FakeEmbedder()),
        _entropy_pruner=FakeEntropyPruner(),
        _token_counter=_fake_word_count,
    )
    history = [{"role": "user", "content": "some reasonably long message here"}]
    result = small_vault.compress(history, "query")
    assert 0 in result.dropped_turn_ids or result.messages == []


def test_compress_returns_compression_result_shape(vault):
    history = [{"role": "user", "content": "hello there"}]
    result = vault.compress(history, "query")
    assert hasattr(result, "messages")
    assert hasattr(result, "original_tokens")
    assert hasattr(result, "compressed_tokens")
    assert hasattr(result, "compression_ratio")
    assert hasattr(result, "tier_assignment")
    assert hasattr(result, "dropped_turn_ids")
    assert hasattr(result, "merged_groups")


def test_compress_empty_history_returns_empty_result(vault):
    result = vault.compress([], "query")
    assert result.messages == []
    assert result.original_tokens == 0


def test_compress_merges_near_duplicate_turns(vault):
    history = [
        {"role": "user", "content": "the invoice number is INV-9981"},
        {"role": "user", "content": "the invoice number is INV-9981"},  # exact duplicate
    ]
    result = vault.compress(history, "invoice")
    assert len(result.merged_groups) >= 1


def test_budget_safety_margin_reduces_effective_budget():
    """A vault with a smaller safety margin should end up dropping
    more content under an identical, tight budget than one with a
    larger margin (fewer tokens are actually available to allocate)."""
    history = [
        {"role": "user", "content": "word " * 30},
        {"role": "user", "content": "word " * 30},
        {"role": "user", "content": "word " * 30},
    ]

    generous = PromptVault(
        token_budget=60,
        budget_safety_margin=1.0,
        preserve_recent_n=0,
        _scorer=RelevanceScorer(embedder=FakeEmbedder()),
        _entropy_pruner=FakeEntropyPruner(),
        _token_counter=_fake_word_count,
    )
    strict = PromptVault(
        token_budget=60,
        budget_safety_margin=0.3,
        preserve_recent_n=0,
        _scorer=RelevanceScorer(embedder=FakeEmbedder()),
        _entropy_pruner=FakeEntropyPruner(),
        _token_counter=_fake_word_count,
    )

    generous_result = generous.compress(history, "query")
    strict_result = strict.compress(history, "query")

    assert strict_result.compressed_tokens <= generous_result.compressed_tokens


# ---------------------------------------------------------------------------
# PromptVault.as_langgraph_node
# ---------------------------------------------------------------------------


def test_as_langgraph_node_raises_not_implemented(vault):
    with pytest.raises(NotImplementedError):
        vault.as_langgraph_node()