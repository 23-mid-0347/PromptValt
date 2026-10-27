"""Offline tests for the LangGraph adapter node."""
import pytest
from promptvault.integrations.langgraph import as_langgraph_node
from promptvault.scorer import RelevanceScorer
from promptvault.vault import PromptVault
from tests.conftest import FakeEmbedder

class FakeEntropyPruner:
    def prune(self, text: str, keep_fraction: float = 0.5) -> str:
        words = text.split()
        return " ".join(words[:max(1, round(len(words) * keep_fraction))])

def word_count(text: str) -> int:
    return len(text.split())

@pytest.fixture
def vault() -> PromptVault:
    return PromptVault(
        token_budget=1000, preserve_recent_n=1,
        _scorer=RelevanceScorer(embedder=FakeEmbedder()),
        _entropy_pruner=FakeEntropyPruner(), _token_counter=word_count,
    )

def test_node_preserves_latest_message(vault):
    node = as_langgraph_node(vault)
    state = {"messages": [
        {"role": "user", "content": "My order number is ABC-123."},
        {"role": "assistant", "content": "I saved order ABC-123."},
        {"role": "user", "content": "Can you check ABC-123?"},
    ]}
    result = node(state)
    assert result["messages"][-1] == state["messages"][-1]
    assert all(isinstance(m["content"], str) for m in result["messages"])

def test_node_accepts_langchain_message_objects(vault):
    class HumanMessage:
        type = "human"
        def __init__(self, content): self.content = content
    class AIMessage:
        type = "ai"
        def __init__(self, content): self.content = content
    result = as_langgraph_node(vault)({"messages": [
        HumanMessage("I need a refund"), AIMessage("I can help"),
        HumanMessage("Refund status?"),
    ]})
    assert result["messages"][-1] == {"role": "user", "content": "Refund status?"}

def test_node_custom_query_key(vault):
    node = as_langgraph_node(vault, query_key="chat_history")
    assert node({"chat_history": [{"role": "user", "content": "hello"}]}) == {"chat_history": []}

def test_node_empty_history(vault):
    assert as_langgraph_node(vault)({"messages": []}) == {"messages": []}

def test_node_missing_key_raises(vault):
    with pytest.raises(KeyError, match="messages"): as_langgraph_node(vault)({"history": []})

def test_node_requires_list(vault):
    with pytest.raises(TypeError, match="must be a list"): as_langgraph_node(vault)({"messages": "no"})

def test_node_rejects_multimodal_content(vault):
    with pytest.raises(ValueError, match="text-only"):
        as_langgraph_node(vault)({"messages": [{"role": "user", "content": [{"type": "text"}]}]})

def test_query_getter_uses_state_field(vault):
    node = as_langgraph_node(vault, query_getter=lambda state: state["question"])
    result = node({"messages": [{"role": "user", "content": "old question"}], "question": "refund status"})
    assert result["messages"] == []

def test_can_compress_latest_message_when_requested(vault):
    node = as_langgraph_node(vault, preserve_latest_message=False)
    assert node({"messages": [{"role": "user", "content": "first"}]})["messages"] == []