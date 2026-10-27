"""Offline and optional integration tests for the LangGraph adapter."""
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
    node = as_langgraph_node(vault, overwrite_messages=False)
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

        def __init__(self, content):
            self.content = content

    class AIMessage:
        type = "ai"

        def __init__(self, content):
            self.content = content
    result = as_langgraph_node(vault, overwrite_messages=False)({"messages": [
        HumanMessage("I need a refund"), AIMessage("I can help"),
        HumanMessage("Refund status?"),
    ]})
    assert result["messages"][-1] == {"role": "user", "content": "Refund status?"}

def test_node_custom_query_key(vault):
    node = as_langgraph_node(vault, query_key="chat_history", overwrite_messages=False)
    assert node({"chat_history": [{"role": "user", "content": "hello"}]}) == {
        "chat_history": [{"role": "user", "content": "hello"}]
    }

def test_node_empty_history(vault):
    assert as_langgraph_node(vault)({"messages": []}) == {"messages": []}

def test_node_missing_key_raises(vault):
    with pytest.raises(KeyError, match="messages"):
        as_langgraph_node(vault, overwrite_messages=False)({"history": []})

def test_node_requires_list(vault):
    with pytest.raises(TypeError, match="must be a list"):
        as_langgraph_node(vault, overwrite_messages=False)({"messages": "no"})

def test_node_rejects_multimodal_content(vault):
    with pytest.raises(TypeError, match="text-only"):
        as_langgraph_node(vault, overwrite_messages=False)({
            "messages": [{"role": "user", "content": [{"type": "text"}]}]
        })

def test_query_getter_uses_state_field(vault):
    node = as_langgraph_node(
        vault,
        query_getter=lambda state: state["question"],
        overwrite_messages=False,
    )
    with pytest.raises(ValueError, match="non-empty string"):
        node({
            "messages": [{"role": "user", "content": "latest message"}],
            "question": "",
        })

def test_can_compress_latest_message_when_requested(vault):
    node = as_langgraph_node(vault, preserve_latest_message=False, overwrite_messages=False)
    result = node({"messages": [{"role": "user", "content": "first"}]})
    assert result["messages"][-1]["content"] == "first"

@pytest.mark.integration
def test_node_replaces_messages_using_langgraph_add_messages_reducer(vault):
    pytest.importorskip("langgraph")
    pytest.importorskip("langchain_core")
    from typing import Annotated, TypedDict

    from langchain_core.messages import AIMessage, HumanMessage
    from langgraph.graph import END, START, StateGraph
    from langgraph.graph.message import add_messages

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    graph = StateGraph(State)
    graph.add_node("compress", as_langgraph_node(vault))
    graph.add_edge(START, "compress")
    graph.add_edge("compress", END)
    app = graph.compile()
    result = app.invoke({"messages": [
        HumanMessage(id="old-user", content="A completely unrelated old topic."),
        AIMessage(id="old-ai", content="Old answer that should be replaced by the compressed list."),
        HumanMessage(id="latest", content="Please continue with the earlier topic."),
    ]})
    ids = {message.id for message in result["messages"]}
    assert "old-user" not in ids
    assert "old-ai" not in ids
    assert result["messages"][-1].content == "Please continue with the earlier topic."