"""LangGraph adapter for PromptVault.

This adapter reads a configured message-list field from graph state, uses
the latest message as the query, compresses prior conversation history, and
returns a state update suitable for LangGraph message reducers.

LangGraph is an optional dependency and is imported only when the returned
node executes with overwrite_messages=True.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from promptvault.vault import PromptVault


def _message_to_dict(message: Any) -> dict[str, str]:
    """Normalize LangChain message objects or role/content dictionaries."""
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type")
        content = message.get("content")
    else:
        role = getattr(message, "type", None) or getattr(message, "role", None)
        content = getattr(message, "content", None)
        if role is None:
            name = type(message).__name__.lower()
            if "humanmessage" in name:
                role = "user"
            elif "aimessage" in name:
                role = "assistant"
    if role is None:
        raise ValueError("Each message must include a role or type field.")
    if not isinstance(content, str):
        raise ValueError("PromptVault currently supports text-only message content.")
    role = str(role).lower()
    role = {"human": "user", "ai": "assistant"}.get(role, role)
    if role not in {"user", "assistant"}:
        raise ValueError(f"PromptVault supports user/assistant messages only; got {role!r}.")
    return {"role": role, "content": content}


def as_langgraph_node(
    vault: PromptVault,
    query_key: str = "messages",
    *,
    query_getter: Callable[[Mapping[str, Any]], str] | None = None,
    preserve_latest_message: bool = True,
    overwrite_messages: bool = True,
) -> Callable[[Mapping[str, Any]], dict[str, Any]]:
    """Create a LangGraph node that compresses conversation state.

    Args:
        vault: Configured PromptVault instance.
        query_key: State key containing the conversation message list.
        query_getter: Optional function to extract the query from graph state.
            Defaults to the latest message text.
        preserve_latest_message: Keep the latest message verbatim and outside
            compression, recommended for standard chat graphs.
        overwrite_messages: If True (the default), emit LangGraph's
            RemoveMessage(REMOVE_ALL_MESSAGES) marker before the new messages.
            This is required when query_key uses the add_messages reducer.
            Set False for a plain list field with no reducer.

    Returns:
        A node function that returns a partial graph-state update.

    This first adapter supports text-only user/assistant messages. System/tool
    messages, multimodal content and provider-specific metadata are out of scope.
    """
    if not isinstance(query_key, str) or not query_key:
        raise ValueError("query_key must be a non-empty string.")
    if not isinstance(vault, PromptVault):
        raise TypeError("vault must be an instance of PromptVault.")

    def promptvault_node(state: Mapping[str, Any]) -> dict[str, Any]:
        if query_key not in state:
            raise KeyError(f"LangGraph state has no {query_key!r} key.")
        raw_messages = state[query_key]
        if not isinstance(raw_messages, list):
            raise TypeError(f"state[{query_key!r}] must be a list of messages.")
        if not raw_messages:
            return {query_key: []}

        messages = [_message_to_dict(item) for item in raw_messages]
        query = query_getter(state) if query_getter else messages[-1]["content"]
        if not isinstance(query, str) or not query.strip():
            raise ValueError("The query must be a non-empty string.")

        if preserve_latest_message:
            history, latest = messages[:-1], messages[-1]
        else:
            history, latest = messages, None

        compressed = vault.compress(history, query).messages if history else []
        if latest is not None:
            compressed = [*compressed, latest]

        if overwrite_messages:
            try:
                from langchain_core.messages import RemoveMessage
                from langgraph.graph.message import REMOVE_ALL_MESSAGES
            except ImportError as exc:
                raise ImportError(
                    "The LangGraph adapter needs the optional dependency. Install it with: "
                    "pip install -e '.[langgraph]'"
                ) from exc
            compressed = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *compressed]

        return {query_key: compressed}

    promptvault_node.__name__ = "promptvault_node"
    return promptvault_node