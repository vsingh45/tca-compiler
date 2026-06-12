"""
Reusable LangGraph agent builder that wires WarmStore into the request loop.

Pre-call:  search the user's namespace for relevant memories and inject them
           into the system message.
Post-call: write the new (user, assistant) exchange back into the warm store.

Works fully synthetic out of the box (FakeListChatModel + KeywordImportanceScorer).
Drop in a real chat model and/or EmbeddingsImportanceScorer to turn it into a
production agent.
"""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from .store import WarmStore


def _flatten_message_content(content: Any) -> str:
    """
    Normalize an AIMessage.content value to a plain string.

    Newer LangChain chat models can return `content` as a list of content
    blocks (e.g., `[{"type": "text", "text": "..."}, ...]`) instead of a
    plain string. Storing the list repr in warm memory is useless for
    keyword/embedding scoring, so flatten to text here.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text" and isinstance(block.get("text"), str):
                    parts.append(block["text"])
                elif isinstance(block.get("text"), str):
                    parts.append(block["text"])
        return "\n".join(parts)
    return str(content)


class WarmAgentState(TypedDict, total=False):
    query: str
    namespace: tuple[str, ...]
    recalled: list[dict[str, Any]]
    response: str


_DEFAULT_SYSTEM = (
    "You are a helpful assistant with access to warm memory of prior exchanges "
    "with this user. Use the recalled context if relevant; ignore it if not."
)


def _format_recalled(recalled: list[dict[str, Any]]) -> str:
    if not recalled:
        return "(no prior context)"
    lines = []
    for entry in recalled:
        key = entry.get("key", "?")
        value = entry.get("value", {})
        lines.append(f"- [{key}] {value}")
    return "\n".join(lines)


def build_warm_memory_agent(
    *,
    model: BaseChatModel,
    store: WarmStore,
    recall_limit: int = 5,
    system_prompt: str = _DEFAULT_SYSTEM,
    namespace_default: tuple[str, ...] = ("default",),
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Build a compiled LangGraph agent that uses `store` as warm memory.

    Returns a compiled graph. Invoke it with:
        agent.invoke({"query": "...", "namespace": ("alice",)})
    """

    def memory_lookup(state: WarmAgentState) -> dict[str, Any]:
        namespace = state.get("namespace") or namespace_default
        query = state.get("query", "")
        if not query:
            return {"recalled": []}
        hits = store.search(namespace, query=query, limit=recall_limit)
        return {
            "recalled": [
                {"key": h.key, "value": h.value, "score": h.score} for h in hits
            ],
            "namespace": namespace,
        }

    def respond(state: WarmAgentState) -> dict[str, Any]:
        recalled = state.get("recalled", []) or []
        memory_block = _format_recalled(recalled)
        messages = [
            SystemMessage(content=f"{system_prompt}\n\nRecalled memory:\n{memory_block}"),
            HumanMessage(content=state.get("query", "")),
        ]
        ai_message = model.invoke(messages)
        raw_content = ai_message.content if isinstance(ai_message, AIMessage) else ai_message
        text = _flatten_message_content(raw_content)
        return {"response": text}

    def memory_write(state: WarmAgentState) -> dict[str, Any]:
        namespace = state.get("namespace") or namespace_default
        query = state.get("query", "")
        response = state.get("response", "")
        if not query and not response:
            return {}
        next_key = store.next_key(namespace, prefix="exchange-")
        store.put(
            namespace,
            next_key,
            {"user": query, "assistant": response},
        )
        return {}

    graph = StateGraph(WarmAgentState)
    graph.add_node("memory_lookup", memory_lookup)
    graph.add_node("respond", respond)
    graph.add_node("memory_write", memory_write)

    graph.add_edge(START, "memory_lookup")
    graph.add_edge("memory_lookup", "respond")
    graph.add_edge("respond", "memory_write")
    graph.add_edge("memory_write", END)

    return graph.compile()
