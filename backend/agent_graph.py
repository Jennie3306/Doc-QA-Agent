"""The single definition of the LangGraph agent."""

from langgraph.graph import END, StateGraph

from agent_state import AgentState
from core.store import DEFAULT_SESSION
from nodes.clarifier import clarifier_node
from nodes.generator import generator_node
from nodes.meta import meta_node
from nodes.retriever import retriever_node
from nodes.router import router_node

_ROUTES = {
    "meta": "meta",
    "clarify": "clarifier",
    "retrieve": "retriever",
}


def route_decision(state: AgentState) -> str:
    """Conditional edge. Unknown decisions fall back to retrieval."""
    return _ROUTES.get(state["decision"], "retriever")


def build_agent():
    g = StateGraph(AgentState)

    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("generator", generator_node)
    g.add_node("meta", meta_node)
    g.add_node("clarifier", clarifier_node)

    g.set_entry_point("router")
    g.add_conditional_edges(
        "router",
        route_decision,
        {"retriever": "retriever", "meta": "meta", "clarifier": "clarifier"},
    )
    g.add_edge("retriever", "generator")
    g.add_edge("generator", END)
    g.add_edge("meta", END)
    g.add_edge("clarifier", END)

    return g.compile()


def initial_state(
    question: str,
    chat_history: list[dict],
    session_id: str = DEFAULT_SESSION,
) -> AgentState:
    """The one place agent state is constructed.

    tests/test_graph.py asserts these keys match AgentState exactly, so
    adding a field without updating here fails a test instead of raising
    KeyError mid-conversation.
    """
    return {
        "question": question,
        "session_id": session_id,
        "retrieved_chunks": [],
        "chunk_scores": [],
        "retrieval_confidence": 0.0,
        "answer": "",
        "decision": "",
        "chat_history": chat_history,
    }
