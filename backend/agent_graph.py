from langgraph.graph import END, StateGraph

from agent_state import AgentState
from nodes.clarifier import clarifier_node
from nodes.generator import generator_node
from nodes.meta import meta_node
from nodes.retriever import retriever_node
from nodes.router import router_node


def route_decision(state: AgentState) -> str:
    return {
        "meta": "meta",
        "clarify": "clarifier",
    }.get(state["decision"], "retriever")


def build_agent():
    g = StateGraph(AgentState)
    g.add_node("router", router_node)
    g.add_node("retriever", retriever_node)
    g.add_node("generator", generator_node)
    g.add_node("meta", meta_node)
    g.add_node("clarifier", clarifier_node)

    g.set_entry_point("router")
    g.add_conditional_edges("router", route_decision, {
        "retriever": "retriever",
        "meta": "meta",
        "clarifier": "clarifier",
    })
    g.add_edge("retriever", "generator")
    g.add_edge("generator", END)
    g.add_edge("meta", END)
    g.add_edge("clarifier", END)

    return g.compile()


def initial_state(question: str, chat_history: list[dict]) -> AgentState:
    """Single place where state is constructed — prevents drift between
    the API and the CLI when new fields are added."""
    return {
        "question": question,
        "retrieved_chunks": [],
        "answer": "",
        "decision": "",
        "iterations": 0,
        "chat_history": chat_history,
        "retrieval_confidence": 0.0,
    }