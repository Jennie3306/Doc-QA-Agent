"""Shared state passed between LangGraph nodes."""

from typing import TypedDict


class AgentState(TypedDict):
    question: str
    session_id: str

    retrieved_chunks: list[str]
    chunk_scores: list[float]  # real per-chunk similarity, one per chunk
    retrieval_confidence: float  # top-1 score

    answer: str
    decision: str  # "retrieve" | "meta" | "clarify"
    chat_history: list[dict]
