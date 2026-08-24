"""Shared state passed between LangGraph nodes."""

from typing import TypedDict


class AgentState(TypedDict):
    question: str

    # Set by the rewriter node (Stage D) when query rewriting is on;
    # empty otherwise. The retriever falls back to `question`, so the
    # field being empty is a valid state rather than a missing one.
    rewritten_question: str

    session_id: str

    retrieved_chunks: list[str]
    chunk_scores: list[float]  # cosine similarity, one per chunk
    chunk_pages: list[int]  # 1-indexed page each chunk came from

    # Cross-encoder logits, empty when the reranker is off. Kept separate
    # from chunk_scores because the two are on different scales: cosine is
    # bounded [0,1], reranker logits are unbounded (+8 to -17 observed).
    rerank_scores: list[float]

    retrieval_confidence: float  # top-1 cosine, measured before reranking

    answer: str
    decision: str  # "retrieve" | "meta" | "clarify"
    chat_history: list[dict]
