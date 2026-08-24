"""Node 2 - Retriever. Vector search over ChromaDB."""

from agent_state import AgentState
from config import settings
from core import store
from core.nim_client import embed


def retriever_node(state: AgentState) -> AgentState:
    query_embedding = embed(state["question"], input_type="query")

    results = store.get_collection(state.get("session_id")).query(
        query_embeddings=[query_embedding],
        n_results=settings.final_top_k,
        include=["documents", "distances"],
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]
    scores = [store.distance_to_similarity(d) for d in distances]

    # Top-1, not the average of top-5.
    #
    # On a valid question typically only 1-2 chunks actually contain the
    # answer; the rest are near-miss noise. Averaging drags the signal down
    # toward the level an out-of-scope question produces, so the in-scope
    # and out-of-scope distributions overlap and no threshold separates
    # them. Top-1 keeps the strongest signal intact.
    confidence = scores[0] if scores else 0.0

    print(
        f"[Retriever] {len(chunks)} chunks | top-1: {confidence:.3f} | "
        f"scores: {[f'{s:.3f}' for s in scores]}"
    )

    return {
        **state,
        "retrieved_chunks": chunks,
        "chunk_scores": scores,
        "retrieval_confidence": confidence,
    }
