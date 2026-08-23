from agent_state import AgentState
from config import settings
from core import store
from core.nim_client import embed


def retriever_node(state: AgentState) -> AgentState:
    query_embedding = embed(state["question"], input_type="query")

    results = store.get_collection().query(
        query_embeddings=[query_embedding],
        n_results=settings.final_top_k,
        include=["documents", "distances"],
    )

    chunks = results["documents"][0]
    distances = results["distances"][0]

    # ⚠️ Phase 1: distance is squared-L2, not cosine — formula assumes cosine.
    #    Also uses average instead of top-1. Both fixed in Phase 1.
    avg_distance = sum(distances) / len(distances)
    confidence = max(0, 1 - (avg_distance / 2))

    print(f"[Retriever] {len(chunks)} chunks | confidence: {confidence:.2f}")

    return {**state, "retrieved_chunks": chunks, "retrieval_confidence": confidence}