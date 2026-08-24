"""Node 2 - Retriever.

Pipeline, each stage behind a config flag so eval/ablation.py can measure
its individual contribution:

    dense search  -> N candidates
    + BM25        -> fused by Reciprocal Rank Fusion
    + reranker    -> final top-k
"""

from agent_state import AgentState
from config import settings
from core import reranker, sparse, store
from core.nim_client import embed


class _Candidates:
    """One id space for chunks arriving from two different indexes."""

    def __init__(self):
        self.texts: list[str] = []
        self.pages: list[int] = []
        self.dense: list[float] = []
        self._pos: dict[str, int] = {}

    def add(self, text: str, page: int, dense_score: float) -> int:
        if text in self._pos:
            return self._pos[text]
        pos = len(self.texts)
        self.texts.append(text)
        self.pages.append(page)
        self.dense.append(dense_score)
        self._pos[text] = pos
        return pos

    def position_of(self, text: str) -> int | None:
        return self._pos.get(text)


def retriever_node(state: AgentState) -> AgentState:
    question = state.get("rewritten_question") or state["question"]
    session_id = state.get("session_id")

    wants_candidates = settings.use_reranker or settings.use_hybrid_search
    n = settings.retrieve_candidates if wants_candidates else settings.final_top_k

    results = store.get_collection(session_id).query(
        query_embeddings=[embed(question, input_type="query")],
        n_results=n,
        include=["documents", "distances", "metadatas"],
    )

    pool = _Candidates()
    dense_order: list[int] = []
    for doc, dist, meta in zip(
        results["documents"][0],
        results["distances"][0],
        results["metadatas"][0],
        strict=True,
    ):
        dense_order.append(
            pool.add(doc, int(meta.get("page", 0)), store.distance_to_similarity(dist))
        )

    # Confidence is measured on the DENSE result, before any reranking.
    # Two reasons: the threshold was calibrated against cosine similarity,
    # and reranker logits are unbounded (+8 to -17 in testing) so they are
    # not comparable to a [0,1] threshold. Reranking changes the ORDER of
    # what reaches the generator, not how confident we are that the corpus
    # contains an answer at all.
    confidence = pool.dense[dense_order[0]] if dense_order else 0.0

    order = dense_order
    stages = [f"dense({len(dense_order)})"]

    # ── Hybrid: fuse the dense ranking with BM25 ──────────────
    if settings.use_hybrid_search:
        index = sparse.load(session_id)
        if index is None:
            print("[Retriever] no BM25 index - re-run ingest to build one")
        else:
            hits = index.search(question, settings.retrieve_candidates)
            # A chunk BM25 ranks highly but the embedding missed entirely
            # is the whole point of hybrid search, so it gets added to the
            # pool rather than discarded.
            bm25_order = [pool.add(index.texts[i], index.pages[i], 0.0) for i in hits]
            order = sparse.reciprocal_rank_fusion([dense_order, bm25_order])
            new = len(pool.texts) - len(dense_order)
            stages.append(f"bm25({len(hits)}, +{new} new)")

    # ── Rerank the fused candidates down to final_top_k ───────
    rerank_scores: list[float] = []
    if settings.use_reranker and len(order) > 1:
        ranked = reranker.rerank(question, [pool.texts[i] for i in order], settings.final_top_k)
        order = [order[i] for i, _ in ranked]
        rerank_scores = [s for _, s in ranked]
        stages.append(f"rerank:{settings.rerank_backend}")

    order = order[: settings.final_top_k]

    chunks = [pool.texts[i] for i in order]
    scores = [pool.dense[i] for i in order]
    pages = [pool.pages[i] for i in order]

    print(
        f"[Retriever] {' -> '.join(stages)} -> {len(chunks)} chunks | "
        f"top-1 dense: {confidence:.3f} | pages: {pages}"
    )

    return {
        **state,
        "retrieved_chunks": chunks,
        "chunk_scores": scores,
        "chunk_pages": pages,
        "rerank_scores": rerank_scores,
        "retrieval_confidence": confidence,
    }
