"""Cross-encoder reranking, with two interchangeable backends.

Why a cross-encoder helps: the embedding model is a BI-encoder - it encodes
query and passage independently and then compares vectors, so it never sees
the two together. A cross-encoder puts query and passage through one forward
pass, so token-level interaction is available to it. Far more accurate, far
too expensive to run over 468 chunks - hence retrieve 20, rerank to 5.

Why two backends: during Phase 2 the reranker endpoint listed in NVIDIA's
own documentation returned `410 Gone - reached its end of life`. An endpoint
died in front of me, so the demo does not get to depend on a single one.
`local` needs no network at all, and doubles as the baseline for the
fine-tuned reranker in Phase 6.
"""

import httpx

from config import settings

_local_model = None


def _rerank_nim(query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
    """Returns [(original_index, logit), ...] best first."""
    resp = httpx.post(
        settings.rerank_url,
        headers={
            "Authorization": f"Bearer {settings.nvidia_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.rerank_model,
            "query": {"text": query},
            "passages": [{"text": p} for p in passages],
            # END, not NONE: the Mistral-4B reranker has a 503-token budget
            # covering query AND passage together, so a long pair must be
            # clipped rather than rejected.
            "truncate": "END",
        },
        timeout=settings.rerank_timeout,
    )
    resp.raise_for_status()

    rankings = resp.json()["rankings"]
    return [(r["index"], float(r["logit"])) for r in rankings[:top_k]]


def _rerank_local(query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
    global _local_model
    if _local_model is None:
        from sentence_transformers import CrossEncoder

        _local_model = CrossEncoder(settings.rerank_local_model, max_length=512)

    scores = _local_model.predict([(query, p) for p in passages])
    ranked = sorted(enumerate(scores), key=lambda x: -x[1])
    return [(i, float(s)) for i, s in ranked[:top_k]]


def rerank(query: str, passages: list[str], top_k: int) -> list[tuple[int, float]]:
    """Rerank passages. Returns [(index_into_passages, score), ...].

    On any backend failure this returns the identity ordering rather than
    raising: a reranker outage should degrade the answer to dense-only
    quality, not turn it into a 500.
    """
    if not passages:
        return []

    passages = passages[: settings.rerank_max_passages]

    try:
        if settings.rerank_backend == "local":
            return _rerank_local(query, passages, top_k)
        return _rerank_nim(query, passages, top_k)
    except Exception as e:  # noqa: BLE001 - any failure degrades, never breaks
        print(f"[Reranker] {settings.rerank_backend} failed, using dense order: {e}")
        return [(i, 0.0) for i in range(min(top_k, len(passages)))]
