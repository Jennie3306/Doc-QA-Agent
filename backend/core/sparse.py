"""BM25 sparse index, persisted alongside the Chroma collection.

Why a second index at all: dense embeddings compress a whole passage into
one vector, which is excellent for meaning and poor for exact tokens.
Proper nouns, license names and figures - "Technology Innovation Institute",
"Apache 2.0", "3,500 billion" - are precisely what gets diluted. BM25 scores
those directly.

This is the root-cause fix for Bug 2 ("Who created Falcon?" returning
nothing), rather than the earlier workaround of rewording the question.
"""

import pickle
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

from config import settings
from core import store

_INDEX_DIR = Path(settings.chroma_path).parent / "bm25"
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")

# In-process cache: rebuilding BM25 from disk on every query costs ~100ms,
# and the index only changes on ingest.
_cache: dict[str, "SparseIndex"] = {}


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, keeping numbers with separators intact.

    "3,500" and "2.0" must survive as single tokens - splitting them on
    punctuation is exactly how a keyword search loses the figures it is
    supposed to be good at.
    """
    return _TOKEN.findall(text.lower())


class SparseIndex:
    """BM25 over the same chunks Chroma holds, with page numbers kept.

    Pages are stored here too so that a chunk found ONLY by BM25 still
    carries a citation. Without that, hybrid search would quietly produce
    results the Evidence panel could not attribute to a page.
    """

    def __init__(self, texts: list[str], pages: list[int]):
        self.texts = texts
        self.pages = pages
        self.bm25 = BM25Okapi([tokenize(t) for t in texts])

    def search(self, query: str, k: int) -> list[int]:
        """Return corpus indices ordered by BM25 score, best first."""
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])
        # Drop zero-score hits: with no shared token BM25 returns 0, and
        # feeding those into the fusion just adds noise at fixed ranks.
        return [i for i in ranked[:k] if scores[i] > 0]


def _path(session_id: str | None) -> Path:
    return _INDEX_DIR / f"{store.collection_name(session_id)}.pkl"


def build(texts: list[str], pages: list[int], session_id: str | None = None) -> None:
    _INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(_path(session_id), "wb") as f:
        pickle.dump({"texts": texts, "pages": pages}, f)
    _cache[store.collection_name(session_id)] = SparseIndex(texts, pages)


def load(session_id: str | None = None) -> SparseIndex | None:
    name = store.collection_name(session_id)
    if name in _cache:
        return _cache[name]

    p = _path(session_id)
    if not p.exists():
        return None

    with open(p, "rb") as f:
        data = pickle.load(f)

    index = SparseIndex(data["texts"], data["pages"])
    _cache[name] = index
    return index


def delete(session_id: str | None = None) -> None:
    p = _path(session_id)
    if p.exists():
        p.unlink()
    _cache.pop(store.collection_name(session_id), None)


def reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Fuse several ranked id lists into one.

    RRF scores by RANK, not by the underlying score: an item at rank r
    contributes 1/(k+r). That matters because a cosine similarity of 0.55
    and a BM25 score of 12.3 are not on the same scale, so adding or
    averaging them is meaningless - one index would silently dominate
    depending on corpus size and query length.

    k=60 comes from the original RRF paper. It damps the gap between the
    top few ranks, so one index cannot monopolise the result just by being
    confident.
    """
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=lambda i: -scores[i])
