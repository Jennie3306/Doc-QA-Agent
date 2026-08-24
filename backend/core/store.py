"""ChromaDB access layer."""

import re

import chromadb
from chromadb.errors import NotFoundError

from config import settings

_db = chromadb.PersistentClient(path=settings.chroma_path)

DEFAULT_SESSION = "default"

# ChromaDB collection names: 3-63 chars, alphanumeric/underscore/hyphen,
# must start and end alphanumeric. A UUID passes; arbitrary user input
# does not, so it is sanitised rather than trusted.
_SAFE = re.compile(r"[^a-zA-Z0-9-]")


def collection_name(session_id: str | None = None) -> str:
    sid = _SAFE.sub("", session_id or DEFAULT_SESSION)[:40] or DEFAULT_SESSION
    return f"{settings.collection_name}_{sid}"


def get_collection(session_id: str | None = None):
    """Fresh collection handle on every call.

    Bug 1: caching the handle caused NotFoundError after delete_collection()
    during re-ingest. The client is reusable; the handle is not.

    hnsw:space="cosine" is the Phase 1 fix. Without it ChromaDB defaults to
    SQUARED L2, so every distance the code treated as a cosine distance was
    on a different scale entirely.

    NOTE: this metadata only applies when the collection is CREATED. An
    existing L2 collection keeps using L2 no matter what is passed here,
    which is why the switch required deleting chroma_db and re-ingesting.
    """
    return _db.get_or_create_collection(
        name=collection_name(session_id),
        metadata={"hnsw:space": "cosine"},
    )


def reset_collection(session_id: str | None = None):
    """Drop and recreate one session's collection, dense and sparse.

    Scoped per session so that on a public deployment one user's upload no
    longer wipes the document another user is chatting with.
    """
    # Imported here, not at module level: sparse.py imports store for
    # collection_name(), so a top-level import would be circular.
    from core import sparse

    name = collection_name(session_id)
    try:
        _db.delete_collection(name)
    except NotFoundError:
        pass  # first upload for this session

    # Drop the BM25 index too. Leaving it behind would leave hybrid search
    # returning chunks from the previous document - and unlike a stale
    # vector store, that failure is silent: the chunks look plausible,
    # they are simply from a file the user already replaced.
    sparse.delete(session_id)

    return _db.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def count(session_id: str | None = None) -> int:
    return get_collection(session_id).count()


def distance_to_similarity(distance: float) -> float:
    """Convert a ChromaDB cosine distance into a similarity in [0, 1].

    With hnsw:space="cosine", distance = 1 - cosine_similarity, so the
    distance range is [0, 2] and similarity = 1 - distance.

    The old formula was `1 - distance/2`, which normalised as if distance
    ran over [0, 2] as a *similarity* scale. Combined with the collection
    actually being L2, the reported confidence was meaningless.
    """
    return max(0.0, min(1.0, 1.0 - distance))
