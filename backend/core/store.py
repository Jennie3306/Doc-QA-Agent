"""ChromaDB access layer."""
import chromadb
from chromadb.errors import NotFoundError

from config import settings

_db = chromadb.PersistentClient(path=settings.chroma_path)


def get_collection():
    """Fresh collection reference on every call.

    See Bug 1: reusing a cached collection object after
    delete_collection() raises NotFoundError.
    """
    return _db.get_or_create_collection(name=settings.collection_name)


def reset_collection():
    """Drop and recreate — used before re-ingesting a document."""
    try:
        _db.delete_collection(settings.collection_name)
    except NotFoundError:
        pass  # nothing to drop on first upload
    return _db.get_or_create_collection(name=settings.collection_name)


def count() -> int:
    return get_collection().count()