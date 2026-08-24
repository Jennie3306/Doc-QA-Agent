"""PDF ingest pipeline, shared by the API and the command line.

Run standalone with:
    python -m core.ingest eval/datasets/test.pdf
"""

from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from core import store
from core.nim_client import embed_batch

PDF_MAGIC = b"%PDF"


def extract_text(pdf_path: str | Path) -> str:
    """Read a PDF into text with page markers."""
    doc = fitz.open(str(pdf_path))
    try:
        return "".join(
            f"\n--- Page {i + 1} ---\n{page.get_text()}" for i, page in enumerate(doc)
        )
    finally:
        doc.close()


def split(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def ingest_text(
    text: str,
    source: str,
    session_id: str | None = None,
) -> int:
    """Chunk, embed and store. Returns the number of chunks indexed.

    Blocking: contains network calls. Callers on the event loop must wrap
    this in asyncio.to_thread().
    """
    chunks = split(text)
    if not chunks:
        raise ValueError("No text extracted - the PDF may be a scan with no text layer")

    embeddings = embed_batch(chunks, input_type="passage")

    col = store.reset_collection(session_id)

    # Chroma writes are batched too: 471 separate add() calls was most of
    # the wall-clock time even after the embedding calls were batched.
    step = settings.embed_batch_size
    for i in range(0, len(chunks), step):
        j = min(i + step, len(chunks))
        col.add(
            ids=[f"chunk_{k}" for k in range(i, j)],
            embeddings=embeddings[i:j],
            documents=chunks[i:j],
            metadatas=[{"chunk_index": k, "source": source} for k in range(i, j)],
        )

    return len(chunks)


def ingest_pdf(pdf_path: str | Path, session_id: str | None = None) -> int:
    path = Path(pdf_path)
    return ingest_text(extract_text(path), source=path.name, session_id=session_id)


def looks_like_pdf(content: bytes) -> bool:
    """Check the magic bytes, not the filename.

    Renaming anything to .pdf used to be enough to get it past validation.
    """
    return content[:4] == PDF_MAGIC


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m core.ingest <path-to.pdf>")
        raise SystemExit(1)

    n = ingest_pdf(sys.argv[1])
    print(f"Indexed {n} chunks into '{store.collection_name()}'")
