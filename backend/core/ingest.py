"""PDF ingest pipeline, shared by the API and the command line.

Run standalone with:
    python -m core.ingest eval/datasets/test.pdf
"""

from pathlib import Path

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from core import sparse, store
from core.nim_client import embed_batch

PDF_MAGIC = b"%PDF"

# A chunk shorter than this is almost always a page header, a footer, a
# stray figure caption or a fragment of a reference list.
#
# Measured impact on the Falcon paper: 1 chunk out of 469. Nearly nothing,
# because RecursiveCharacterTextSplitter already breaks on paragraph
# boundaries and rarely emits fragments. Kept for PDFs with worse
# structure, but it is not doing real work here and should not be claimed
# as an improvement.
MIN_CHUNK_CHARS = 80


def extract_pages(pdf_path: str | Path) -> list[tuple[int, str]]:
    """Return [(page_number, text), ...], 1-indexed."""
    doc = fitz.open(str(pdf_path))
    try:
        return [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()


def extract_text(pdf_path: str | Path) -> str:
    """Flat text with page markers. Kept for callers that want raw text."""
    return "".join(f"\n--- Page {n} ---\n{text}" for n, text in extract_pages(pdf_path))


def split_with_pages(pages: list[tuple[int, str]]) -> list[dict]:
    """Chunk each page separately so page numbers survive.

    Phase 1 concatenated every page into one string before splitting, which
    threw the page numbers away: the Evidence panel could only say
    "Chunk 3", which tells the user nothing they can check against the
    actual document.

    Measured effect on chunk count: 471 -> 469, i.e. it went DOWN by 2.
    The prediction was +15-25% from partial chunks at each page tail. That
    did not happen, for two reasons: the flat version inserted 57
    "--- Page N ---" markers (~2 chunks' worth of text), and
    RecursiveCharacterTextSplitter already splits on "\\n\\n" first, so it
    was mostly respecting page boundaries anyway.

    So this change buys citation traceability, not retrieval quality.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )

    out: list[dict] = []
    for page_no, text in pages:
        if not text.strip():
            continue
        for piece in splitter.split_text(text):
            cleaned = piece.strip()
            if len(cleaned) < MIN_CHUNK_CHARS:
                continue
            out.append({"text": cleaned, "page": page_no})

    return out


def split(text: str) -> list[str]:
    """Plain splitter without page tracking. Used by chunk_optimizer."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)


def ingest_chunks(
    chunks: list[dict],
    source: str,
    session_id: str | None = None,
) -> int:
    """Embed and store pre-split chunks. Returns the number indexed.

    Blocking: contains network calls. Callers on the event loop must wrap
    this in asyncio.to_thread().
    """
    if not chunks:
        raise ValueError("No text extracted - the PDF may be a scan with no text layer")

    texts = [c["text"] for c in chunks]
    pages = [c["page"] for c in chunks]

    embeddings = embed_batch(texts, input_type="passage")

    col = store.reset_collection(session_id)

    step = settings.embed_batch_size
    for i in range(0, len(chunks), step):
        j = min(i + step, len(chunks))
        col.add(
            ids=[f"chunk_{k}" for k in range(i, j)],
            embeddings=embeddings[i:j],
            documents=texts[i:j],
            metadatas=[{"chunk_index": k, "page": pages[k], "source": source} for k in range(i, j)],
        )

    # The BM25 index is built from the same chunks in the same order, so
    # the sparse and dense sides never drift apart. reset_collection() has
    # just dropped the old sparse index, so this always rebuilds rather
    # than appending.
    sparse.build(texts, pages, session_id)

    return len(chunks)


def ingest_pdf(pdf_path: str | Path, session_id: str | None = None) -> int:
    path = Path(pdf_path)
    chunks = split_with_pages(extract_pages(path))
    return ingest_chunks(chunks, source=path.name, session_id=session_id)


def ingest_pdf_bytes(
    content: bytes,
    source: str,
    session_id: str | None = None,
) -> int:
    """Ingest straight from memory. Used by the upload endpoint.

    PyMuPDF reads from a byte stream, so there is no temp file to write,
    read back and then fail to clean up - which was the source of the
    Phase 0 temp-file leak.
    """
    doc = fitz.open(stream=content, filetype="pdf")
    try:
        pages = [(i + 1, page.get_text()) for i, page in enumerate(doc)]
    finally:
        doc.close()

    return ingest_chunks(split_with_pages(pages), source=source, session_id=session_id)


def looks_like_pdf(content: bytes) -> bool:
    """Check the magic bytes, not the filename.

    Renaming anything to .pdf used to be enough to get past validation.
    """
    return content[:4] == PDF_MAGIC


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m core.ingest <path-to.pdf>")
        raise SystemExit(1)

    n = ingest_pdf(sys.argv[1])
    print(f"Indexed {n} chunks into '{store.collection_name()}' (dense + BM25)")
