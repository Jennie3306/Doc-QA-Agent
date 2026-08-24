"""Chunk size A/B experiment.

Run with:  python -m eval.chunk_optimizer

Uses a SEPARATE ChromaDB instance (chroma_db_test) so the production
collection is never touched. That is why it keeps its own PersistentClient
instead of going through core.store.

IMPORTANT FINDING: the score below is average embedding distance, a PROXY
metric. It disagreed with real retrieval accuracy:

    chunk_size=300 -> proxy 0.536, real accuracy 4/5
    chunk_size=500 -> proxy 0.526, real accuracy 5/5  <- chosen
    chunk_size=800 -> proxy 0.504, real accuracy 5/5

300 scored best on the proxy but failed "Who built Falcon?" because the
author names were split across several small chunks. Production uses 500:
real-world accuracy beats proxy metrics.
"""

import warnings
from pathlib import Path

import chromadb
import fitz
from chromadb.errors import NotFoundError
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.nim_client import embed

warnings.filterwarnings("ignore")

BACKEND_DIR = Path(__file__).resolve().parent.parent
TEST_PDF = BACKEND_DIR / "eval" / "datasets" / "test.pdf"
TEST_DB = BACKEND_DIR / "chroma_db_test"

TEST_QUESTIONS = [
    "What attention mechanism does Falcon use?",
    "Who built the Falcon language models?",
    "How many tokens was Falcon-180B trained on?",
    "What is RefinedWeb?",
    "What benchmarks did Falcon achieve?",
]

CONFIGS = [
    {"chunk_size": 300, "overlap": 30, "name": "small"},
    {"chunk_size": 500, "overlap": 50, "name": "medium"},
    {"chunk_size": 800, "overlap": 80, "name": "large"},
]


def load_pdf(path: Path) -> str:
    doc = fitz.open(str(path))
    text = "".join(f"\n--- Page {i + 1} ---\n{page.get_text()}" for i, page in enumerate(doc))
    doc.close()
    return text


def build_collection(text: str, chunk_size: int, chunk_overlap: int, name: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_text(text)

    db = chromadb.PersistentClient(path=str(TEST_DB))
    try:
        db.delete_collection(name)
    except NotFoundError:
        pass
    collection = db.get_or_create_collection(name)

    for i, chunk in enumerate(chunks):
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embed(chunk, input_type="passage")],
            documents=[chunk],
            metadatas=[{"chunk_index": i}],
        )

    return collection, len(chunks)


def test_retrieval(collection, questions: list[str], top_k: int = 3) -> float:
    """Average embedding-distance score. A proxy - see module docstring."""
    total_score = 0.0

    for question in questions:
        results = collection.query(
            query_embeddings=[embed(question, input_type="query")],
            n_results=top_k,
            include=["distances"],
        )
        distances = results["distances"][0]
        avg_distance = sum(distances) / len(distances)
        # Lower distance = better retrieval
        total_score += max(0.0, 1 - (avg_distance / 2))

    return total_score / len(questions)


def run_optimization() -> int:
    print("=" * 55)
    print("  Chunk Size Optimization")
    print("=" * 55)

    if not TEST_PDF.exists():
        raise FileNotFoundError(f"Test PDF not found at {TEST_PDF}")

    text = load_pdf(TEST_PDF)
    results = []

    for config in CONFIGS:
        print(f"\nTesting chunk_size={config['chunk_size']}...")

        collection, num_chunks = build_collection(
            text,
            config["chunk_size"],
            config["overlap"],
            f"test_{config['name']}",
        )
        score = test_retrieval(collection, TEST_QUESTIONS)

        results.append(
            {
                "name": config["name"],
                "chunk_size": config["chunk_size"],
                "num_chunks": num_chunks,
                "score": score,
            }
        )

        print(f"  Chunks created: {num_chunks}")
        print(f"  Proxy score: {score:.3f}")

    best = max(results, key=lambda x: x["score"])

    print("\n" + "=" * 55)
    print("  Results Summary")
    print("=" * 55)
    for r in results:
        marker = " <- highest proxy" if r["name"] == best["name"] else ""
        print(
            f"  {r['name']:8} | size={r['chunk_size']} | "
            f"chunks={r['num_chunks']:4} | score={r['score']:.3f}{marker}"
        )

    print(f"\n  Highest proxy score: chunk_size={best['chunk_size']}")
    print("  NOTE: production uses 500 - the proxy score disagreed with")
    print("        real retrieval accuracy. See the module docstring.")
    print("=" * 55)

    # Cleanup test collections
    db = chromadb.PersistentClient(path=str(TEST_DB))
    for config in CONFIGS:
        try:
            db.delete_collection(f"test_{config['name']}")
        except NotFoundError:
            pass

    return best["chunk_size"]


if __name__ == "__main__":
    run_optimization()
