"""Retrieval-only benchmark: does the gold keyword appear in the top-k chunks?

⚠️ Phase 3: keyword-in-chunk is a coarse proxy. Replaced by Recall@k
   and MRR against gold page labels once eval/datasets/qa.jsonl exists.
"""
import warnings

from core import store
from core.nim_client import embed

warnings.filterwarnings("ignore")

BENCHMARK = [
    {
        "question": "What attention mechanism does Falcon use?",
        "must_contain": ["multigroup", "multi-query"],
    },
    {
        "question": "Who built the Falcon language models?",
        "must_contain": ["technology innovation institute", "tii"],
    },
    {
        "question": "How many tokens was Falcon-180B trained on?",
        "must_contain": ["3,500", "3500", "trillion"],
    },
    {
        "question": "What is RefinedWeb?",
        "must_contain": ["common crawl", "web", "filtered"],
    },
    {
        "question": "What GPU infrastructure was used to train Falcon-180B?",
        "must_contain": ["a100", "4,096", "aws"],
    },
]


def benchmark_retrieval(collection, top_k: int = 5) -> tuple[float, int, int]:
    passed = 0
    total = len(BENCHMARK)

    print(f"\n{'Question':<45} {'Result':<10} {'Keywords found'}")
    print("-" * 80)

    for test in BENCHMARK:
        question = test["question"]
        must_contain = test["must_contain"]

        results = collection.query(
            query_embeddings=[embed(question, input_type="query")],
            n_results=top_k,
        )
        all_chunks_text = " ".join(results["documents"][0]).lower()
        found = [k for k in must_contain if k.lower() in all_chunks_text]

        if found:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        short_q = question[:43] + ".." if len(question) > 43 else question
        print(f"  {short_q:<45} {status:<10} {found}")

    return (passed / total) * 100, passed, total


def run_benchmark() -> None:
    print("=" * 55)
    print("  Retrieval Benchmark")
    print("=" * 55)

    collection = store.get_collection()
    print(f"\nDatabase: {collection.count()} chunks")

    print("\n--- Strategy 1: top_k=3 (original) ---")
    score3, passed3, total = benchmark_retrieval(collection, top_k=3)

    print("\n--- Strategy 2: top_k=5 (optimized) ---")
    score5, passed5, total = benchmark_retrieval(collection, top_k=5)

    print("\n" + "=" * 55)
    print("  Benchmark Summary")
    print("=" * 55)
    print(f"  top_k=3: {passed3}/{total} ({score3:.0f}%)")
    print(f"  top_k=5: {passed5}/{total} ({score5:.0f}%)")

    improvement = score5 - score3
    if improvement > 0:
        print(f"  Improvement: +{improvement:.0f}% from increasing top_k")
    else:
        print("  top_k=3 and top_k=5 perform equally on this dataset")
    print("=" * 55)


if __name__ == "__main__":
    run_benchmark()