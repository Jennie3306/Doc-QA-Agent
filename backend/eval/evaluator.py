"""End-to-end evaluation: retrieve → generate → check keywords.

⚠️ Phase 3: n=6 is far too small for the "100%" claim to mean anything.
   Expanded to ~48 questions across 4 documents in Phase 3.
⚠️ Phase 3: keyword matching is a weak proxy — replaced by
   Recall@k / MRR (retrieval) and RAGAS faithfulness (generation).
"""

import warnings

from core import store
from core.nim_client import chat, embed

warnings.filterwarnings("ignore")

TOP_K = 5
MAX_TOKENS = 400

# Ground truth Q&A pairs — answers manually verified from the PDF
EVAL_SET = [
    {
        "question": "What attention mechanism does Falcon use?",
        "expected_keywords": ["multigroup", "attention"],
        "should_answer": True,
    },
    {
        "question": "What is RefinedWeb?",
        "expected_keywords": ["common crawl", "filtered", "deduplicated"],
        "should_answer": True,
    },
    {
        "question": "How many tokens was Falcon-180B trained on?",
        "expected_keywords": ["3,500", "3500", "billion"],
        "should_answer": True,
    },
    {
        "question": "Who built the Falcon language models?",
        "expected_keywords": ["technology innovation institute"],
        "should_answer": True,
    },
    {
        "question": "What is the price of the Falcon API?",
        "expected_keywords": [],
        "should_answer": False,
    },
    {
        "question": "Who is the CEO of NVIDIA?",
        "expected_keywords": [],
        "should_answer": False,
    },
]

SYSTEM_PROMPT = (
    "You are a precise document assistant.\n"
    "Answer ONLY based on the context. If not found say exactly: "
    "'I could not find this in the document.'"
)


def retrieve_chunks(question: str, collection, top_k: int = TOP_K) -> list[str]:
    results = collection.query(
        query_embeddings=[embed(question, input_type="query")],
        n_results=top_k,
    )
    return results["documents"][0]


def generate_answer(question: str, chunks: list[str]) -> str:
    context = "\n\n---\n\n".join(f"Chunk {i + 1}:\n{chunk}" for i, chunk in enumerate(chunks))
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
    ]
    return chat(messages, max_tokens=MAX_TOKENS)


def evaluate() -> None:
    collection = store.get_collection()

    print("=" * 55)
    print("  RAG Agent Evaluation")
    print("=" * 55)

    passed = 0
    total = len(EVAL_SET)

    for i, test in enumerate(EVAL_SET):
        question = test["question"]
        expected_keywords = test["expected_keywords"]
        should_answer = test["should_answer"]

        print(f"\nTest {i + 1}: {question}")
        print("-" * 40)

        chunks = retrieve_chunks(question, collection, top_k=TOP_K)
        answer = generate_answer(question, chunks)
        answer_lower = answer.lower()

        print(f"Answer: {answer[:200]}...")

        if should_answer:
            keywords_found = [k for k in expected_keywords if k.lower() in answer_lower]
            if keywords_found and "could not find" not in answer_lower:
                print(f"PASS — keywords found: {keywords_found}")
                passed += 1
            else:
                print(f"FAIL — missing keywords: {expected_keywords}")
        else:
            if "could not find" in answer_lower:
                print("PASS — correctly refused to answer")
                passed += 1
            else:
                print("FAIL — should have said 'could not find'")

    print("\n" + "=" * 55)
    score = (passed / total) * 100
    print(f"  Final Score: {passed}/{total} ({score:.0f}%)")

    if score == 100:
        print("  Perfect score! Agent is working correctly.")
    elif score >= 75:
        print("  Good performance. Review failed tests.")
    else:
        print("  Needs improvement. Check retrieval quality.")
    print("=" * 55)


if __name__ == "__main__":
    evaluate()
