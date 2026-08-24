"""LLM-as-judge scoring on a 1-5 scale.

Run with:  python -m eval.quality_scorer

Phase 3: this judge exhibits central-tendency bias - scores cluster at 3
regardless of answer quality (see Bug 6 in Interview_Preparation.md).
Replaced in Phase 3 by a binary rubric + pairwise comparison.
The absolute score below should NOT be quoted as a headline metric.
"""

import warnings

from openai import OpenAIError

from core import store
from core.nim_client import chat, embed

warnings.filterwarnings("ignore")

TOP_K = 5
MAX_TOKENS_ANSWER = 400
NO_ANSWER = "I could not generate a response."

EVAL_QUESTIONS = [
    {
        "question": "What attention mechanism does Falcon use?",
        "reference": "Falcon uses multigroup attention, an extension of multiquery "
        "attention, to improve inference scalability.",
    },
    {
        "question": "Who built the Falcon language models?",
        "reference": "The Falcon models were built by the Falcon LLM Team at the "
        "Technology Innovation Institute in Abu Dhabi.",
    },
    {
        "question": "How many tokens was Falcon-180B trained on?",
        "reference": "Falcon-180B was trained on 3,500 billion tokens (3.5 trillion tokens).",
    },
    {
        "question": "What is the license for Falcon-7B?",
        "reference": "Falcon-7B is released under the Apache 2.0 license.",
    },
]


def get_agent_answer(question: str, collection) -> str:
    """Run the RAG pipeline and return an answer, never None."""
    results = collection.query(
        query_embeddings=[embed(question, input_type="query")],
        n_results=TOP_K,
    )
    chunks = results["documents"][0]
    context = "\n\n---\n\n".join(f"Chunk {i + 1}:\n{c}" for i, c in enumerate(chunks))

    messages = [
        {
            "role": "system",
            "content": f"Answer ONLY based on this context:\n{context}\n"
            f"If not found, say: 'I could not find this in the document.'",
        },
        {"role": "user", "content": question},
    ]

    try:
        answer = chat(messages, max_tokens=MAX_TOKENS_ANSWER)
    except OpenAIError as e:
        print(f"[Scorer] API error: {e}")
        return NO_ANSWER

    return answer or NO_ANSWER


def score_answer(question: str, agent_answer: str, reference_answer: str) -> int:
    """Judge the answer 1-5. Falls back to 3 if the judge output is unusable.

    Note the failure mode: a parse failure and a genuine middling score both
    produce 3, which is exactly the value the central-tendency bias already
    pulls toward. The prints below exist so you can see how often the
    fallback actually fires - if it fires often, the reported average is
    largely parse failures rather than judgements.
    """
    judge_prompt = f"""Score this answer 1-5. Reply with ONE digit only.

Question: {question}
Reference: {reference_answer}
Answer: {agent_answer}

Rules:
- If the answer contains the same key facts as the reference -> score 5
- If the answer is mostly correct but adds unnecessary info -> score 4
- If the answer is partially correct -> score 3
- If the answer is mostly wrong -> score 2
- If the answer is completely wrong or refused -> score 1

Does the answer contain the same core facts as the reference?
Reply with a single digit 1-5:"""

    raw = ""
    try:
        raw = chat(
            [{"role": "user", "content": judge_prompt}],
            max_tokens=5,
            temperature=0.0,
        )
        return min(max(int(raw.strip()[0]), 1), 5)
    except (ValueError, IndexError):
        print(f"[Judge] Unparseable score, defaulting to 3: {raw!r}")
        return 3
    except OpenAIError as e:
        # Without this branch a mid-run API failure aborts the whole
        # evaluation and discards every result scored so far.
        print(f"[Judge] API error, defaulting to 3: {e}")
        return 3


def run_quality_scoring() -> None:
    print("=" * 55)
    print("  Answer Quality Scorer")
    print("  (Nemotron judging Nemotron - biased toward 3, see docstring)")
    print("=" * 55)

    collection = store.get_collection()

    total_score = 0
    results = []

    for test in EVAL_QUESTIONS:
        question = test["question"]

        print(f"\nQ: {question}")
        answer = get_agent_answer(question, collection)
        print(f"A: {answer[:150]}...")

        score = score_answer(question, answer, test["reference"])
        total_score += score
        results.append({"question": question, "score": score})

        stars = "*" * score + "." * (5 - score)
        print(f"Score: {stars} ({score}/5)")

    avg_score = total_score / len(EVAL_QUESTIONS)

    print("\n" + "=" * 55)
    print("  Quality Score Summary")
    print("=" * 55)
    for r in results:
        stars = "*" * r["score"] + "." * (5 - r["score"])
        print(f"  {stars}  {r['question'][:45]}")
    print(f"\n  Average Quality Score: {avg_score:.1f}/5.0")
    print("  Indicative only - do not quote as a headline metric.")
    print("=" * 55)


if __name__ == "__main__":
    run_quality_scoring()
