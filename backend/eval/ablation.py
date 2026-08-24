"""Ablation study: measure what each retrieval feature actually contributes.

Run with:  python -m eval.ablation

Toggles the feature flags at runtime and re-runs the same questions through
each configuration, reporting Recall@k and MRR rather than pass/fail.

Why not pass/fail: the existing benchmark already sits at 5/5, so it cannot
show improvement OR regression - every change looks identical. MRR moves
even when pass/fail does not, because it measures WHERE the right chunk
landed, not merely whether it appeared somewhere in the top-k.

Writes eval/results/ablation.md.
"""

import warnings
from pathlib import Path

from config import settings
from core import sparse, store
from core.nim_client import embed
from core.reranker import rerank

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parent / "results" / "ablation.md"

# A chunk counts as relevant if it contains any of these strings. Coarse,
# but it needs no manual labelling and it is consistent across configs,
# which is what a comparison requires. Phase 3 replaces this with gold
# page labels in eval/datasets/qa.jsonl.
QUESTIONS = [
    {
        "q": "What attention mechanism does Falcon use?",
        "relevant": ["multigroup attention", "multiquery"],
    },
    {
        "q": "Who built the Falcon language models?",
        "relevant": ["technology innovation institute"],
    },
    {
        "q": "How many tokens was Falcon-180B trained on?",
        "relevant": ["3,500"],
    },
    {
        "q": "What is RefinedWeb?",
        "relevant": ["refinedweb"],
    },
    {
        "q": "What GPU infrastructure was used to train Falcon-180B?",
        "relevant": ["a100", "4,096"],
    },
    {
        "q": "What is the license for Falcon-7B?",
        "relevant": ["apache 2.0"],
    },
    {
        "q": "Who created Falcon?",  # the original Bug 2 question
        "relevant": ["technology innovation institute"],
    },
    {
        "q": "What tokenizer does Falcon use?",
        "relevant": ["tokenizer", "byte-pair", "bpe"],
    },
    {
        "q": "How was the training data deduplicated?",
        "relevant": ["dedup", "minhash", "exact"],
    },
    {
        "q": "What benchmarks did Falcon evaluate on?",
        "relevant": ["benchmark", "mmlu", "hellaswag"],
    },
]

CONFIGS = [
    {"name": "dense only", "hybrid": False, "rerank": False},
    {"name": "+ hybrid BM25", "hybrid": True, "rerank": False},
    {"name": "+ reranker", "hybrid": False, "rerank": True},
    {"name": "+ both", "hybrid": True, "rerank": True},
]


def is_relevant(chunk: str, needles: list[str]) -> bool:
    low = chunk.lower()
    return any(n.lower() in low for n in needles)


def retrieve(question: str, hybrid: bool, use_rerank: bool) -> list[str]:
    """Same pipeline as retriever_node, with flags passed explicitly."""
    n = settings.retrieve_candidates if (hybrid or use_rerank) else settings.final_top_k

    res = store.get_collection().query(
        query_embeddings=[embed(question, input_type="query")],
        n_results=n,
        include=["documents", "distances"],
    )
    texts = list(res["documents"][0])
    order = list(range(len(texts)))

    if hybrid:
        index = sparse.load()
        if index is not None:
            pos = {t: i for i, t in enumerate(texts)}
            bm25_order = []
            for i in index.search(question, settings.retrieve_candidates):
                t = index.texts[i]
                if t not in pos:
                    pos[t] = len(texts)
                    texts.append(t)
                bm25_order.append(pos[t])
            order = sparse.reciprocal_rank_fusion([order, bm25_order])

    if use_rerank and len(order) > 1:
        ranked = rerank(question, [texts[i] for i in order], settings.final_top_k)
        order = [order[i] for i, _ in ranked]

    return [texts[i] for i in order[: settings.final_top_k]]


def score_config(hybrid: bool, use_rerank: bool) -> dict:
    recall_1 = recall_3 = recall_5 = 0
    rr_total = 0.0

    for item in QUESTIONS:
        chunks = retrieve(item["q"], hybrid, use_rerank)
        hits = [i for i, c in enumerate(chunks) if is_relevant(c, item["relevant"])]

        if hits:
            first = hits[0]
            rr_total += 1.0 / (first + 1)
            recall_5 += 1
            if first < 3:
                recall_3 += 1
            if first == 0:
                recall_1 += 1

    n = len(QUESTIONS)
    return {
        "recall@1": recall_1 / n,
        "recall@3": recall_3 / n,
        "recall@5": recall_5 / n,
        "mrr": rr_total / n,
    }


def run() -> None:
    if store.count() == 0:
        raise SystemExit("Collection empty. Run: python -m core.ingest eval/datasets/test.pdf")
    if sparse.load() is None:
        raise SystemExit("No BM25 index. Re-run ingest after adding sparse.build().")

    print("=" * 68)
    print(f"  Ablation study  ({len(QUESTIONS)} questions, {store.count()} chunks)")
    print(f"  reranker backend: {settings.rerank_backend}")
    print("=" * 68)
    print(f"\n  {'config':<16} {'R@1':>7} {'R@3':>7} {'R@5':>7} {'MRR':>7}")
    print("  " + "-" * 50)

    rows = []
    for cfg in CONFIGS:
        m = score_config(cfg["hybrid"], cfg["rerank"])
        rows.append({**cfg, **m})
        print(
            f"  {cfg['name']:<16} {m['recall@1']:>7.2f} {m['recall@3']:>7.2f} "
            f"{m['recall@5']:>7.2f} {m['mrr']:>7.3f}"
        )

    base = rows[0]
    best = max(rows, key=lambda r: r["mrr"])

    print("\n" + "=" * 68)
    print(f"  Best by MRR: {best['name']}  ({best['mrr']:.3f} vs {base['mrr']:.3f} baseline)")
    if best["mrr"] <= base["mrr"]:
        print("  NOTE: nothing beat dense-only on this set. Report that honestly -")
        print("        n=10 on one document may simply be too small to show a gap.")
    print("=" * 68)

    _write(rows, base)
    print(f"\nWrote {RESULTS}")


def _write(rows: list[dict], base: dict) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Retrieval ablation",
        "",
        f"{len(QUESTIONS)} questions against the Falcon paper "
        f"({store.count()} chunks). Reranker backend: `{settings.rerank_backend}`.",
        "",
        "| config | Recall@1 | Recall@3 | Recall@5 | MRR | ΔMRR |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        delta = r["mrr"] - base["mrr"]
        sign = "+" if delta > 0 else ""
        lines.append(
            f"| {r['name']} | {r['recall@1']:.2f} | {r['recall@3']:.2f} | "
            f"{r['recall@5']:.2f} | {r['mrr']:.3f} | {sign}{delta:.3f} |"
        )

    lines += [
        "",
        "## Reading this table",
        "",
        "Recall@5 says whether the right chunk made it into the context at all.",
        "MRR says how high it landed. They move independently: a reranker that",
        "lifts the right chunk from position 4 to position 1 leaves Recall@5",
        "untouched while MRR rises sharply - and position matters, because a",
        "chunk buried at rank 5 competes with four irrelevant ones for the",
        "model's attention.",
        "",
        "## Caveat",
        "",
        f"n={len(QUESTIONS)} on a single document. Relevance is judged by keyword",
        "presence, not by human labelling. Treat differences under ~0.05 MRR as",
        "noise. Phase 3 re-runs this against 48 questions over 4 documents with",
        "gold page labels.",
        "",
    ]
    RESULTS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
