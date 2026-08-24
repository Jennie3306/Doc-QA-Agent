"""Calibrate the confidence threshold empirically.

Run with:  python -m eval.calibrate_threshold

Requires the default-session collection to be populated:
    python -m core.ingest eval/datasets/test.pdf

The old threshold (0.05) was picked by feel and never fired, which meant
the "first line of defence against out-of-scope questions" was dead code.
This script replaces the guess with a measurement: run known in-scope and
known out-of-scope questions, look at where the two distributions sit, and
pick the value that separates them.

Writes eval/results/threshold_calibration.md.
"""

import warnings
from pathlib import Path

from config import settings
from core import store
from core.nim_client import embed

warnings.filterwarnings("ignore")

RESULTS = Path(__file__).resolve().parent / "results" / "threshold_calibration.md"

IN_SCOPE = [
    "What attention mechanism does Falcon use?",
    "Who built the Falcon language models?",
    "How many tokens was Falcon-180B trained on?",
    "What is RefinedWeb?",
    "What GPU infrastructure was used to train Falcon-180B?",
    "What is the license for Falcon-7B?",
    "How was the training data filtered?",
    "What benchmarks did Falcon evaluate on?",
    "How many parameters does Falcon-40B have?",
    "What tokenizer does Falcon use?",
    "How does multigroup attention differ from multiquery attention?",
    "What was the deduplication strategy for RefinedWeb?",
]

# Split deliberately into two difficulties. The easy set is what the
# original eval used; passing it proves almost nothing.
OUT_OF_SCOPE_EASY = [
    "What is the capital of France?",
    "Who is the CEO of NVIDIA?",
    "What is the price of the Falcon API?",
    "How do I bake sourdough bread?",
    "What is the weather in Hanoi today?",
]

# The set that matters. Same vocabulary, same topic, facts that are not in
# the document. Retrieval returns plausible-looking chunks, which is
# exactly where a RAG system hallucinates.
OUT_OF_SCOPE_HARD = [
    "What is Falcon's context window in version 3?",
    "How much did it cost to train Falcon-180B in US dollars?",
    "What is Falcon's performance on the GSM8K benchmark?",
    "How many people work at the Technology Innovation Institute?",
    "What learning rate schedule did Falcon-7B use in fine-tuning?",
    "Which cloud provider hosts the Falcon inference API?",
    "What is the carbon footprint of training Falcon-40B?",
]


def top1_score(question: str, collection) -> float:
    results = collection.query(
        query_embeddings=[embed(question, input_type="query")],
        n_results=settings.final_top_k,
        include=["distances"],
    )
    distances = results["distances"][0]
    return store.distance_to_similarity(distances[0])


def summarise(name: str, scores: list[float]) -> dict:
    ordered = sorted(scores)
    return {
        "name": name,
        "n": len(ordered),
        "min": ordered[0],
        "max": ordered[-1],
        "mean": sum(ordered) / len(ordered),
        "median": ordered[len(ordered) // 2],
        "scores": ordered,
    }


def evaluate_threshold(t: float, in_scope: list[float], out_scope: list[float]) -> dict:
    """False negative = a real question wrongly refused.
    False positive = an unanswerable question wrongly allowed through."""
    fn = sum(1 for s in in_scope if s < t)
    fp = sum(1 for s in out_scope if s >= t)
    return {
        "threshold": t,
        "false_negatives": fn,
        "false_positives": fp,
        "fn_rate": fn / len(in_scope),
        "fp_rate": fp / len(out_scope),
    }


def run() -> None:
    collection = store.get_collection()
    n = collection.count()
    if n == 0:
        raise SystemExit("Collection is empty. Run: python -m core.ingest eval/datasets/test.pdf")

    print("=" * 62)
    print("  Confidence Threshold Calibration")
    print(f"  Collection: {store.collection_name()} ({n} chunks)")
    print("=" * 62)

    groups = {}
    for label, questions in [
        ("in_scope", IN_SCOPE),
        ("out_easy", OUT_OF_SCOPE_EASY),
        ("out_hard", OUT_OF_SCOPE_HARD),
    ]:
        print(f"\n--- {label} ({len(questions)} questions) ---")
        scores = []
        for q in questions:
            s = top1_score(q, collection)
            scores.append(s)
            print(f"  {s:.3f}  {q[:55]}")
        groups[label] = summarise(label, scores)

    print("\n" + "=" * 62)
    print("  Distributions")
    print("=" * 62)
    print(f"  {'group':<10} {'n':>3} {'min':>7} {'median':>8} {'mean':>7} {'max':>7}")
    for g in groups.values():
        print(
            f"  {g['name']:<10} {g['n']:>3} {g['min']:>7.3f} "
            f"{g['median']:>8.3f} {g['mean']:>7.3f} {g['max']:>7.3f}"
        )

    in_scores = groups["in_scope"]["scores"]
    out_scores = groups["out_easy"]["scores"] + groups["out_hard"]["scores"]

    gap_lo = max(out_scores)
    gap_hi = min(in_scores)
    separable = gap_hi > gap_lo

    print("\n" + "=" * 62)
    print("  Separability")
    print("=" * 62)
    print(f"  highest out-of-scope score : {gap_lo:.3f}")
    print(f"  lowest  in-scope     score : {gap_hi:.3f}")
    if separable:
        print(f"  CLEAN GAP - any threshold in ({gap_lo:.3f}, {gap_hi:.3f}) separates")
        recommended = round((gap_lo + gap_hi) / 2, 2)
    else:
        print("  OVERLAP - no threshold separates perfectly; pick a trade-off")
        recommended = None

    print("\n" + "=" * 62)
    print("  Threshold sweep")
    print("=" * 62)
    print(f"  {'thr':>5} {'FN':>4} {'FP':>4} {'FN rate':>9} {'FP rate':>9}")

    rows = []
    for i in range(0, 21):
        t = i / 20
        r = evaluate_threshold(t, in_scores, out_scores)
        rows.append(r)
        print(
            f"  {t:>5.2f} {r['false_negatives']:>4} {r['false_positives']:>4} "
            f"{r['fn_rate']:>8.0%} {r['fp_rate']:>8.0%}"
        )

    if recommended is None:
        # Prefer letting a borderline question through over refusing a real
        # one: a wrong answer is visible to the user, a wrongly refused
        # question just looks broken.
        viable = [r for r in rows if r["fn_rate"] == 0]
        best = min(viable, key=lambda r: r["fp_rate"]) if viable else rows[0]
        recommended = best["threshold"]

    print("\n" + "=" * 62)
    print(f"  RECOMMENDED confidence_threshold = {recommended}")
    print(f"  (currently {settings.confidence_threshold} in config.py)")
    print("=" * 62)

    _write_report(groups, gap_lo, gap_hi, separable, rows, recommended)
    print(f"\nWrote {RESULTS}")


def _write_report(groups, gap_lo, gap_hi, separable, rows, recommended) -> None:
    RESULTS.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Confidence Threshold Calibration",
        "",
        "Top-1 cosine similarity, measured on the Falcon paper collection.",
        "",
        "## Distributions",
        "",
        "| group | n | min | median | mean | max |",
        "|---|---|---|---|---|---|",
    ]
    for g in groups.values():
        lines.append(
            f"| {g['name']} | {g['n']} | {g['min']:.3f} | "
            f"{g['median']:.3f} | {g['mean']:.3f} | {g['max']:.3f} |"
        )

    lines += [
        "",
        "## Separability",
        "",
        f"- Highest out-of-scope score: **{gap_lo:.3f}**",
        f"- Lowest in-scope score: **{gap_hi:.3f}**",
        f"- {'Clean gap' if separable else 'Distributions overlap'}",
        "",
        "## Threshold sweep",
        "",
        "FN = a real question wrongly refused. FP = an unanswerable question",
        "wrongly allowed through to the generator.",
        "",
        "| threshold | FN | FP | FN rate | FP rate |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['threshold']:.2f} | {r['false_negatives']} | "
            f"{r['false_positives']} | {r['fn_rate']:.0%} | {r['fp_rate']:.0%} |"
        )

    lines += [
        "",
        f"## Chosen: `confidence_threshold = {recommended}`",
        "",
        "Bias is toward false positives over false negatives: a wrong answer",
        "is visible and checkable by the user, whereas a wrongly refused",
        "question just makes the system look broken.",
        "",
        "## Caveat",
        "",
        f"Measured on {sum(g['n'] for g in groups.values())} questions against a",
        "single document. The separation is indicative, not established -",
        "Phase 3 re-runs this across four documents.",
        "",
    ]

    RESULTS.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run()
