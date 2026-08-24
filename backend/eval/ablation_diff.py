"""Per-question ablation diff.

Run with:  python -m eval.ablation_diff

The aggregate table cannot support conclusions at n=10 - every 0.10 gap is
a single question. This shows WHICH question moved and in which direction,
which is diagnostic rather than statistical: "hybrid search rescued the
exact question that failed in Bug 2" is a claim worth making, while
"hybrid improves MRR by 0.033" is not.
"""

import warnings

from eval.ablation import CONFIGS, QUESTIONS, is_relevant, retrieve

warnings.filterwarnings("ignore")


def rank_of_first_relevant(chunks: list[str], needles: list[str]) -> int | None:
    for i, c in enumerate(chunks):
        if is_relevant(c, needles):
            return i + 1  # 1-indexed
    return None


def run() -> None:
    print("=" * 78)
    print("  Per-question rank of the first relevant chunk (lower is better)")
    print("  'miss' = no relevant chunk in the top 5")
    print("=" * 78)

    names = [c["name"] for c in CONFIGS]
    header = f"  {'question':<44}" + "".join(f"{n[:11]:>12}" for n in names)
    print("\n" + header)
    print("  " + "-" * (44 + 12 * len(names)))

    results: dict[str, list] = {}

    for item in QUESTIONS:
        row = []
        for cfg in CONFIGS:
            chunks = retrieve(item["q"], cfg["hybrid"], cfg["rerank"])
            row.append(rank_of_first_relevant(chunks, item["relevant"]))
        results[item["q"]] = row

        cells = "".join(f"{(str(r) if r else 'miss'):>12}" for r in row)
        q = item["q"][:42] + ".." if len(item["q"]) > 42 else item["q"]
        print(f"  {q:<44}{cells}")

    # ── What actually changed ─────────────────────────────────
    print("\n" + "=" * 78)
    print("  Changes vs dense-only")
    print("=" * 78)

    any_change = False
    for q, row in results.items():
        base = row[0]
        for i, cfg in enumerate(CONFIGS[1:], start=1):
            cur = row[i]
            if cur == base:
                continue
            any_change = True
            if base is None:
                verdict = f"RESCUED (miss -> rank {cur})"
            elif cur is None:
                verdict = f"LOST (rank {base} -> miss)"
            elif cur < base:
                verdict = f"improved (rank {base} -> {cur})"
            else:
                verdict = f"WORSE (rank {base} -> {cur})"
            print(f"  [{cfg['name']}] {q[:45]}")
            print(f"      {verdict}")

    if not any_change:
        print("  No question changed rank in any configuration.")

    print("\n" + "=" * 78)
    print("  A 'RESCUED' row is the strongest single result here: a question")
    print("  that dense retrieval could not answer at all now can be. That is")
    print("  a qualitative change, and it survives the small sample size in a")
    print("  way that a 0.03 MRR delta does not.")
    print("=" * 78)


if __name__ == "__main__":
    run()
