# Retrieval ablation

10 questions against the Falcon paper (468 chunks). Reranker backend: `nim`.

| config | Recall@1 | Recall@3 | Recall@5 | MRR | ΔMRR |
|---|---|---|---|---|---|
| dense only | 0.80 | 0.90 | 0.90 | 0.850 | 0.000 |
| + hybrid BM25 | 0.80 | 1.00 | 1.00 | 0.883 | +0.033 |
| + reranker | 0.70 | 1.00 | 1.00 | 0.850 | 0.000 |
| + both | 0.70 | 1.00 | 1.00 | 0.850 | 0.000 |

## Reading this table

**Recall@5 is the metric that matters here.** It says whether the right chunk
made it into the context at all — and since all five retrieved chunks are
injected into the prompt together, that is the only thing that changes the
answer.

MRR is reported for completeness, not as a decision criterion. See Findings
below for why it misleads in this setting.

## Findings

**Hybrid BM25 rescued the Bug 2 question.** "Who created Falcon?" went from
miss to rank 1. This is the question that failed in week 7, where the earlier
"fix" was to reword the question — not a fix. Both hybrid and the reranker
rescue it independently, which corroborates the diagnosis: the bi-encoder
dilutes proper nouns, and both BM25 (exact token match) and a cross-encoder
(joint query-passage encoding) recover what it lost.

**The apparent regressions are not regressions.** Two questions moved from
rank 1 to rank 2-3. All five retrieved chunks go into the prompt, so a chunk
at rank 3 is just as available to the model as one at rank 1.

**Which means MRR is the wrong metric here.** MRR penalises a 1→2 shift
heavily (1.0 → 0.5) because it was designed for search interfaces, where the
user clicks the first result. In a RAG pipeline that feeds the whole top-k
into the context, the only thing that matters is whether the right chunk is
in there at all — Recall@k. Reported MRR above for completeness, but Recall@5
is the number to read.

**The reranker adds nothing over hybrid on this set** — both reach Recall@5
1.00 — while costing ~300ms and an external dependency that returned 410 Gone
once already. This benchmark is saturated (9/10 already at rank 1 with dense
alone), so there is no headroom to show a difference. Keeping it enabled for
Phase 3 to judge on multi-hop and adversarial questions; if it still shows
nothing there, turn it off.

## Caveat

n=10 on a single document. Relevance is judged by keyword
presence, not by human labelling. Treat any single-question difference as anecdotal: at n=10, one question is 0.10 of Recall. The one result strong enough to report is a miss becoming a hit, because that is a change in kind rather than in degree. Phase 3 re-runs this against 48 questions over 4 documents with
gold page labels.