# Confidence Threshold Calibration

Top-1 cosine similarity, measured on the Falcon paper collection.

## Distributions

| group | n | min | median | mean | max |
|---|---|---|---|---|---|
| in_scope | 12 | 0.432 | 0.560 | 0.540 | 0.658 |
| out_easy | 5 | 0.262 | 0.284 | 0.324 | 0.504 |
| out_hard | 7 | 0.313 | 0.501 | 0.474 | 0.531 |

## Separability

- Highest out-of-scope score: **0.531**
- Lowest in-scope score: **0.432**
- Distributions overlap

## Finding

The threshold separates **off-topic** questions from real ones, but not
**same-topic questions whose answer is absent**.

- out_easy (median 0.284) is cleanly below in_scope (median 0.560)
- out_hard (median 0.505) sits inside the in_scope range

This is inherent to distance-based confidence: embedding similarity measures
topical proximity, not answer presence. "What learning rate schedule did
Falcon-7B use?" (0.531) retrieves chunks that genuinely discuss Falcon-7B
training — they are relevant, they just do not contain that fact.

Consequence: at threshold 0.35 the system refuses 4/5 off-topic questions
before spending an LLM call, but 6/7 hard cases reach the generator. The
actual defence against hallucination on those is the system prompt
instruction, verified separately.

Evidence that the prompt is doing the work: "What is the price of the Falcon
API?" scores 0.504, passes the threshold, and the model still correctly
answers "I could not find this in the document."

Phase 3 will measure abstention accuracy on the out_hard set directly,
since that is the only case that matters for hallucination.

## Threshold sweep

FN = a real question wrongly refused. FP = an unanswerable question
wrongly allowed through to the generator.

| threshold | FN | FP | FN rate | FP rate |
|---|---|---|---|---|
| 0.00 | 0 | 12 | 0% | 100% |
| 0.05 | 0 | 12 | 0% | 100% |
| 0.10 | 0 | 12 | 0% | 100% |
| 0.15 | 0 | 12 | 0% | 100% |
| 0.20 | 0 | 12 | 0% | 100% |
| 0.25 | 0 | 12 | 0% | 100% |
| 0.30 | 0 | 9 | 0% | 75% |
| 0.35 | 0 | 7 | 0% | 58% |
| 0.40 | 0 | 7 | 0% | 58% |
| 0.45 | 1 | 7 | 8% | 58% |
| 0.50 | 3 | 5 | 25% | 42% |
| 0.55 | 5 | 0 | 42% | 0% |
| 0.60 | 10 | 0 | 83% | 0% |
| 0.65 | 11 | 0 | 92% | 0% |
| 0.70 | 12 | 0 | 100% | 0% |
| 0.75 | 12 | 0 | 100% | 0% |
| 0.80 | 12 | 0 | 100% | 0% |
| 0.85 | 12 | 0 | 100% | 0% |
| 0.90 | 12 | 0 | 100% | 0% |
| 0.95 | 12 | 0 | 100% | 0% |
| 1.00 | 12 | 0 | 100% | 0% |

## Chosen: `confidence_threshold = 0.35`

Bias is toward false positives over false negatives: a wrong answer
is visible and checkable by the user, whereas a wrongly refused
question just makes the system look broken.

## Caveat

Measured on 24 questions against a
single document. The separation is indicative, not established -
Phase 3 re-runs this across four documents.