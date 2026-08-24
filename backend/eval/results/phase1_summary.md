# Phase 1 — Correctness fixes

| | Trước | Sau |
|---|---|---|
| Ingest 471 chunks | ~180s | **16s** |
| Distance metric | squared L2 (code nói cosine) | cosine |
| Confidence | avg của top-5, công thức `1 - d/2` | top-1, `1 - d` |
| Threshold | 0.05, chưa từng trigger | 0.35, calibrate trên 24 câu |
| Evidence panel score | `confidence - i*0.05` (bịa) | cosine similarity thật |
| Upload | block event loop | `asyncio.to_thread` |
| Collection | một cái dùng chung | một cái mỗi session |

## Threshold calibration
Xem `threshold_calibration.md`. Kết luận: threshold tách được câu lạc đề
(out_easy median 0.284) khỏi câu thật (in_scope median 0.560), nhưng
KHÔNG tách được câu cùng chủ đề mà fact không tồn tại (out_hard median
0.505). Đó là giới hạn bản chất của distance-based confidence.

## Regression check
- retrieval_benchmark: 5/5 (không đổi sau khi sang cosine)
- evaluator: 6/6 (không đổi)
- pytest: 18 passed, 3 xfailed