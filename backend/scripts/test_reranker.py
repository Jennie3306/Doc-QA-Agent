"""Smoke-test the NVIDIA reranking API before wiring it into the pipeline.

Run with:  python -m scripts.test_reranker

Why this exists as a standalone script: the reranker is the only part of
Phase 2 whose API might simply not work. NVIDIA renamed the model family
to "Nemotron" and the older ids carry a deprecation notice, so the name
and the endpoint both need verifying against a live call rather than
against documentation. Finding that out here costs 30 seconds; finding it
out after the retriever has been rewritten costs an afternoon.

The script tries several endpoint/model combinations and reports which
one answers. Put the winner in config.py.
"""

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NVIDIA_API_KEY")
if not API_KEY:
    sys.exit("NVIDIA_API_KEY not set")

QUERY = "What is the GPU memory bandwidth of H100 SXM?"
PASSAGES = [
    "The Hopper GPU is paired with the Grace CPU using NVIDIA's chip-to-chip "
    "interconnect, delivering 900GB/s of bandwidth.",
    "A100 provides up to 20X higher performance over the prior generation.",
    "Accelerated servers with H100 deliver 3 terabytes per second (TB/s) of "
    "memory bandwidth per GPU.",
    "The capital of France is Paris.",
]
# Passage 2 (index 2) is the correct answer. A working reranker must put it
# first; if it does not, the call succeeded but the model is not usable.
EXPECTED_TOP = 2

# Ordered by how likely each is to be the current one. The URL path uses
# an underscore in "3_2" while the model id uses a dot - that mismatch is
# in NVIDIA's own docs and is easy to get wrong.
CANDIDATES = [
    (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ),
    (
        "https://ai.api.nvidia.com/v1/ranking",
        "nvidia/llama-nemotron-rerank-1b-v2",
    ),
    (
        "https://ai.api.nvidia.com/v1/ranking",
        "nvidia/llama-3.2-nv-rerankqa-1b-v2",
    ),
    (
        "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking",
        "nvidia/nv-rerankqa-mistral-4b-v3",
    ),
    (
        "https://integrate.api.nvidia.com/v1/ranking",
        "nvidia/llama-nemotron-rerank-1b-v2",
    ),
]


def try_one(url: str, model: str) -> tuple[bool, str]:
    payload = {
        "model": model,
        "query": {"text": QUERY},
        "passages": [{"text": p} for p in PASSAGES],
        "truncate": "END",
    }
    try:
        r = httpx.post(
            url,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30.0,
        )
    except httpx.HTTPError as e:
        return False, f"connection error: {e}"

    if r.status_code != 200:
        body = r.text[:160].replace("\n", " ")
        return False, f"HTTP {r.status_code}: {body}"

    data = r.json()
    rankings = data.get("rankings")
    if not rankings:
        return False, f"unexpected response shape: {json.dumps(data)[:160]}"

    order = [item["index"] for item in rankings]
    top = order[0]
    ok = top == EXPECTED_TOP
    detail = f"order={order}, top logit={rankings[0].get('logit'):.3f}"
    if not ok:
        detail += f"  <- expected index {EXPECTED_TOP} first"
    return ok, detail


def main() -> None:
    print("=" * 72)
    print("  NVIDIA Reranker API smoke test")
    print("=" * 72)

    winners = []
    for url, model in CANDIDATES:
        print(f"\n  model : {model}")
        print(f"  url   : {url}")
        ok, detail = try_one(url, model)
        print(f"  {'PASS' if ok else 'FAIL'}  {detail}")
        if ok:
            winners.append((url, model))

    print("\n" + "=" * 72)
    if not winners:
        print("  No working combination found.")
        print("  Check https://build.nvidia.com -> Retrieval tab for the")
        print("  current reranking model, and copy its Python example.")
        print("  Phase 2 Stage C can be skipped - hybrid search alone still")
        print("  fixes most of the Bug 2 retrieval failures.")
    else:
        url, model = winners[0]
        print("  WORKING - put these in config.py:")
        print(f'    rerank_model: str = "{model}"')
        print(f'    rerank_url: str = "{url}"')
    print("=" * 72)


if __name__ == "__main__":
    main()
