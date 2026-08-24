"""Central configuration. Every tunable parameter lives here.

Nothing else in the codebase should hardcode a model name, chunk size,
top-k value, threshold or file path.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Credentials ────────────────────────────────────────────
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── Models ─────────────────────────────────────────────────
    llm_model: str = "nvidia/nemotron-3-nano-30b-a3b"
    embed_model: str = "nvidia/nv-embedqa-e5-v5"

    # ── Chunking ───────────────────────────────────────────────
    # 500 chosen empirically: the proxy embedding score preferred 300, but
    # real retrieval accuracy was 5/5 at 500 vs 4/5 at 300 - author names
    # were being split across small chunks.
    # See eval/results/chunk_size_experiment.md
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_batch_size: int = 64

    # ── Retrieval ──────────────────────────────────────────────
    final_top_k: int = 5

    # Candidates pulled before fusion/reranking. Only used when hybrid or
    # the reranker is on; dense-only fetches final_top_k directly.
    retrieve_candidates: int = 20

    # Calibrated empirically - see eval/results/threshold_calibration.md
    # Separates off-topic questions (out_easy median 0.284) from real ones
    # (in_scope median 0.560). Does NOT catch same-topic questions whose
    # answer is absent (out_hard median 0.505); that overlap is inherent to
    # distance-based confidence and is handled by the system prompt instead.
    confidence_threshold: float = 0.35

    # ── Feature flags ──────────────────────────────────────────
    # Every retrieval feature is toggleable so eval/ablation.py can measure
    # each one independently. Without flags there is no ablation table, and
    # the ablation table is the most valuable output of Phase 2.
    use_hybrid_search: bool = True  # Stage B - BM25 + RRF
    use_reranker: bool = True  # Stage C - cross-encoder
    use_query_rewriting: bool = False  # Stage D - not built yet

    # ── Reranker ───────────────────────────────────────────────
    # "nim" (hosted API) or "local" (sentence-transformers, no network).
    #
    # Two backends exist because the endpoint NVIDIA's own docs pointed at
    # returned `410 Gone - reached its end of life on 2026-05-18` during
    # development. The values below were verified with a live call:
    # the correct passage ranked first at logit +8.2, the distractors at
    # -12 to -17. Re-verify with: python -m scripts.test_reranker
    rerank_backend: str = "nim"
    rerank_model: str = "nvidia/rerank-qa-mistral-4b"
    rerank_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/reranking"
    rerank_local_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_max_passages: int = 100  # API allows up to 512
    rerank_timeout: float = 20.0

    # ── Generation ─────────────────────────────────────────────
    max_tokens_answer: int = 1024
    max_tokens_meta: int = 600
    history_window: int = 6  # messages passed to the generator
    history_window_meta: int = 8

    # ── Storage ────────────────────────────────────────────────
    # Absolute, so the DB is found regardless of which directory uvicorn
    # was launched from. A relative path silently created a second, empty
    # database when the working directory changed.
    chroma_path: str = str(BACKEND_DIR / "chroma_db")
    collection_name: str = "nvidia_docs"

    # ── API ────────────────────────────────────────────────────
    # Comma-separated in .env, e.g. CORS_ORIGINS=http://localhost:5173
    # Stored as a string because pydantic-settings parses list[str] from
    # the environment as JSON, which "*" is not.
    cors_origins_raw: str = "*"
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB
    max_history_messages: int = 12
    max_message_chars: int = 2000

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",")]


settings = Settings()
