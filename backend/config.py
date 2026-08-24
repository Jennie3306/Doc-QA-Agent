from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Credentials ────────────────────────────────────────
    nvidia_api_key: str
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # ── Models ─────────────────────────────────────────────
    llm_model: str = "nvidia/nemotron-3-nano-30b-a3b"
    embed_model: str = "nvidia/nv-embedqa-e5-v5"

    # ── Chunking ───────────────────────────────────────────
    # 500 chosen empirically: see eval/results/chunk_size_experiment.md
    chunk_size: int = 500
    chunk_overlap: int = 50
    embed_batch_size: int = 64 

    # ── Retrieval ──────────────────────────────────────────
    final_top_k: int = 5
    # Calibrated empirically — see eval/results/threshold_calibration.md
    # Separates off-topic questions (out_easy median 0.284) from real ones
    # (in_scope median 0.560). Does NOT catch same-topic questions whose
    # answer is absent (out_hard median 0.505) — that overlap is inherent
    # to distance-based confidence and is handled by the system prompt.
    confidence_threshold: float = 0.35

    # ── Generation ─────────────────────────────────────────
    max_tokens_answer: int = 1024
    max_tokens_meta: int = 600
    history_window: int = 6              # messages passed to generator
    history_window_meta: int = 8

    # ── Storage ────────────────────────────────────────────
    chroma_path: str = str(BACKEND_DIR / "chroma_db")
    collection_name: str = "nvidia_docs"

    # ── API ────────────────────────────────────────────────
    # Comma-separated in .env, e.g. CORS_ORIGINS=http://localhost:5173
    cors_origins_raw: str = "*"
    max_upload_bytes: int = 20 * 1024 * 1024   # 20 MB
    max_history_messages: int = 12
    max_message_chars: int = 2000 

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",")]


settings = Settings()