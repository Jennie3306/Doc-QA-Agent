from pathlib import Path

from config import settings


def test_chroma_path_is_absolute():
    """Relative paths break when uvicorn is started from a different cwd."""
    assert Path(settings.chroma_path).is_absolute()


def test_model_names_set():
    assert settings.llm_model.startswith("nvidia/")
    assert settings.embed_model.startswith("nvidia/")


def test_chunk_overlap_smaller_than_size():
    assert 0 <= settings.chunk_overlap < settings.chunk_size


def test_api_key_present():
    assert settings.nvidia_api_key.startswith("nvapi-")


def test_settings_has_fields_used_by_main():
    """main.py reads these at request time - a missing field only surfaces
    as a 500 during an upload, which is far too late to find out."""
    for field in (
        "max_upload_bytes",
        "max_history_messages",
        "cors_origins",
        "final_top_k",
        "chunk_size",
        "chunk_overlap",
    ):
        assert hasattr(settings, field), f"Settings is missing {field}"
