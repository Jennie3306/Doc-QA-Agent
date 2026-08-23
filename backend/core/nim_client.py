"""Thin wrapper around the NVIDIA NIM API."""
from openai import OpenAI

from config import settings

_client = OpenAI(
    base_url=settings.nvidia_base_url,
    api_key=settings.nvidia_api_key,
)


def embed(text: str, input_type: str) -> list[float]:
    """Embed a single text. input_type is 'query' or 'passage'.

    NOTE: batching is introduced in Phase 1 — kept as a single-item
    call here so Phase 0 changes no behaviour.
    """
    resp = _client.embeddings.create(
        model=settings.embed_model,
        input=text,
        encoding_format="float",
        extra_body={"input_type": input_type, "truncate": "NONE"},
    )
    return resp.data[0].embedding


def chat(messages: list[dict], max_tokens: int, temperature: float | None = None) -> str:
    """Call the LLM and return text. Returns '' instead of None on empty output."""
    kwargs = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = _client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    return content if content and content.strip() else ""