"""Single wrapper around the NVIDIA NIM API."""

from openai import OpenAI

from config import settings

_client = OpenAI(
    base_url=settings.nvidia_base_url,
    api_key=settings.nvidia_api_key,
)


def embed(text: str, input_type: str) -> list[float]:
    """Embed one text. `input_type` is 'query' or 'passage'.

    Required, not defaulted: nv-embedqa-e5-v5 is asymmetric, and silently
    using "passage" where "query" was meant degrades retrieval without
    raising anything.
    """
    return embed_batch([text], input_type)[0]


def embed_batch(texts: list[str], input_type: str) -> list[list[float]]:
    """Embed many texts, batched.

    Two details that are easy to get wrong:

    1. `sorted(resp.data, key=lambda d: d.index)` - the API does not
       guarantee response order. Without the sort, chunk 3's embedding can
       end up attached to chunk 7. That silently wrecks retrieval and
       crashes nothing.

    2. truncate="END" rather than "NONE" - NONE raises when a chunk exceeds
       the model's token limit, which kills the whole ingest run partway
       through. END clips the tail instead.
    """
    out: list[list[float]] = []

    for i in range(0, len(texts), settings.embed_batch_size):
        batch = texts[i : i + settings.embed_batch_size]
        resp = _client.embeddings.create(
            model=settings.embed_model,
            input=batch,
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "END"},
        )
        ordered = sorted(resp.data, key=lambda d: d.index)
        out.extend(d.embedding for d in ordered)

    if len(out) != len(texts):
        raise RuntimeError(f"Embedding count mismatch: got {len(out)} for {len(texts)} texts")

    return out


def chat(
    messages: list[dict],
    max_tokens: int,
    temperature: float | None = None,
) -> str:
    """Call the LLM. Always returns a string, never None."""
    kwargs: dict = {
        "model": settings.llm_model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if temperature is not None:
        kwargs["temperature"] = temperature

    resp = _client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    return content if content and content.strip() else ""
