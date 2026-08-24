"""Node 3 - Generator. Produces a grounded answer from retrieved chunks."""

from openai import OpenAIError

from agent_state import AgentState
from config import settings
from core.nim_client import chat

FALLBACK_LOW_CONFIDENCE = (
    "I could not find reliable information about this in the document. "
    "Please try rephrasing or ask about a different topic."
)
FALLBACK_EMPTY = "I had trouble generating a response. Please try rephrasing."
FALLBACK_ERROR = "An error occurred. Please try again."

SYSTEM_TEMPLATE = """You are a precise document assistant powered by NVIDIA Nemotron.
Your answers must be:
- Direct and concise - state the answer immediately
- Factually grounded - only use information from the context below
- Well cited - cite the page number, e.g. "According to page 7, ..."
- Complete - include all relevant details from the context

If the answer is not in the context below, say exactly:
'I could not find this in the document.'

Document context:
{context}"""


def generator_node(state: AgentState) -> AgentState:
    # Threshold calibrated in eval/calibrate_threshold.py against the
    # measured gap between in-scope and out-of-scope top-1 scores.
    confidence = state["retrieval_confidence"]
    if confidence < settings.confidence_threshold:
        print(f"[Generator] Below threshold ({confidence:.3f}) - fallback")
        return {**state, "answer": FALLBACK_LOW_CONFIDENCE}

    chunks = state["retrieved_chunks"]
    pages = state.get("chunk_pages") or [0] * len(chunks)

    # Page numbers go into the prompt so the model cites something the user
    # can actually look up. "Chunk 3" was meaningless to a reader.
    context = "\n\n---\n\n".join(f"[Page {pages[i]}]\n{chunk}" for i, chunk in enumerate(chunks))

    messages = [{"role": "system", "content": SYSTEM_TEMPLATE.format(context=context)}]
    messages.extend(state["chat_history"][-settings.history_window :])
    messages.append({"role": "user", "content": state["question"]})

    try:
        answer = chat(messages, max_tokens=settings.max_tokens_answer)
    except OpenAIError as e:
        print(f"[Generator] API error: {e}")
        return {**state, "answer": FALLBACK_ERROR}

    if not answer:
        return {**state, "answer": FALLBACK_EMPTY}

    print(f"[Generator] Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}
