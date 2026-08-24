from openai import OpenAIError

from agent_state import AgentState
from config import settings
from core.nim_client import chat

FALLBACK_LOW_CONFIDENCE = (
    "I could not find reliable information about this in the document. "
    "Please try rephrasing or ask about a different topic."
)


def generator_node(state: AgentState) -> AgentState:
    chunks = state["retrieved_chunks"]
    confidence = state["retrieval_confidence"]

    # Threshold calibrated in eval/calibrate_threshold.py against the
    # measured gap between in-scope and out-of-scope top-1 scores.
    if confidence < settings.confidence_threshold:
        print(f"[Generator] Below threshold ({confidence:.3f}) - fallback")
        return {**state, "answer": FALLBACK_LOW_CONFIDENCE}

    context = "\n\n---\n\n".join(
        f"Chunk {i+1}:\n{chunk}" for i, chunk in enumerate(chunks)
    )

    system_prompt = f"""You are a precise document assistant powered by NVIDIA Nemotron.
Your answers must be:
- Direct and concise — state the answer immediately
- Factually grounded — only use information from the context below
- Well cited — always mention which Chunk your answer comes from
- Complete — include all relevant details from the context

If the answer is not in the context below, say exactly:
'I could not find this in the document.'

Document context:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["chat_history"][-settings.history_window:])
    messages.append({"role": "user", "content": state["question"]})

    try:
        answer = chat(messages, max_tokens=settings.max_tokens_answer)
    except OpenAIError as e:
        print(f"[Generator] API error: {e}")
        return {**state, "answer": "An error occurred. Please try again."}

    if not answer:
        return {
            **state,
            "answer": "I had trouble generating a response. Please try rephrasing.",
        }

    print(f"[Generator] Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}