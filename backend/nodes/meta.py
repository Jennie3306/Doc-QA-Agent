from openai import OpenAIError

from agent_state import AgentState
from config import settings
from core.nim_client import chat

NO_HISTORY = "We haven't discussed anything yet. Please ask me a question about the document first!"

SYSTEM_PROMPT = """You are a helpful assistant.
The user wants you to summarize or reflect on the conversation so far.
Use ONLY the conversation history to answer — do not add new information."""


def meta_node(state: AgentState) -> AgentState:
    chat_history = state["chat_history"]

    if not chat_history:
        return {**state, "answer": NO_HISTORY}

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(chat_history[-settings.history_window_meta :])
    messages.append({"role": "user", "content": state["question"]})

    try:
        answer = chat(messages, max_tokens=settings.max_tokens_meta)
    except OpenAIError as e:
        print(f"[Meta] API error: {e}")
        return {**state, "answer": "I had trouble summarizing."}

    print("[Meta] Summary generated")
    return {**state, "answer": answer or "I had trouble summarizing."}
