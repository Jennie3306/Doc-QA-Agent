from agent_state import AgentState

META_KEYWORDS = ["summarize", "summary", "bullet", "recap",
                 "what did you", "told me", "tldr", "wrap up"]

CLARIFY_KEYWORDS = ["what do you mean", "can you explain",
                    "i don't understand", "clarify"]

# ⚠️ Known issues, fixed in Phase 4:
#   - "told me" misroutes "What did the paper tell me about X?" to meta
#   - len < 10 misroutes valid short questions like "RefinedWeb?"


def router_node(state: AgentState) -> AgentState:
    q = state["question"].lower()

    if any(k in q for k in META_KEYWORDS):
        decision = "meta"
    elif any(k in q for k in CLARIFY_KEYWORDS):
        decision = "clarify"
    elif len(state["question"].strip()) < 10:
        decision = "clarify"
    else:
        decision = "retrieve"

    print(f"[Router] Decision: {decision}")
    return {**state, "decision": decision}