from agent_state import AgentState


def clarifier_node(state: AgentState) -> AgentState:
    answer = (
        f"Your question '{state['question']}' seems a bit broad. "
        f"Could you be more specific? For example:\n"
        f"- What specific aspect are you asking about?\n"
        f"- Are you referring to a particular model, technique, or result?\n"
        f"- What context are you looking for?"
    )

    print("[Clarifier] Asking for clarification")
    return {**state, "answer": answer}
