from agent_graph import build_agent, initial_state, route_decision


def test_graph_compiles():
    assert build_agent() is not None


def test_initial_state_has_all_keys():
    from agent_state import AgentState

    state = initial_state("test question", [])
    assert set(state.keys()) == set(AgentState.__annotations__.keys())


def test_route_decision_mapping():
    assert route_decision({"decision": "meta"}) == "meta"
    assert route_decision({"decision": "clarify"}) == "clarifier"
    assert route_decision({"decision": "retrieve"}) == "retriever"
    assert route_decision({"decision": "garbage"}) == "retriever"  # fallback
