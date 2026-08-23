import pytest

from nodes.router import router_node


def _state(q: str) -> dict:
    return {
        "question": q,
        "retrieved_chunks": [],
        "answer": "",
        "decision": "",
        "iterations": 0,
        "chat_history": [],
        "retrieval_confidence": 0.0,
    }


def _route(q: str) -> str:
    return router_node(_state(q))["decision"]


def test_meta_keyword():
    assert _route("Summarize what you told me") == "meta"


def test_tldr_routes_to_meta():
    assert _route("Give me a tldr of our conversation") == "meta"


def test_normal_question():
    assert _route("What attention mechanism does Falcon use?") == "retrieve"


def test_short_question_goes_to_clarify():
    assert _route("Falcon?") == "clarify"


def test_explicit_clarify_keyword():
    assert _route("Can you explain that differently?") == "clarify"


# ── Known defects, fixed in Phase 4 ────────────────────────────

@pytest.mark.xfail(reason="'can you explain' keyword hijacks valid questions — Phase 4")
def test_explain_a_concept_should_retrieve():
    # A very natural way to ask a document question, but CLARIFY_KEYWORDS
    # contains "can you explain" so the agent asks the user to clarify
    # instead of searching.
    assert _route("Can you explain the multigroup attention mechanism?") == "retrieve"


@pytest.mark.xfail(reason="'summary' keyword misroutes to meta — Phase 4")
def test_asking_for_a_summary_of_document_content_should_retrieve():
    # META route uses chat history only, so this returns nothing useful
    # even though the answer is in the document.
    assert _route("What is the summary of the RefinedWeb filtering pipeline?") == "retrieve"


@pytest.mark.xfail(reason="len<10 rule blocks legitimate follow-ups — Phase 4")
def test_short_followup_should_retrieve():
    assert _route("And 7B?") == "retrieve"