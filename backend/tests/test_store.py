from core import store


def test_distance_to_similarity_bounds():
    assert store.distance_to_similarity(0.0) == 1.0  # identical vectors
    assert store.distance_to_similarity(1.0) == 0.0  # orthogonal
    assert store.distance_to_similarity(2.0) == 0.0  # opposite, clamped
    assert store.distance_to_similarity(-0.01) == 1.0  # float noise, clamped


def test_distance_to_similarity_is_monotonic():
    """Smaller distance must always mean higher similarity."""
    a = store.distance_to_similarity(0.2)
    b = store.distance_to_similarity(0.6)
    assert a > b


def test_collection_name_is_session_scoped():
    assert store.collection_name("abc123") != store.collection_name("def456")


def test_collection_name_sanitises_input():
    name = store.collection_name("../../etc/passwd")
    assert "/" not in name
    assert "." not in name


def test_collection_name_falls_back_on_empty():
    assert store.collection_name("") == store.collection_name(None)
    assert store.collection_name("!!!") == store.collection_name(None)
