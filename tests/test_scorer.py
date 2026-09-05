from tests.conftest import make_turn


def test_empty_turns_returns_empty_dict(scorer):
    assert scorer.score([], "any query") == {}


def test_scores_are_bounded_and_keyed_by_turn_id(scorer):
    turns = [
        make_turn(1, "The weather in Paris is lovely in spring."),
        make_turn(2, "My order #4521 never arrived."),
        make_turn(3, "Can you recommend a good pizza place?"),
    ]
    scores = scorer.score(turns, "Where is my order?")

    assert set(scores.keys()) == {1, 2, 3}
    for s in scores.values():
        assert 0.0 <= s <= 1.0


def test_lexical_overlap_is_rewarded(scorer):
    """A turn sharing exact words with the query should score higher
    than an unrelated turn with no shared vocabulary or meaning."""
    turns = [
        make_turn(1, "The invoice number for your purchase is INV-9981."),
        make_turn(2, "I enjoy hiking on weekends near the coast."),
    ]
    scores = scorer.score(turns, "What is my invoice number?")
    assert scores[1] > scores[2]


def test_identical_relevance_batch_does_not_crash(scorer):
    """Edge case for the min-max normalizer: if every turn scores
    identically, normalization must not divide by zero."""
    turns = [make_turn(1, "hello"), make_turn(2, "hello")]
    scores = scorer.score(turns, "hello")
    assert all(0.0 <= s <= 1.0 for s in scores.values())


def test_embeddings_are_cached_on_turn(scorer):
    turn = make_turn(1, "Some content to embed.")
    assert turn.embedding is None
    scorer.score([turn], "a query")
    assert turn.embedding is not None

    cached = turn.embedding
    scorer.score([turn], "a different query")
    assert turn.embedding == cached  # not recomputed
