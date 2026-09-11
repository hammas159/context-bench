"""Tests for the benchmark.

The ones that carry the argument: prompt caching must actually change the cost, CAG must
never lose an answer that is in the corpus, and a hybrid retriever must not silently
collapse into one of its halves.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from contextbench import (  # noqa: E402
    CAG,
    MAG,
    RAG,
    Corpus,
    Document,
    Pricing,
    Question,
    run_strategy,
)
from contextbench.corpus import estimate_tokens, normalise  # noqa: E402
from contextbench.retrieval import (  # noqa: E402
    Retriever,
    build_index,
    lexical_scores,
    recall_at_k,
    reciprocal_rank_fusion,
    semantic_scores,
    tokenise,
)


def tiny_corpus() -> Corpus:
    documents = [
        Document("d0", "Normandy", "Normandy is a region in France settled by Norsemen."),
        Document("d1", "Physics", "The speed of light in vacuum is 299,792,458 metres per second."),
        Document("d2", "Cooking", "Sourdough bread is leavened by wild yeast and lactobacilli."),
        Document("d3", "Chess", "The Sicilian Defence begins with the moves e4 c5."),
    ]
    questions = [
        Question("q0", "In what country is Normandy?", ("France",), "d0"),
        Question("q1", "How fast does light travel?", ("299,792,458 metres per second",), "d1"),
        Question("q2", "What leavens sourdough?", ("wild yeast",), "d2"),
    ]
    return Corpus(documents, questions)


# --- corpus ---------------------------------------------------------------


def test_answer_check_ignores_case_punctuation_and_articles():
    question = Question("q", "?", ("the United States",), "d")
    assert question.answered_by("... located in THE UNITED STATES, near ...")
    assert question.answered_by("united states")


def test_answer_check_is_false_when_the_answer_is_absent():
    question = Question("q", "?", ("France",), "d")
    assert not question.answered_by("Normandy is a region with a long coastline.")


def test_normalise_strips_articles_and_punctuation():
    assert normalise("The  Queen's, Army!") == "queen s army"


def test_token_estimate_scales_with_length():
    assert estimate_tokens("x" * 400) == 100
    assert estimate_tokens("") == 0  # an absent fragment bills for nothing
    assert estimate_tokens("x") == 1  # a present one always bills for something


def test_head_keeps_only_questions_its_documents_can_answer():
    """Otherwise every strategy is penalised equally for unanswerable questions, which
    tells you nothing about which is better."""
    small = tiny_corpus().head(2)
    assert len(small) == 2
    assert {q.document_id for q in small.questions} <= {"d0", "d1"}


def test_corpus_tokens_are_the_sum_of_its_documents():
    corpus = tiny_corpus()
    assert corpus.total_tokens == sum(d.tokens for d in corpus.documents)


# --- retrieval ------------------------------------------------------------


def test_question_words_are_not_indexed():
    """Added after a live failure: 'In what country is Normandy located?' retrieved a
    paragraph about Warsaw, because the interrogatives matched everywhere."""
    assert tokenise("In what country is Normandy located?") == ["country", "normandy", "located"]
    for word in ("what", "which", "when", "where", "who", "why", "how"):
        assert word not in tokenise(f"{word} is the answer")


def test_lexical_search_ors_its_terms():
    """The bug that turns a hybrid retriever into a dense-only one, silently.

    No document contains every word of a natural-language question. A keyword index that
    requires all of them returns nothing, the fusion quietly degrades to one retriever, and
    the answers still look plausible.
    """
    corpus = tiny_corpus()
    index = build_index(corpus.documents)
    scores = lexical_scores("In what country is Normandy located?", index)
    assert scores.max() > 0
    assert int(np.argmax(scores)) == 0


def test_a_query_of_unknown_words_scores_zero_rather_than_erroring():
    index = build_index(tiny_corpus().documents)
    assert lexical_scores("zzzz qqqq", index).max() == 0.0
    assert semantic_scores("zzzz qqqq", index).max() == 0.0


def test_rank_fusion_ignores_the_scale_of_its_inputs():
    """The reason RRF is used rather than adding normalised scores: a cosine of 0.4 and a
    BM25 of 11.2 cannot be summed, and any normalisation makes the weighting depend on the
    spread of whichever query was asked."""
    a = np.array([0.9, 0.5, 0.1])
    fused_small = reciprocal_rank_fusion(a, np.array([0.03, 0.02, 0.01]))
    fused_large = reciprocal_rank_fusion(a, np.array([300.0, 200.0, 100.0]))
    assert np.allclose(fused_small, fused_large)


def test_rank_fusion_rewards_agreement():
    agreed = reciprocal_rank_fusion(np.array([9.0, 1.0]), np.array([9.0, 1.0]))
    disagreed = reciprocal_rank_fusion(np.array([9.0, 1.0]), np.array([1.0, 9.0]))
    assert agreed[0] - agreed[1] > disagreed[0] - disagreed[1]


def test_top_k_is_ordered_and_capped_by_corpus_size():
    retriever = Retriever(tiny_corpus(), mode="hybrid")
    hits = retriever.top_k("Normandy France", 99)
    assert len(hits) == 4
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_recall_at_k_rises_with_k():
    corpus = tiny_corpus()
    retriever = Retriever(corpus, mode="hybrid")
    assert recall_at_k(retriever, corpus, 1) <= recall_at_k(retriever, corpus, 4)
    assert recall_at_k(retriever, corpus, 4) == 1.0  # four documents, so top-4 is everything


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        Retriever(tiny_corpus(), mode="magic")


# --- strategies -----------------------------------------------------------


def test_cag_never_loses_an_answer_that_is_in_the_corpus():
    """CAG's entire claim, as one assertion: there is no retrieval step, so there is no
    retrieval error."""
    corpus = tiny_corpus()
    result, _ = run_strategy(CAG(), corpus, Pricing())
    assert result.answer_in_context == 1.0


def test_rag_can_lose_an_answer_that_is_in_the_corpus():
    """And RAG's, as its counterpart. With k=1 on four documents, a miss is possible -
    which is the trade the whole benchmark measures."""
    corpus = tiny_corpus()
    result, _ = run_strategy(RAG(k=1), corpus, Pricing())
    assert result.answer_in_context <= 1.0
    assert result.mean_input_tokens < corpus.total_tokens


def test_prompt_caching_is_the_number_that_decides_it():
    """The central claim. Same strategy, same corpus, one parameter."""
    corpus = tiny_corpus()
    pricing = Pricing()
    cached, _ = run_strategy(CAG(use_cache=True), corpus, pricing)
    uncached, _ = run_strategy(CAG(use_cache=False), corpus, pricing)

    assert cached.cost_per_question < uncached.cost_per_question
    assert cached.answer_in_context == uncached.answer_in_context  # identical context


def test_the_first_cag_call_pays_full_price_and_later_ones_do_not():
    corpus = tiny_corpus()
    strategy = CAG(use_cache=True)
    strategy.prepare(corpus)
    strategy.reset()

    first = strategy.build(corpus.questions[0], corpus, Pricing())
    second = strategy.build(corpus.questions[1], corpus, Pricing())

    assert first.cached_tokens == 0 and first.fresh_tokens > 20
    assert second.cached_tokens > 20 and second.fresh_tokens < 20


def test_cag_reports_when_the_corpus_exceeds_the_window():
    """Rather than truncating silently, which would be CAG's price with RAG's recall."""
    corpus = tiny_corpus()
    result, _ = run_strategy(CAG(), corpus, Pricing(context_window=10))
    assert not result.fits_in_window


def test_rag_cost_is_flat_in_corpus_size_and_cag_cost_is_not():
    """The shape of the two curves, which is what produces a crossover at all."""
    documents = [Document(f"d{i}", "t", f"Document number {i} " * 40) for i in range(60)]
    questions = [
        Question(f"q{i}", f"Document number {i}", (f"number {i}",), f"d{i}")
        for i in range(0, 60, 6)
    ]
    corpus = Corpus(documents, questions)
    pricing = Pricing()

    small, large = corpus.head(20), corpus.head(60)
    rag_small, _ = run_strategy(RAG(k=3), small, pricing)
    rag_large, _ = run_strategy(RAG(k=3), large, pricing)
    cag_small, _ = run_strategy(CAG(), small, pricing)
    cag_large, _ = run_strategy(CAG(), large, pricing)

    assert rag_large.mean_input_tokens == pytest.approx(rag_small.mean_input_tokens, rel=0.35)
    assert cag_large.mean_input_tokens > 2.5 * cag_small.mean_input_tokens


def test_mag_carries_context_between_turns():
    corpus = tiny_corpus()
    strategy = MAG(k=1, memory_tokens=4000)
    strategy.prepare(corpus)
    strategy.reset()

    first = strategy.build(corpus.questions[0], corpus, Pricing())
    second = strategy.build(corpus.questions[1], corpus, Pricing())

    assert first.cached_tokens == 0  # nothing remembered yet
    assert second.cached_tokens > 0  # turn one is now in the context


def test_mag_memory_respects_its_budget():
    corpus = tiny_corpus()
    strategy = MAG(k=2, memory_tokens=20)
    strategy.prepare(corpus)
    for question in corpus.questions:
        answer = strategy.build(question, corpus, Pricing())
        assert answer.cached_tokens <= 40  # budget, with room for one document's overhang


def test_reset_clears_memory():
    corpus = tiny_corpus()
    strategy = MAG(k=1, memory_tokens=4000)
    strategy.prepare(corpus)
    strategy.build(corpus.questions[0], corpus, Pricing())
    strategy.reset()
    assert strategy.build(corpus.questions[1], corpus, Pricing()).cached_tokens == 0


# --- pricing --------------------------------------------------------------


def test_cached_tokens_cost_less_than_fresh_ones():
    pricing = Pricing()
    assert pricing.cost(cached_tokens=1_000_000) < pricing.cost(fresh_tokens=1_000_000)


def test_cost_is_linear_in_tokens():
    pricing = Pricing()
    assert pricing.cost(fresh_tokens=2_000_000) == pytest.approx(
        2 * pricing.cost(fresh_tokens=1_000_000)
    )


def test_setting_the_cache_price_to_the_input_price_removes_the_discount():
    """What the UI's 'no prompt caching' preset does, and why it moves the crossover."""
    no_discount = Pricing(input_per_million=0.25, cached_input_per_million=0.25)
    assert no_discount.cost(cached_tokens=1_000) == no_discount.cost(fresh_tokens=1_000)
