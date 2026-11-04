"""
Tests for the cross-encoder reranker (Day 4).

These tests use the real cross-encoder model to verify correct behavior.
The model is small (~80 MB) and runs on CPU, so tests are feasible
without a GPU, though slightly slower than pure unit tests.
"""

import pytest
from app.schemas import SearchResult
from app.rerank.reranker import rerank, reset_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_results() -> list[SearchResult]:
    """Create sample search results to rerank."""
    return [
        SearchResult(
            chunk_id="chunk_1",
            text="Apple reported total revenue of $383.3 billion in fiscal year 2023, a decrease from $394.3 billion in fiscal year 2022.",
            score=0.75,
            metadata={"ticker": "AAPL", "doc_type": "10-K"},
        ),
        SearchResult(
            chunk_id="chunk_2",
            text="The company sells iPhones, iPads, Macs, and wearables through its retail and online stores worldwide.",
            score=0.80,
            metadata={"ticker": "AAPL", "doc_type": "10-K"},
        ),
        SearchResult(
            chunk_id="chunk_3",
            text="Risk factors include supply chain disruptions, foreign currency fluctuations, and regulatory changes in key markets.",
            score=0.70,
            metadata={"ticker": "AAPL", "doc_type": "10-K"},
        ),
        SearchResult(
            chunk_id="chunk_4",
            text="Total net sales for Q3 2023 were $81.8 billion, compared to $83.0 billion in Q3 2022.",
            score=0.65,
            metadata={"ticker": "AAPL", "doc_type": "10-Q"},
        ),
        SearchResult(
            chunk_id="chunk_5",
            text="The board of directors declared a cash dividend of $0.24 per share.",
            score=0.60,
            metadata={"ticker": "AAPL", "doc_type": "10-K"},
        ),
    ]


# ---------------------------------------------------------------------------
# Core reranking tests
# ---------------------------------------------------------------------------

class TestRerank:
    """Tests for the rerank() function."""

    def test_rerank_returns_correct_count(self, sample_results: list[SearchResult]):
        """Rerank should return exactly top_k results."""
        reranked = rerank("What was Apple's total revenue?", sample_results, top_k=3)
        assert len(reranked) == 3

    def test_rerank_top_k_larger_than_input(self, sample_results: list[SearchResult]):
        """If top_k > len(results), return all results."""
        reranked = rerank("What was Apple's total revenue?", sample_results, top_k=100)
        assert len(reranked) == len(sample_results)

    def test_rerank_empty_results(self):
        """Reranking an empty list should return an empty list."""
        reranked = rerank("What was Apple's total revenue?", [], top_k=5)
        assert reranked == []

    def test_rerank_single_result(self):
        """Reranking a single result should return it unchanged."""
        single = [SearchResult(chunk_id="c1", text="Revenue was $100B.", score=0.5, metadata={})]
        reranked = rerank("What is the revenue?", single, top_k=5)
        assert len(reranked) == 1
        assert reranked[0].chunk_id == "c1"

    def test_rerank_scores_are_cross_encoder_scores(self, sample_results: list[SearchResult]):
        """After reranking, scores should be from the cross-encoder, not original retrieval."""
        reranked = rerank("What was Apple's total revenue?", sample_results, top_k=5)
        # Cross-encoder scores are typically different from the original scores
        # and are NOT cosine similarity (can be negative or > 1)
        original_scores = {r.chunk_id: r.score for r in sample_results}
        for r in reranked:
            # The score should be the cross-encoder score, which is also stored in metadata
            assert "rerank_score" in r.metadata
            assert r.score == r.metadata["rerank_score"]
            # Verify original score is preserved in metadata
            assert "retrieval_score" in r.metadata
            assert r.metadata["retrieval_score"] == original_scores[r.chunk_id]

    def test_rerank_results_are_sorted_descending(self, sample_results: list[SearchResult]):
        """Reranked results should be sorted by score in descending order."""
        reranked = rerank("What was Apple's total revenue?", sample_results, top_k=5)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_relevant_chunk_ranks_higher(self, sample_results: list[SearchResult]):
        """The revenue chunk should rank higher than the dividend chunk for a revenue query."""
        reranked = rerank(
            "What was Apple's total revenue in fiscal year 2023?",
            sample_results,
            top_k=5,
        )
        # chunk_1 directly answers the question — it should be in the top 2
        top_ids = [r.chunk_id for r in reranked[:2]]
        assert "chunk_1" in top_ids, (
            f"Revenue chunk should rank in top 2 for a revenue query, got: {top_ids}"
        )

    def test_rerank_preserves_metadata(self, sample_results: list[SearchResult]):
        """Reranking should preserve original metadata fields."""
        reranked = rerank("What was Apple's total revenue?", sample_results, top_k=5)
        for r in reranked:
            assert "ticker" in r.metadata
            assert r.metadata["ticker"] == "AAPL"

    def test_rerank_all_chunk_ids_preserved(self, sample_results: list[SearchResult]):
        """All original chunk IDs should appear in full reranked output."""
        reranked = rerank("revenue", sample_results, top_k=len(sample_results))
        reranked_ids = {r.chunk_id for r in reranked}
        original_ids = {r.chunk_id for r in sample_results}
        assert reranked_ids == original_ids

    def test_rerank_different_queries_different_order(self, sample_results: list[SearchResult]):
        """Different queries should produce different rankings."""
        revenue_reranked = rerank("What was total revenue?", sample_results, top_k=5)
        risk_reranked = rerank("What are the key risk factors?", sample_results, top_k=5)
        revenue_order = [r.chunk_id for r in revenue_reranked]
        risk_order = [r.chunk_id for r in risk_reranked]
        # The risk query should rank the risk chunk higher
        assert revenue_order != risk_order, "Different queries should produce different orderings"

    def test_rerank_risk_query_ranks_risk_chunk_higher(self, sample_results: list[SearchResult]):
        """A risk-related query should rank the risk chunk in the top positions."""
        reranked = rerank(
            "What supply chain risks does Apple face?",
            sample_results,
            top_k=5,
        )
        top_ids = [r.chunk_id for r in reranked[:2]]
        assert "chunk_3" in top_ids, (
            f"Risk chunk should rank in top 2 for a risk query, got: {top_ids}"
        )

    def test_rerank_top_k_zero_returns_empty(self, sample_results: list[SearchResult]):
        """top_k=0 is degenerate but should return empty list gracefully."""
        # top_k=0 means "return nothing"
        reranked = rerank("revenue", sample_results, top_k=0)
        assert reranked == []


class TestResetModel:
    """Tests for the reset_model function."""

    def test_reset_model_does_not_raise(self):
        """reset_model() should not raise even if model was never loaded."""
        reset_model()  # Should not raise

