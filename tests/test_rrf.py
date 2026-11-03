"""
Tests for Reciprocal Rank Fusion.

These tests define the CORRECT behavior that your implementation must match.
Run with: pytest tests/test_rrf.py -v

Test categories:
1. Basic fusion of two ranked lists
2. Single list (degenerates to re-scoring)
3. Empty inputs
4. Disjoint lists (no overlap)
5. Full overlap (same results in both lists)
6. Top-n truncation
7. k parameter effect
8. Duplicate chunk_ids within a single list
9. Score correctness (exact values)
10. Preserves text and metadata from best-ranking list
"""

import pytest
from app.schemas import SearchResult
from app.retrieval.rrf import reciprocal_rank_fusion


def _make_result(chunk_id: str, score: float = 0.0, text: str = "", metadata: dict = None) -> SearchResult:
    """Helper to create a SearchResult for testing."""
    return SearchResult(
        chunk_id=chunk_id,
        text=text or f"Text for {chunk_id}",
        score=score,
        metadata=metadata or {"source": "test"},
    )


class TestRRFBasicFusion:
    """Test the core RRF fusion logic."""

    def test_two_lists_with_overlap(self):
        """
        When a chunk appears in both lists, its RRF score should be the SUM
        of its reciprocal rank contributions from each list.

        Dense:  [A, B, C]  → A gets 1/(60+1), B gets 1/(60+2), C gets 1/(60+3)
        Sparse: [B, D, A]  → B gets 1/(60+1), D gets 1/(60+2), A gets 1/(60+3)

        B appears in both → B_score = 1/61 + 1/61 > A_score = 1/61 + 1/63
        So B should rank first.
        """
        dense = [_make_result("A", 0.9), _make_result("B", 0.7), _make_result("C", 0.5)]
        sparse = [_make_result("B", 15.0), _make_result("D", 8.0), _make_result("A", 3.0)]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        # B should rank first (appears at rank 1 in sparse, rank 2 in dense)
        # A should rank second (appears at rank 1 in dense, rank 3 in sparse)
        assert fused[0].chunk_id == "B"
        assert fused[1].chunk_id == "A"

        # All 4 unique chunks should be in the result
        result_ids = [r.chunk_id for r in fused]
        assert set(result_ids) == {"A", "B", "C", "D"}

    def test_three_lists(self):
        """RRF should work with more than two ranked lists."""
        list1 = [_make_result("A"), _make_result("B")]
        list2 = [_make_result("B"), _make_result("C")]
        list3 = [_make_result("A"), _make_result("C")]

        fused = reciprocal_rank_fusion([list1, list2, list3], k=60)

        # A appears in list1 (rank 1) and list3 (rank 1): 2/(60+1)
        # B appears in list1 (rank 2) and list2 (rank 1): 1/(60+2) + 1/(60+1)
        # C appears in list2 (rank 2) and list3 (rank 2): 2/(60+2)
        # A_score = 2/61 ≈ 0.03279
        # B_score = 1/62 + 1/61 ≈ 0.03252
        # C_score = 2/62 ≈ 0.03226
        assert fused[0].chunk_id == "A"
        assert fused[1].chunk_id == "B"
        assert fused[2].chunk_id == "C"


class TestRRFEdgeCases:
    """Test edge cases and degenerate inputs."""

    def test_empty_input(self):
        """Empty ranked_lists should return empty results."""
        assert reciprocal_rank_fusion([]) == []

    def test_empty_inner_lists(self):
        """All empty inner lists should return empty results."""
        assert reciprocal_rank_fusion([[], []]) == []

    def test_single_list(self):
        """A single list should just re-score with RRF weights."""
        results = [_make_result("A", 0.9), _make_result("B", 0.5)]
        fused = reciprocal_rank_fusion([results], k=60)

        assert len(fused) == 2
        assert fused[0].chunk_id == "A"
        assert fused[1].chunk_id == "B"
        # Scores should be RRF weights
        assert abs(fused[0].score - 1 / 61) < 1e-6
        assert abs(fused[1].score - 1 / 62) < 1e-6

    def test_disjoint_lists(self):
        """Lists with no overlapping results — all results should appear."""
        dense = [_make_result("A"), _make_result("B")]
        sparse = [_make_result("C"), _make_result("D")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        assert len(fused) == 4
        result_ids = {r.chunk_id for r in fused}
        assert result_ids == {"A", "B", "C", "D"}

        # A and C both at rank 1 in their lists → tied scores
        # B and D both at rank 2 → tied scores
        assert abs(fused[0].score - fused[1].score) < 1e-6  # A and C tied
        assert abs(fused[2].score - fused[3].score) < 1e-6  # B and D tied

    def test_full_overlap_same_order(self):
        """Identical lists — RRF scores should be doubled but order preserved."""
        list1 = [_make_result("A"), _make_result("B"), _make_result("C")]
        list2 = [_make_result("A"), _make_result("B"), _make_result("C")]

        fused = reciprocal_rank_fusion([list1, list2], k=60)

        assert [r.chunk_id for r in fused] == ["A", "B", "C"]
        # Each score should be 2 * 1/(60+rank)
        assert abs(fused[0].score - 2 / 61) < 1e-6
        assert abs(fused[1].score - 2 / 62) < 1e-6

    def test_full_overlap_reversed_order(self):
        """Same chunks, reversed order — middle-ranked chunk might win."""
        list1 = [_make_result("A"), _make_result("B"), _make_result("C")]
        list2 = [_make_result("C"), _make_result("B"), _make_result("A")]

        fused = reciprocal_rank_fusion([list1, list2], k=60)

        # B is rank 2 in both → 2/(60+2) = 2/62
        # A is rank 1 + rank 3 → 1/61 + 1/63
        # C is rank 3 + rank 1 → 1/63 + 1/61
        # A and C should tie, both > B? Let's check:
        # A = C = 1/61 + 1/63 ≈ 0.03226
        # B = 2/62 ≈ 0.03226
        # Actually A = C ≈ 0.032268 and B ≈ 0.032258 so A and C slightly beat B!
        result_ids = [r.chunk_id for r in fused]
        assert result_ids[2] == "B"  # B should be last


class TestRRFTopN:
    """Test top_n truncation."""

    def test_top_n_limits_output(self):
        """Should return at most top_n results."""
        results = [_make_result(f"chunk_{i}") for i in range(10)]

        fused = reciprocal_rank_fusion([results], k=60, top_n=3)
        assert len(fused) == 3

    def test_top_n_larger_than_results(self):
        """top_n larger than available results should return all results."""
        results = [_make_result("A"), _make_result("B")]

        fused = reciprocal_rank_fusion([results], k=60, top_n=100)
        assert len(fused) == 2


class TestRRFKParameter:
    """Test the effect of the k smoothing parameter."""

    def test_small_k_amplifies_top_ranks(self):
        """With k=1, rank 1 gets 1/2 and rank 2 gets 1/3 — huge gap."""
        results = [_make_result("A"), _make_result("B")]

        fused = reciprocal_rank_fusion([results], k=1)

        # k=1: A gets 1/(1+1)=0.5, B gets 1/(1+2)=0.333
        ratio_k1 = fused[0].score / fused[1].score
        assert ratio_k1 > 1.4  # 0.5/0.333 ≈ 1.5

    def test_large_k_flattens_scores(self):
        """With k=1000, rank differences barely matter."""
        results = [_make_result("A"), _make_result("B")]

        fused = reciprocal_rank_fusion([results], k=1000)

        # k=1000: A gets 1/1001, B gets 1/1002
        ratio_k1000 = fused[0].score / fused[1].score
        assert ratio_k1000 < 1.01  # Scores nearly equal


class TestRRFScoreCorrectness:
    """Test exact RRF score computation."""

    def test_exact_scores_two_lists(self):
        """Verify precise RRF score computation."""
        dense = [_make_result("A"), _make_result("B")]
        sparse = [_make_result("B"), _make_result("A")]

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        # Both A and B appear in both lists
        # A: rank 1 in dense + rank 2 in sparse = 1/61 + 1/62
        # B: rank 2 in dense + rank 1 in sparse = 1/62 + 1/61
        # They should have EQUAL scores (same RRF contribution)
        expected = 1 / 61 + 1 / 62
        assert abs(fused[0].score - expected) < 1e-10
        assert abs(fused[1].score - expected) < 1e-10


class TestRRFMetadataPreservation:
    """Test that metadata and text from the best-ranking list are preserved."""

    def test_preserves_best_rank_metadata(self):
        """Should use the SearchResult from the list where chunk ranked highest."""
        dense_a = _make_result("A", text="Dense text for A", metadata={"source": "dense"})
        sparse_a = _make_result("A", text="Sparse text for A", metadata={"source": "sparse"})

        dense = [dense_a, _make_result("B")]
        sparse = [_make_result("C"), sparse_a]  # A is rank 2 in sparse

        fused = reciprocal_rank_fusion([dense, sparse], k=60)

        # A ranked #1 in dense, #2 in sparse → should use dense's metadata
        a_result = next(r for r in fused if r.chunk_id == "A")
        assert a_result.text == "Dense text for A"
        assert a_result.metadata["source"] == "dense"


class TestRRFDuplicatesWithinList:
    """Test handling of duplicate chunk_ids within a single list."""

    def test_deduplicates_within_list(self):
        """If same chunk_id appears twice in one list, only first occurrence counts."""
        results = [
            _make_result("A"),
            _make_result("B"),
            _make_result("A"),  # Duplicate!
        ]

        fused = reciprocal_rank_fusion([results], k=60)

        # A should appear only once
        a_results = [r for r in fused if r.chunk_id == "A"]
        assert len(a_results) == 1
        # A's score should be from rank 1 only, NOT rank 1 + rank 3
        assert abs(a_results[0].score - 1 / 61) < 1e-6

