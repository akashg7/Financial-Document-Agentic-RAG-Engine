"""
Tests for Semantic Caching and Cache Lineage / Invalidation (Day 8).

Tests:
1. CachedAnswer model serialization and conversion to AnswerResponse.
2. L1 Exact string match hit, miss, and whitespace/case normalization.
3. L2 Semantic similarity match for paraphrased financial queries.
4. L2 Semantic boundary rejection for low-similarity or cross-domain queries.
5. Partition scope isolation (ticker / document type separation).
6. Lineage registration: forward and reverse index integrity.
7. Surgical chunk-level cache invalidation (evicts only dependent queries).
8. Document-level invalidation (evicts all queries dependent on doc or doc's chunks).
9. Impact analysis via get_affected_queries.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.schemas import AnswerResponse, Citation
from app.cache.models import CachedAnswer, CacheStats
from app.cache import backend
from app.cache.semantic_cache import (
    SemanticCache,
    normalize_query,
    compute_query_hash,
    make_scope_key,
    cosine_similarity,
)
from app.cache.lineage import LineageManager, get_lineage_manager


@pytest.fixture(autouse=True)
def clean_cache():
    """Ensure clean cache storage before and after each test."""
    backend.force_memory_fallback()
    backend.reset_cache_backend()
    yield
    backend.reset_cache_backend()


# ============================================================================
# 1. Models & Helper Tests
# ============================================================================

def test_cached_answer_conversion():
    """Verify CachedAnswer converts accurately to AnswerResponse with cached=True."""
    entry = CachedAnswer(
        cache_key="cache_test123",
        query="What was Apple's total revenue in 2023?",
        query_embedding=[0.1, 0.2, 0.3],
        answer="Apple net sales were $383,285 million in 2023 [Chunk: AAPL_001].",
        citations=[Citation(chunk_id="AAPL_001", doc_id="AAPL_10-K_2023", quote="Apple net sales")],
        confidence=0.88,
        decision="generate",
        ticker_filter="AAPL",
        doc_type_filter="10-K",
        dependent_chunk_ids=["AAPL_001"],
        dependent_doc_ids=["AAPL_10-K_2023"],
        hit_count=3,
    )
    resp = entry.to_answer_response(query="What was Apple's total revenue in 2023?", latency_ms=0.45)
    assert resp.cached is True
    assert resp.confidence == 0.88
    assert resp.retries == 0
    assert len(resp.citations) == 1
    assert resp.latency_ms == 0.45


def test_query_normalization():
    """Verify whitespace stripping and case folding."""
    q1 = "   What was Apple's REVENUE in 2023?   "
    q2 = "what was apple's revenue in 2023?"
    assert normalize_query(q1) == normalize_query(q2)
    assert compute_query_hash(normalize_query(q1)) == compute_query_hash(normalize_query(q2))


def test_scope_key_generation():
    """Verify scope keys handle ticker and doc_type cases."""
    assert make_scope_key("AAPL", "10-K") == "AAPL:10-K"
    assert make_scope_key("aapl", "10-k") == "AAPL:10-K"
    assert make_scope_key(None, "10-K") == "ALL:10-K"
    assert make_scope_key(None, None) == "ALL:ALL"


def test_cosine_similarity_math():
    """Verify mathematical properties of cosine similarity."""
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    v3 = [0.0, 1.0, 0.0]
    assert pytest.approx(cosine_similarity(v1, v2), 0.001) == 1.0
    assert pytest.approx(cosine_similarity(v1, v3), 0.001) == 0.0
    assert cosine_similarity([], v1) == 0.0


# ============================================================================
# 2. L1 Exact & L2 Semantic Cache Tests
# ============================================================================

def test_l1_exact_match_hit():
    """Exact query string produces an L1 cache hit with zero semantic search overhead."""
    cache = SemanticCache()

    mock_answer = AnswerResponse(
        query="What was Apple's revenue in 2023?",
        answer="Apple reported $383,285 million in net sales.",
        citations=[Citation(chunk_id="AAPL_001", doc_id="AAPL_10-K_2023", quote="Apple reported")],
        confidence=0.89,
        decision="generate",
    )

    # 1. Set cache entry
    try:
        cache_key = cache.set(
            query="What was Apple's revenue in 2023?",
            answer_response=mock_answer,
            ticker_filter="AAPL",
            doc_type_filter="10-K",
        )
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    assert cache_key is not None

    # 2. Lookup exact query (varying casing and leading spaces)
    entry, hit_type = cache.get(
        query="  WHAT WAS APPLE'S REVENUE IN 2023?  ",
        ticker_filter="AAPL",
        doc_type_filter="10-K",
    )
    assert hit_type == "exact"
    assert entry is not None
    assert entry.confidence == 0.89
    assert entry.dependent_chunk_ids == ["AAPL_001"]


def test_l2_semantic_similarity_hit():
    """Semantically equivalent query hits L2 semantic cache above similarity threshold."""
    cache = SemanticCache(default_similarity_threshold=0.85)

    mock_answer = AnswerResponse(
        query="Apple total net sales fiscal year 2023",
        answer="Apple net sales were $383.3 billion.",
        citations=[Citation(chunk_id="AAPL_001", doc_id="AAPL_10-K_2023", quote="Apple net sales")],
        confidence=0.91,
        decision="generate",
    )

    try:
        cache.set(
            query="Apple total net sales fiscal year 2023",
            answer_response=mock_answer,
            ticker_filter="AAPL",
        )
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    # Paraphrased query
    entry, hit_type = cache.get(
        query="What was Apple total net sales in fiscal 2023?",
        ticker_filter="AAPL",
        similarity_threshold=0.88,
    )
    assert hit_type == "semantic"
    assert entry is not None
    assert "383.3 billion" in entry.answer


def test_l2_semantic_cache_miss_on_different_topic():
    """Unrelated query produces a cache miss."""
    cache = SemanticCache(default_similarity_threshold=0.90)

    mock_answer = AnswerResponse(
        query="Apple total net sales fiscal year 2023",
        answer="Apple net sales were $383.3 billion.",
        citations=[Citation(chunk_id="AAPL_001", doc_id="AAPL_10-K_2023", quote="Apple net sales")],
        confidence=0.91,
        decision="generate",
    )

    try:
        cache.set(
            query="Apple total net sales fiscal year 2023",
            answer_response=mock_answer,
            ticker_filter="AAPL",
        )
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    entry, hit_type = cache.get(
        query="What were Tesla vehicle deliveries in 2023?",
        ticker_filter="AAPL",
    )
    assert hit_type == "miss"
    assert entry is None


def test_scope_partition_isolation():
    """Queries for AAPL do not match cached entries for MSFT even if identical wording."""
    cache = SemanticCache()

    mock_answer = AnswerResponse(
        query="What was total revenue?",
        answer="Microsoft revenue was $211.9 billion.",
        citations=[Citation(chunk_id="MSFT_001", doc_id="MSFT_10-K_2023", quote="Microsoft revenue")],
        confidence=0.88,
        decision="generate",
    )

    try:
        cache.set(
            query="What was total revenue?",
            answer_response=mock_answer,
            ticker_filter="MSFT",
        )
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    entry, hit_type = cache.get(
        query="What was total revenue?",
        ticker_filter="AAPL",  # Different ticker scope!
    )
    assert hit_type == "miss"
    assert entry is None


# ============================================================================
# 3. Lineage & Cache Invalidation Tests (Hand-Write Piece #4)
# ============================================================================

def test_lineage_registration_and_retrieval():
    """Verify LineageManager stores forward and reverse dependency edges."""
    mgr = LineageManager()

    try:
        mgr.register_lineage(
            cache_key="cache_001",
            chunk_ids=["chunk_a", "chunk_b"],
            doc_ids=["doc_10k_2023"],
        )
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    lineage = mgr.get_lineage("cache_001")
    assert "chunk_a" in lineage["chunk_ids"]
    assert "chunk_b" in lineage["chunk_ids"]
    assert "doc_10k_2023" in lineage["doc_ids"]


def test_surgical_chunk_invalidation():
    """Invalidating a specific chunk only evicts cache entries that cited it."""
    cache = SemanticCache()
    mgr = LineageManager()

    try:
        # Query 1 depends on chunk_revenue
        ans1 = AnswerResponse(
            query="Apple revenue 2023",
            answer="$383B",
            citations=[Citation(chunk_id="AAPL_chunk_revenue", doc_id="AAPL_10-K_2023", quote="rev")],
            confidence=0.9,
            decision="generate",
        )
        k1 = cache.set("Apple revenue 2023", ans1, ticker_filter="AAPL")

        # Query 2 depends on chunk_debt
        ans2 = AnswerResponse(
            query="Apple debt 2023",
            answer="$105B",
            citations=[Citation(chunk_id="AAPL_chunk_debt", doc_id="AAPL_10-K_2023", quote="debt")],
            confidence=0.9,
            decision="generate",
        )
        k2 = cache.set("Apple debt 2023", ans2, ticker_filter="AAPL")

        # Invalidate ONLY chunk_revenue
        evicted = mgr.invalidate_by_chunk_ids(["AAPL_chunk_revenue"])
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    assert k1 in evicted
    assert k2 not in evicted

    # Verify Query 1 is now a cache miss
    e1, hit1 = cache.get("Apple revenue 2023", ticker_filter="AAPL")
    assert hit1 == "miss"

    # Verify Query 2 is still a cache hit
    e2, hit2 = cache.get("Apple debt 2023", ticker_filter="AAPL")
    assert hit2 == "exact"


def test_document_level_invalidation():
    """Invalidating a document evicts all queries that cited any of its constituent chunks."""
    cache = SemanticCache()
    mgr = LineageManager()

    try:
        # Register doc chunks
        mgr.register_document_chunks("AAPL_10-K_2023", ["chunk_rev", "chunk_tax", "chunk_rnd"])

        # Query citing chunk_rev
        ans1 = AnswerResponse(
            query="Apple sales 2023",
            answer="Sales 383B",
            citations=[Citation(chunk_id="chunk_rev", doc_id="AAPL_10-K_2023", quote="sales")],
            confidence=0.9,
            decision="generate",
        )
        k1 = cache.set("Apple sales 2023", ans1, ticker_filter="AAPL")

        # Query citing chunk_tax
        ans2 = AnswerResponse(
            query="Apple taxes 2023",
            answer="Taxes 16B",
            citations=[Citation(chunk_id="chunk_tax", doc_id="AAPL_10-K_2023", quote="tax")],
            confidence=0.9,
            decision="generate",
        )
        k2 = cache.set("Apple taxes 2023", ans2, ticker_filter="AAPL")

        # Invalidate the entire document
        evicted = mgr.invalidate_by_document("AAPL_10-K_2023")
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    assert k1 in evicted
    assert k2 in evicted

    # Both queries must now MISS
    _, hit1 = cache.get("Apple sales 2023", ticker_filter="AAPL")
    _, hit2 = cache.get("Apple taxes 2023", ticker_filter="AAPL")
    assert hit1 == "miss"
    assert hit2 == "miss"


def test_impact_analysis_affected_queries():
    """Verify get_affected_queries reports blast radius without evicting entries."""
    cache = SemanticCache()
    mgr = LineageManager()

    try:
        ans = AnswerResponse(
            query="What was Apple R&D spend in 2023?",
            answer="R&D was $29.9B",
            citations=[Citation(chunk_id="chunk_rnd_01", doc_id="AAPL_10-K_2023", quote="rnd")],
            confidence=0.9,
            decision="generate",
        )
        cache.set("What was Apple R&D spend in 2023?", ans, ticker_filter="AAPL")

        affected = mgr.get_affected_queries(chunk_id="chunk_rnd_01")
    except NotImplementedError:
        pytest.skip("LineageManager not yet implemented in app/cache/lineage.py")

    assert "What was Apple R&D spend in 2023?" in affected

    # Entry should still exist (inspection only)
    e, hit = cache.get("What was Apple R&D spend in 2023?", ticker_filter="AAPL")
    assert hit == "exact"

