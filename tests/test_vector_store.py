"""
Tests for the Qdrant vector store operations.

These tests use an in-memory Qdrant client (no Docker needed) to verify:
1. Collection initialization is idempotent
2. Chunk upsert creates correct points with payloads and named vectors (dense + sparse)
3. Dense search returns ranked results
4. Sparse (BM25) search returns keyword-relevant results
5. Payload filters (ticker, doc_type) work correctly for both dense and sparse
6. get_existing_hashes retrieves all content hashes
7. Deterministic point ID generation

WHY in-memory Qdrant instead of mocking:
- Qdrant provides a ':memory:' mode that runs the full engine in-process.
  This gives us realistic behavior (actual vector search, actual filtering)
  without Docker. Mocking would only test our code, not the integration.
"""

import pytest
import numpy as np

from qdrant_client import QdrantClient

from app.schemas import Chunk, ChunkMetadata, DocumentType
from app.retrieval.vector_store import (
    init_collection,
    upsert_chunks,
    search_dense,
    search_sparse,
    get_existing_hashes,
    set_qdrant_client,
    get_qdrant_client,
    _chunk_id_to_int,
)
from app.retrieval.embedder import embed_texts
from app.retrieval.sparse import fit_bm25, encode_sparse_document, reset_bm25


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def in_memory_qdrant():
    """
    Replace the Qdrant client with an in-memory instance for each test.
    This ensures test isolation — each test gets a fresh, empty Qdrant.
    """
    client = QdrantClient(":memory:")
    set_qdrant_client(client)
    yield client
    # Cleanup: reset the client after test
    set_qdrant_client(None)
    reset_bm25()


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    """Create a set of realistic test chunks with different tickers."""
    chunks = []
    test_data = [
        ("AAPL", "Apple reported revenue of $383 billion for fiscal year 2023."),
        ("AAPL", "iPhone segment generated $200.6 billion in revenue."),
        ("AAPL", "Services revenue reached $85.2 billion, growing 9% year over year."),
        ("MSFT", "Microsoft reported revenue of $211 billion for fiscal year 2023."),
        ("MSFT", "Azure cloud revenue grew 29% year over year."),
        ("GOOGL", "Alphabet reported revenue of $307 billion for fiscal year 2023."),
    ]

    for i, (ticker, text) in enumerate(test_data):
        import hashlib
        content_hash = hashlib.sha256(text.encode()).hexdigest()
        chunk = Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id=f"{ticker}_10-K_2023_chunk_{i:04d}",
                doc_id=f"{ticker}_10-K_2023",
                doc_type=DocumentType.SEC_10K,
                ticker=ticker,
                company_name=f"{ticker} Inc.",
                filing_date="2023-10-27",
                fiscal_year="2023",
                chunk_index=i,
                content_hash=content_hash,
            ),
        )
        chunks.append(chunk)
    return chunks


# ============================================================
# Collection Initialization Tests
# ============================================================

class TestCollectionInit:
    """Tests for Qdrant collection initialization."""

    def test_creates_collection(self, in_memory_qdrant):
        """init_collection should create the collection."""
        init_collection()
        collections = in_memory_qdrant.get_collections().collections
        names = [c.name for c in collections]
        assert "sec_filings" in names

    def test_idempotent(self, in_memory_qdrant):
        """Calling init_collection twice should not raise an error."""
        init_collection()
        init_collection()  # Should not raise
        collections = in_memory_qdrant.get_collections().collections
        names = [c.name for c in collections]
        assert names.count("sec_filings") == 1


# ============================================================
# Upsert Tests
# ============================================================

class TestUpsertChunks:
    """Tests for upserting chunks into Qdrant."""

    def test_upsert_empty_list(self, in_memory_qdrant):
        """Upserting an empty list should return 0."""
        init_collection()
        assert upsert_chunks([]) == 0

    def test_upsert_sample_chunks(self, in_memory_qdrant, sample_chunks):
        """Upserting sample chunks should store them in Qdrant."""
        init_collection()
        count = upsert_chunks(sample_chunks)
        assert count == len(sample_chunks)

        # Verify points are in the collection
        info = in_memory_qdrant.get_collection("sec_filings")
        assert info.points_count == len(sample_chunks)

    def test_upsert_with_sparse_vectors(self, in_memory_qdrant, sample_chunks):
        """Upserting chunks with sparse vectors should store both named vectors."""
        init_collection()
        fit_bm25([c.text for c in sample_chunks])
        sparse_vecs = [encode_sparse_document(c.text) for c in sample_chunks]

        count = upsert_chunks(sample_chunks, sparse_vectors=sparse_vecs)
        assert count == len(sample_chunks)

        info = in_memory_qdrant.get_collection("sec_filings")
        assert info.points_count == len(sample_chunks)

    def test_upsert_is_idempotent(self, in_memory_qdrant, sample_chunks):
        """Upserting the same chunks twice should not create duplicates."""
        init_collection()
        upsert_chunks(sample_chunks)
        upsert_chunks(sample_chunks)  # Second upsert

        info = in_memory_qdrant.get_collection("sec_filings")
        assert info.points_count == len(sample_chunks)

    def test_batch_upsert(self, in_memory_qdrant, sample_chunks):
        """Batch size smaller than chunk count should still upsert all."""
        init_collection()
        count = upsert_chunks(sample_chunks, batch_size=2)
        assert count == len(sample_chunks)

        info = in_memory_qdrant.get_collection("sec_filings")
        assert info.points_count == len(sample_chunks)


# ============================================================
# Dense Search Tests
# ============================================================

class TestDenseSearch:
    """Tests for dense vector search."""

    def test_search_returns_ranked_results(self, in_memory_qdrant, sample_chunks):
        """Search should return results sorted by relevance."""
        init_collection()
        upsert_chunks(sample_chunks)

        results = search_dense("iPhone revenue", top_k=3)
        assert len(results) == 3
        # First result should be the iPhone chunk
        assert "iPhone" in results[0].text
        # Scores should be in descending order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_ticker_filter(self, in_memory_qdrant, sample_chunks):
        """Filtering by ticker should return only chunks from that ticker."""
        init_collection()
        upsert_chunks(sample_chunks)

        results = search_dense("revenue", top_k=10, ticker_filter="MSFT")
        assert len(results) > 0
        for r in results:
            assert r.metadata["ticker"] == "MSFT"

    def test_search_with_doc_type_filter(self, in_memory_qdrant, sample_chunks):
        """Filtering by doc_type should return only matching chunks."""
        init_collection()
        upsert_chunks(sample_chunks)

        # All our test data is 10-K, so filtering for 10-K should return all
        results = search_dense("revenue", top_k=10, doc_type_filter="10-K")
        assert len(results) == len(sample_chunks)

    def test_search_empty_collection(self, in_memory_qdrant):
        """Searching an empty collection should return empty list."""
        init_collection()
        results = search_dense("revenue", top_k=5)
        assert len(results) == 0


# ============================================================
# Sparse Search Tests
# ============================================================

class TestSparseSearch:
    """Tests for sparse BM25 vector search."""

    def test_sparse_search_finds_exact_keywords(self, in_memory_qdrant, sample_chunks):
        """Sparse search should rank chunks with exact keyword matches at the top."""
        init_collection()
        fit_bm25([c.text for c in sample_chunks])
        sparse_vecs = [encode_sparse_document(c.text) for c in sample_chunks]
        upsert_chunks(sample_chunks, sparse_vectors=sparse_vecs)

        results = search_sparse("Azure cloud", top_k=3)
        assert len(results) > 0
        assert "Azure" in results[0].text
        assert results[0].metadata["ticker"] == "MSFT"

    def test_sparse_search_with_ticker_filter(self, in_memory_qdrant, sample_chunks):
        """Sparse search should respect ticker filters."""
        init_collection()
        fit_bm25([c.text for c in sample_chunks])
        sparse_vecs = [encode_sparse_document(c.text) for c in sample_chunks]
        upsert_chunks(sample_chunks, sparse_vectors=sparse_vecs)

        results = search_sparse("revenue", top_k=5, ticker_filter="AAPL")
        assert len(results) > 0
        for r in results:
            assert r.metadata["ticker"] == "AAPL"

    def test_sparse_search_unknown_terms_empty(self, in_memory_qdrant, sample_chunks):
        """Search with terms not in the corpus returns empty results gracefully."""
        init_collection()
        fit_bm25([c.text for c in sample_chunks])
        results = search_sparse("nonexistentwordxyz", top_k=5)
        assert results == []


# ============================================================
# Existing Hashes Tests
# ============================================================

class TestGetExistingHashes:
    """Tests for retrieving content hashes for dedup."""

    def test_empty_collection(self, in_memory_qdrant):
        """Empty collection should return empty set."""
        init_collection()
        hashes = get_existing_hashes()
        assert hashes == set()

    def test_returns_all_hashes(self, in_memory_qdrant, sample_chunks):
        """Should return the content hash of every stored chunk."""
        init_collection()
        upsert_chunks(sample_chunks)

        hashes = get_existing_hashes()
        expected_hashes = {c.metadata.content_hash for c in sample_chunks}
        assert hashes == expected_hashes


# ============================================================
# Point ID Tests
# ============================================================

class TestChunkIdToInt:
    """Tests for the chunk_id → integer point ID conversion."""

    def test_deterministic(self):
        """Same chunk_id should always produce the same integer."""
        id1 = _chunk_id_to_int("AAPL_10-K_2023_chunk_0001")
        id2 = _chunk_id_to_int("AAPL_10-K_2023_chunk_0001")
        assert id1 == id2

    def test_different_ids_different_ints(self):
        """Different chunk_ids should produce different integers."""
        id1 = _chunk_id_to_int("AAPL_10-K_2023_chunk_0001")
        id2 = _chunk_id_to_int("AAPL_10-K_2023_chunk_0002")
        assert id1 != id2

    def test_positive_integer(self):
        """Point IDs should be positive (Qdrant requirement)."""
        id1 = _chunk_id_to_int("test_chunk_id")
        assert id1 > 0
