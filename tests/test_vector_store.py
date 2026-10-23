"""
Tests for the Qdrant vector store operations.

These tests use an in-memory Qdrant client (no Docker needed) to verify:
1. Collection initialization is idempotent
2. Chunk upsert creates correct points with payloads
3. Dense search returns ranked results
4. Payload filters (ticker, doc_type) work correctly
5. get_existing_hashes retrieves all content hashes
6. Deterministic point ID generation

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
    get_existing_hashes,
    set_qdrant_client,
    get_qdrant_client,
    _chunk_id_to_int,
)
from app.retrieval.embedder import embed_texts


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
        """Calling init_collection twice should not raise."""
        init_collection()
        init_collection()  # Should not raise
        collections = in_memory_qdrant.get_collections().collections
        # Should still be just one collection
        names = [c.name for c in collections]
        assert names.count("sec_filings") == 1


# ============================================================
# Upsert Tests
# ============================================================

class TestUpsertChunks:
    """Tests for embedding + upserting chunks to Qdrant."""

    def test_upsert_creates_points(self, in_memory_qdrant, sample_chunks):
        """Upserting chunks should create points in Qdrant."""
        init_collection()
        count = upsert_chunks(sample_chunks)
        assert count == len(sample_chunks)

        # Verify points exist
        collection_info = in_memory_qdrant.get_collection("sec_filings")
        assert collection_info.points_count == len(sample_chunks)

    def test_upsert_preserves_payload(self, in_memory_qdrant, sample_chunks):
        """Upserted points should have correct payload fields."""
        init_collection()
        upsert_chunks(sample_chunks[:1])  # Just upsert the first chunk

        # Retrieve the point
        point_id = _chunk_id_to_int(sample_chunks[0].metadata.chunk_id)
        points = in_memory_qdrant.retrieve(
            collection_name="sec_filings",
            ids=[point_id],
            with_payload=True,
        )
        assert len(points) == 1
        payload = points[0].payload

        assert payload["ticker"] == "AAPL"
        assert payload["doc_type"] == "10-K"
        assert payload["text"] == sample_chunks[0].text
        assert payload["content_hash"] == sample_chunks[0].metadata.content_hash

    def test_upsert_empty_list(self, in_memory_qdrant):
        """Upserting an empty list should return 0 and not raise."""
        init_collection()
        count = upsert_chunks([])
        assert count == 0

    def test_upsert_is_idempotent(self, in_memory_qdrant, sample_chunks):
        """Upserting the same chunks twice should not create duplicates."""
        init_collection()
        upsert_chunks(sample_chunks)
        upsert_chunks(sample_chunks)  # Same data again

        collection_info = in_memory_qdrant.get_collection("sec_filings")
        assert collection_info.points_count == len(sample_chunks)


# ============================================================
# Search Tests
# ============================================================

class TestDenseSearch:
    """Tests for dense vector search."""

    def test_search_returns_results(self, in_memory_qdrant, sample_chunks):
        """Search should return results from indexed chunks."""
        init_collection()
        upsert_chunks(sample_chunks)

        results = search_dense("What was Apple's revenue?", top_k=3)
        assert len(results) > 0
        assert len(results) <= 3

    def test_search_result_shape(self, in_memory_qdrant, sample_chunks):
        """Each result should have chunk_id, text, score, and metadata."""
        init_collection()
        upsert_chunks(sample_chunks)

        results = search_dense("revenue", top_k=1)
        assert len(results) >= 1

        result = results[0]
        assert result.chunk_id != ""
        assert result.text != ""
        assert isinstance(result.score, float)
        assert result.score > 0

    def test_search_relevance_ranking(self, in_memory_qdrant, sample_chunks):
        """
        Searching for 'Apple revenue' should rank AAPL chunks higher
        than MSFT or GOOGL chunks.
        """
        init_collection()
        upsert_chunks(sample_chunks)

        results = search_dense("Apple revenue fiscal year", top_k=6)

        # The top result should be about Apple
        assert results[0].metadata.get("ticker") == "AAPL"

    def test_search_with_ticker_filter(self, in_memory_qdrant, sample_chunks):
        """Filtering by ticker should return only chunks from that ticker."""
        init_collection()
        upsert_chunks(sample_chunks)

        results = search_dense("revenue", top_k=10, ticker_filter="MSFT")
        assert len(results) > 0
        for result in results:
            assert result.metadata.get("ticker") == "MSFT"

    def test_search_with_doc_type_filter(self, in_memory_qdrant, sample_chunks):
        """Filtering by doc_type should return only matching chunks."""
        init_collection()
        upsert_chunks(sample_chunks)

        # All our test data is 10-K, so filtering for 10-K should return all
        results = search_dense("revenue", top_k=10, doc_type_filter="10-K")
        assert len(results) > 0
        for result in results:
            assert result.metadata.get("doc_type") == "10-K"

    def test_search_empty_collection(self, in_memory_qdrant):
        """Searching an empty collection should return no results."""
        init_collection()
        results = search_dense("revenue", top_k=5)
        assert len(results) == 0


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
