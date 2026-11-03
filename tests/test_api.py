"""
Tests for the FastAPI application endpoints.

Day 1 tests:
- Health check
- POST /ingest creates a job
- GET /job/{id} returns status
- GET /job/{unknown_id} returns 404

Day 2 additions:
- POST /search/dense returns search results
- Search request validation (query required)

Day 3 additions:
- POST /search/sparse returns BM25 keyword search results
- POST /search/hybrid returns RRF-fused results
"""

import hashlib
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.schemas import IngestionStatus
from app.ingestion.redis_job_store import reset_store


@pytest.fixture(autouse=True)
def clean_job_store():
    """Ensure the job store is empty before and after each test."""
    reset_store()
    yield
    reset_store()


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    """Tests for GET /health."""

    @pytest.mark.asyncio
    async def test_health_check_returns_200(self, client):
        """Health endpoint should return 200 with status=healthy."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestIngestionEndpoint:
    """Tests for POST /api/v1/ingest."""

    @pytest.mark.asyncio
    async def test_ingest_returns_job_id(self, client):
        """POST /ingest should return 200 with a valid UUID job_id."""
        resp = await client.post(
            "/api/v1/ingest",
            json={
                "source": "sec_edgar",
                "tickers": ["AAPL"],
                "filing_types": ["10-K"],
                "max_documents": 1,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert len(data["job_id"]) == 36  # UUID length
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_ingest_with_defaults(self, client):
        """POST /ingest with minimal fields should use defaults."""
        resp = await client.post(
            "/api/v1/ingest",
            json={"source": "huggingface"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_ingest_missing_source_fails(self, client):
        """POST /ingest without 'source' should return 422 validation error."""
        resp = await client.post("/api/v1/ingest", json={})
        assert resp.status_code == 422


class TestJobStatusEndpoint:
    """Tests for GET /api/v1/job/{job_id}."""

    @pytest.mark.asyncio
    async def test_job_status_after_create(self, client):
        """Immediately after creation, job should be queryable."""
        # Create a job
        create_resp = await client.post(
            "/api/v1/ingest",
            json={"source": "sec_edgar", "tickers": ["AAPL"], "max_documents": 1},
        )
        job_id = create_resp.json()["job_id"]

        # Check status
        status_resp = await client.get(f"/api/v1/job/{job_id}")
        assert status_resp.status_code == 200
        data = status_resp.json()
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "processing", "completed", "failed"]

    @pytest.mark.asyncio
    async def test_unknown_job_returns_404(self, client):
        """Requesting status of a non-existent job should return 404."""
        resp = await client.get("/api/v1/job/nonexistent-id-12345")
        assert resp.status_code == 404


class TestDenseSearchEndpoint:
    """Tests for POST /api/v1/search/dense."""

    @pytest.mark.asyncio
    async def test_search_returns_response_shape(self, client):
        """
        Search endpoint should return the correct response shape
        even on an empty collection.
        """
        from qdrant_client import QdrantClient
        from app.retrieval.vector_store import set_qdrant_client, init_collection

        # Use in-memory Qdrant
        memory_client = QdrantClient(":memory:")
        set_qdrant_client(memory_client)
        init_collection()

        try:
            resp = await client.post(
                "/api/v1/search/dense",
                json={"query": "What was Apple's revenue?", "top_k": 5},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "query" in data
            assert "results" in data
            assert "retrieval_method" in data
            assert data["retrieval_method"] == "dense"
            assert "latency_ms" in data
            assert isinstance(data["latency_ms"], float)
        finally:
            set_qdrant_client(None)

    @pytest.mark.asyncio
    async def test_search_with_results(self, client):
        """Search should return actual results when data is indexed."""
        from qdrant_client import QdrantClient
        from app.retrieval.vector_store import set_qdrant_client, init_collection, upsert_chunks
        from app.schemas import Chunk, ChunkMetadata, DocumentType

        memory_client = QdrantClient(":memory:")
        set_qdrant_client(memory_client)
        init_collection()

        # Insert a test chunk
        text = "Apple reported revenue of $383 billion for fiscal year 2023."
        chunk = Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id="AAPL_10-K_2023_chunk_0000",
                doc_id="AAPL_10-K_2023",
                doc_type=DocumentType.SEC_10K,
                ticker="AAPL",
                chunk_index=0,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            ),
        )
        upsert_chunks([chunk])

        try:
            resp = await client.post(
                "/api/v1/search/dense",
                json={"query": "Apple revenue", "top_k": 3},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data["results"]) >= 1
            assert data["results"][0]["text"] == text
            assert data["results"][0]["chunk_id"] == "AAPL_10-K_2023_chunk_0000"
        finally:
            set_qdrant_client(None)

    @pytest.mark.asyncio
    async def test_search_requires_query(self, client):
        """Search should reject requests without a query."""
        resp = await client.post(
            "/api/v1/search/dense",
            json={"top_k": 5},
        )
        assert resp.status_code == 422  # Validation error


class TestSparseSearchEndpoint:
    """Tests for POST /api/v1/search/sparse."""

    @pytest.mark.asyncio
    async def test_sparse_search_returns_response_shape(self, client):
        """Sparse search endpoint returns valid response structure."""
        from qdrant_client import QdrantClient
        from app.retrieval.vector_store import set_qdrant_client, init_collection, upsert_chunks
        from app.retrieval.sparse import fit_bm25, encode_sparse_document
        from app.schemas import Chunk, ChunkMetadata, DocumentType

        memory_client = QdrantClient(":memory:")
        set_qdrant_client(memory_client)
        init_collection()

        text = "Apple reported revenue of $383 billion for fiscal year 2023."
        fit_bm25([text])
        chunk = Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id="AAPL_10-K_2023_chunk_0000",
                doc_id="AAPL_10-K_2023",
                doc_type=DocumentType.SEC_10K,
                ticker="AAPL",
                chunk_index=0,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            ),
        )
        upsert_chunks([chunk], sparse_vectors=[encode_sparse_document(text)])

        try:
            resp = await client.post(
                "/api/v1/search/sparse",
                json={"query": "revenue", "top_k": 3},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["retrieval_method"] == "sparse"
            assert len(data["results"]) >= 1
            assert "revenue" in data["results"][0]["text"].lower()
        finally:
            set_qdrant_client(None)


class TestHybridSearchEndpoint:
    """Tests for POST /api/v1/search/hybrid."""

    @pytest.mark.asyncio
    async def test_hybrid_search_fuses_results(self, client):
        """Hybrid search endpoint returns RRF-fused results."""
        from qdrant_client import QdrantClient
        from app.retrieval.vector_store import set_qdrant_client, init_collection, upsert_chunks
        from app.retrieval.sparse import fit_bm25, encode_sparse_document
        from app.schemas import Chunk, ChunkMetadata, DocumentType

        memory_client = QdrantClient(":memory:")
        set_qdrant_client(memory_client)
        init_collection()

        text = "Apple reported revenue of $383 billion for fiscal year 2023."
        fit_bm25([text])
        chunk = Chunk(
            text=text,
            metadata=ChunkMetadata(
                chunk_id="AAPL_10-K_2023_chunk_0000",
                doc_id="AAPL_10-K_2023",
                doc_type=DocumentType.SEC_10K,
                ticker="AAPL",
                chunk_index=0,
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
            ),
        )
        upsert_chunks([chunk], sparse_vectors=[encode_sparse_document(text)])

        try:
            resp = await client.post(
                "/api/v1/search/hybrid",
                json={"query": "Apple revenue", "top_k": 3},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["retrieval_method"] == "hybrid"
            assert len(data["results"]) >= 1
            assert data["results"][0]["chunk_id"] == "AAPL_10-K_2023_chunk_0000"
            assert data["results"][0]["score"] > 0
        finally:
            set_qdrant_client(None)
