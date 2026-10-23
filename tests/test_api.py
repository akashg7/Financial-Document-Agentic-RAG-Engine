"""
Tests for the FastAPI application endpoints.

Day 1 tests:
- Health check
- POST /ingest creates a job
- GET /job/{id} returns status
- GET /job/{unknown_id} returns 404

Day 2 additions:
- POST /search/dense returns search results
- Search with filters
- Search on empty collection

NOTE: The search endpoint tests use an in-memory Qdrant client
injected via set_qdrant_client(). This avoids needing Docker for tests.
"""

import pytest
import hashlib
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.ingestion.redis_job_store import force_memory_fallback, reset_store


@pytest.fixture(autouse=True)
def setup_job_store():
    """Use in-memory job store for all API tests."""
    force_memory_fallback()
    reset_store()
    yield
    reset_store()


@pytest.fixture
async def client():
    """Async HTTP client for testing FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthCheck:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_returns_status(self, client):
        resp = await client.get("/health")
        data = resp.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestIngestionEndpoint:
    """Tests for POST /api/v1/ingest."""

    @pytest.mark.asyncio
    async def test_ingest_returns_job_id(self, client):
        """POST /ingest should return a job_id and pending status."""
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
        assert data["status"] == "pending"

    @pytest.mark.asyncio
    async def test_ingest_invalid_source_still_creates_job(self, client):
        """Even an unknown source creates a job (it will fail during processing)."""
        resp = await client.post(
            "/api/v1/ingest",
            json={"source": "unknown_source"},
        )
        assert resp.status_code == 200
        assert "job_id" in resp.json()


class TestJobStatusEndpoint:
    """Tests for GET /api/v1/job/{job_id}."""

    @pytest.mark.asyncio
    async def test_job_status_after_create(self, client):
        """After creating a job, its status should be retrievable."""
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
