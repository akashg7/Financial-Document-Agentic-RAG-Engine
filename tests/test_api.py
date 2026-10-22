"""
Tests for the FastAPI application endpoints.

Day 1 tests:
- Health check
- POST /ingest creates a job
- GET /job/{id} returns status
- GET /job/{unknown_id} returns 404
"""

import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app


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
