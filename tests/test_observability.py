"""
Tests for Prometheus metrics and LangSmith observability instrumentation.

Day 9:
- GET /metrics exposes Prometheus scrape data with custom domain metrics
- Custom metrics (counters, histograms, gauges) update during query execution
- LangSmith tracing configuration initializes environment correctly
"""

import os
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.config import settings
from app.observability.tracing import setup_langsmith
from app.observability.metrics import (
    QUERIES_TOTAL,
    QUERY_LATENCY,
    CONFIDENCE_SCORE,
    CACHE_HITS,
    CACHE_MISSES,
    RATE_LIMITED_REQUESTS,
    ACTIVE_QUERIES,
)


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestPrometheusMetricsEndpoint:
    """Tests for the GET /metrics endpoint and metric registration."""

    @pytest.mark.asyncio
    async def test_metrics_endpoint_returns_prometheus_format(self, client: AsyncClient):
        """Verify /metrics returns 200 and standard Prometheus text format."""
        response = await client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers.get("content-type", "")

        body = response.text
        # Verify key domain metrics are exported
        assert "rag_queries_total" in body
        assert "rag_query_latency_seconds" in body
        assert "rag_confidence_score" in body
        assert "rag_cache_hits_total" in body
        assert "rag_cache_misses_total" in body
        assert "rag_retrieval_latency_seconds" in body
        assert "rag_generation_latency_seconds" in body
        assert "rag_chunks_retrieved" in body
        assert "rag_rate_limited_total" in body
        assert "rag_active_queries" in body
        assert "rag_engine_info" in body

    def test_metrics_increment_and_observe(self):
        """Verify custom metrics update their internal state correctly."""
        # Test counter
        initial_queries = QUERIES_TOTAL.labels(decision="generate")._value.get()
        QUERIES_TOTAL.labels(decision="generate").inc()
        assert QUERIES_TOTAL.labels(decision="generate")._value.get() == initial_queries + 1

        # Test cache counters
        initial_hits = CACHE_HITS.labels(tier="L1_exact")._value.get()
        CACHE_HITS.labels(tier="L1_exact").inc()
        assert CACHE_HITS.labels(tier="L1_exact")._value.get() == initial_hits + 1

        initial_misses = CACHE_MISSES._value.get()
        CACHE_MISSES.inc()
        assert CACHE_MISSES._value.get() == initial_misses + 1

        # Test active queries gauge context manager
        assert ACTIVE_QUERIES._value.get() == 0.0
        with ACTIVE_QUERIES.track_inprogress():
            assert ACTIVE_QUERIES._value.get() == 1.0
        assert ACTIVE_QUERIES._value.get() == 0.0


class TestLangSmithTracing:
    """Tests for LangSmith distributed tracing initialization."""

    def test_setup_langsmith_when_disabled(self, monkeypatch):
        """Verify setup_langsmith returns False and does not alter os.environ when disabled."""
        monkeypatch.setattr(settings, "langsmith_tracing_enabled", False)
        # Clear any preexisting env vars
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        result = setup_langsmith()
        assert result is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") != "true"

    def test_setup_langsmith_when_enabled_without_key(self, monkeypatch):
        """Verify setup_langsmith safely logs warning and returns False if key is missing."""
        monkeypatch.setattr(settings, "langsmith_tracing_enabled", True)
        monkeypatch.setattr(settings, "langsmith_api_key", "")
        monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)

        result = setup_langsmith()
        assert result is False
        assert os.environ.get("LANGCHAIN_TRACING_V2") != "true"

    def test_setup_langsmith_when_enabled_with_key(self, monkeypatch):
        """Verify setup_langsmith sets required LangChain environment variables when configured."""
        monkeypatch.setattr(settings, "langsmith_tracing_enabled", True)
        monkeypatch.setattr(settings, "langsmith_api_key", "ls_test_key_12345")
        monkeypatch.setattr(settings, "langsmith_project", "financial-rag-test")

        result = setup_langsmith()
        assert result is True
        assert os.environ.get("LANGCHAIN_TRACING_V2") == "true"
        assert os.environ.get("LANGCHAIN_API_KEY") == "ls_test_key_12345"
        assert os.environ.get("LANGCHAIN_PROJECT") == "financial-rag-test"
        assert os.environ.get("LANGCHAIN_ENDPOINT") == "https://api.smith.langchain.com"

