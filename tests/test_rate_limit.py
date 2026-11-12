"""
Tests for SlowAPI rate limiting configuration and exception handling.

Day 9:
- Verify rate limiting triggers HTTP 429 on excessive requests
- Verify custom rate limit handler returns structured JSON error
- Verify RATE_LIMITED_REQUESTS Prometheus counter increments on 429
- Verify /health and /metrics are exempt from rate limiting
"""

import pytest
from httpx import AsyncClient, ASGITransport
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.main import app, rate_limit_exceeded_handler
from app.observability.metrics import RATE_LIMITED_REQUESTS


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestRateLimiter:
    """Tests for rate limiting enforcement and metrics tracking."""

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excessive_requests(self):
        """
        Verify that a dedicated test endpoint with a strict 2/minute limit
        blocks the 3rd request with HTTP 429 and updates the Prometheus counter.
        """
        test_limiter = Limiter(key_func=get_remote_address)
        test_app = FastAPI()
        test_app.state.limiter = test_limiter
        test_app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
        test_app.add_middleware(SlowAPIMiddleware)

        @test_app.get("/rate-limited-test")
        @test_limiter.limit("2/minute")
        async def limited_endpoint(request: Request):
            return {"status": "ok"}

        initial_count = RATE_LIMITED_REQUESTS._value.get()

        transport = ASGITransport(app=test_app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            # First two requests should pass
            r1 = await ac.get("/rate-limited-test")
            assert r1.status_code == 200

            r2 = await ac.get("/rate-limited-test")
            assert r2.status_code == 200

            # Third request should be blocked
            r3 = await ac.get("/rate-limited-test")
            assert r3.status_code == 429

            data = r3.json()
            assert data["error"] == "RateLimitExceeded"
            assert "Rate limit exceeded" in data["detail"]
            assert data.get("retry_after_seconds") == 60

        # Verify Prometheus metric counter incremented
        final_count = RATE_LIMITED_REQUESTS._value.get()
        assert final_count == initial_count + 1

    @pytest.mark.asyncio
    async def test_health_and_metrics_endpoints_exempt(self, client: AsyncClient):
        """Verify operational endpoints (/health and /metrics) are never throttled."""
        for _ in range(5):
            r_health = await client.get("/health")
            assert r_health.status_code == 200

            r_metrics = await client.get("/metrics")
            assert r_metrics.status_code == 200

