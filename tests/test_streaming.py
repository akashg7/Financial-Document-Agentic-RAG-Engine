"""
Tests for Server-Sent Events (SSE) streaming endpoint and generator.

Day 9:
- Stream generator yields SSE-formatted events ('status', 'chunk', 'done')
- POST /api/v1/query/stream returns text/event-stream
- Streaming delivers valid AnswerResponse in final done event
- Cache hit fast-path in SSE streaming
"""

import json
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.api.streaming import stream_agent_response
from app.schemas import AnswerResponse


@pytest.fixture
async def client():
    """Async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestStreamGenerator:
    """Unit tests for the stream_agent_response async generator."""

    @pytest.mark.asyncio
    async def test_stream_yields_valid_sse_format(self):
        """Verify all yielded events adhere to the W3C SSE wire specification."""
        events = []
        async for chunk in stream_agent_response(
            query="What was Apple total revenue in fiscal 2023?",
            top_k=2,
            ticker_filter="AAPL",
            max_retries=1,
        ):
            events.append(chunk)

        assert len(events) >= 3, f"Expected at least 3 events, got {len(events)}"

        # Every chunk must end with double newline according to SSE standard
        for event_str in events:
            assert event_str.endswith("\n\n"), f"SSE chunk missing double newline: {event_str}"
            assert event_str.startswith("event: "), f"SSE chunk missing event header: {event_str}"
            assert "\ndata: " in event_str, f"SSE chunk missing data field: {event_str}"

    @pytest.mark.asyncio
    async def test_stream_event_sequence(self):
        """Verify the progression of pipeline stages in status events."""
        event_types = []
        statuses = []
        done_payload = None

        async for chunk in stream_agent_response(
            query="What was Apple iPhone revenue in 2023?",
            top_k=2,
            ticker_filter="AAPL",
            max_retries=1,
        ):
            lines = chunk.strip().split("\n")
            event_type = lines[0].replace("event: ", "").strip()
            data_json = json.loads(lines[1].replace("data: ", "").strip())

            event_types.append(event_type)
            if event_type == "status":
                statuses.append(data_json.get("stage"))
            elif event_type == "done":
                done_payload = data_json

        # Must include status events, at least one chunk event, and a final done event
        assert "status" in event_types
        assert "chunk" in event_types
        assert "done" in event_types
        assert event_types[-1] == "done"

        # Check status progression
        assert "cache_check" in statuses

        # Verify done payload conforms to AnswerResponse
        assert done_payload is not None
        validated_resp = AnswerResponse(**done_payload)
        assert validated_resp.query == "What was Apple iPhone revenue in 2023?"
        assert isinstance(validated_resp.answer, str)
        assert len(validated_resp.answer) > 0


class TestStreamingEndpoint:
    """Integration tests for POST /api/v1/query/stream endpoint."""

    @pytest.mark.asyncio
    async def test_query_stream_endpoint_returns_200_and_event_stream(self, client: AsyncClient):
        """Verify the streaming endpoint returns HTTP 200 with text/event-stream media type."""
        response = await client.post(
            "/api/v1/query/stream",
            json={
                "query": "What were the primary risk factors in fiscal 2023?",
                "top_k": 2,
                "ticker_filter": "AAPL",
                "max_retries": 1,
            },
        )
        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type

        # Verify stream contains expected event markers
        body_text = response.text
        assert "event: status" in body_text
        assert "event: chunk" in body_text
        assert "event: done" in body_text

    @pytest.mark.asyncio
    async def test_query_stream_validation_error(self, client: AsyncClient):
        """Verify 422 Unprocessable Entity when query string is empty."""
        response = await client.post(
            "/api/v1/query/stream",
            json={"query": ""},
        )
        assert response.status_code == 422

