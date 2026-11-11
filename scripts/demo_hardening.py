"""
Demonstration and validation script for Day 9 Production Hardening.

Exercises:
1. Server-Sent Events (SSE) streaming (/api/v1/query/stream)
2. SlowAPI Rate Limiting & HTTP 429 structured responses
3. Prometheus metrics instrumentation and /metrics scraping

Saves execution telemetry to /results/hardening_demo_<timestamp>.json.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import time
from datetime import datetime, timezone
from httpx import AsyncClient, ASGITransport

import asyncio
from app.main import app
from app.observability.metrics import QUERIES_TOTAL, CACHE_HITS, RATE_LIMITED_REQUESTS


async def run_hardening_demo():
    print("=" * 70)
    print("  DAY 9 PRODUCTION HARDENING: SSE, RATE LIMITING & OBSERVABILITY")
    print("=" * 70)

    transport = ASGITransport(app=app)
    results_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "sse_streaming": {},
        "rate_limiting": {},
        "prometheus_metrics": {},
    }

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # ----------------------------------------------------
        # 1. SSE Streaming Demonstration
        # ----------------------------------------------------
        print("\n[1/3] Testing Server-Sent Events (SSE) Streaming...")
        stream_query = "What was Apple's total revenue in fiscal year 2023?"
        t0 = time.time()

        resp = await client.post(
            "/api/v1/query/stream",
            json={
                "query": stream_query,
                "top_k": 3,
                "ticker_filter": "AAPL",
                "max_retries": 1,
            },
        )

        stream_latency_ms = (time.time() - t0) * 1000
        print(f"  -> HTTP Status: {resp.status_code}")
        print(f"  -> Content-Type: {resp.headers.get('content-type')}")
        print(f"  -> Time to complete stream: {stream_latency_ms:.2f} ms")

        # Parse SSE events from response text
        raw_chunks = resp.text.strip().split("\n\n")
        events = []
        for raw_chunk in raw_chunks:
            if not raw_chunk.strip():
                continue
            lines = raw_chunk.strip().split("\n")
            event_type = lines[0].replace("event: ", "").strip()
            data_str = lines[1].replace("data: ", "").strip()
            try:
                data_json = json.loads(data_str)
            except Exception:
                data_json = data_str
            events.append({"event": event_type, "data": data_json})
            if event_type == "status":
                print(f"     [EVENT: status] stage='{data_json.get('stage')}' message='{data_json.get('message')}'")
            elif event_type == "chunk":
                preview = data_json.get("text", "")[:60] + "..." if len(data_json.get("text", "")) > 60 else data_json.get("text", "")
                print(f"     [EVENT: chunk] text='{preview}'")
            elif event_type == "done":
                print(f"     [EVENT: done] decision='{data_json.get('decision')}' confidence={data_json.get('confidence')}")

        results_data["sse_streaming"] = {
            "query": stream_query,
            "status_code": resp.status_code,
            "events_received": len(events),
            "latency_ms": round(stream_latency_ms, 2),
            "final_decision": events[-1]["data"].get("decision") if events and events[-1]["event"] == "done" else None,
        }

        # ----------------------------------------------------
        # 2. Rate Limiting Demonstration
        # ----------------------------------------------------
        print("\n[2/3] Testing Rate Limiting on Ingestion Endpoint (limit: 5/minute)...")
        rate_limit_responses = []
        for i in range(7):
            r = await client.post(
                "/api/v1/ingest",
                json={
                    "source": "huggingface",
                    "max_documents": 1,
                },
            )
            rate_limit_responses.append(r.status_code)
            print(f"  -> Request #{i+1}: HTTP {r.status_code}")
            if r.status_code == 429:
                err_body = r.json()
                print(f"     Throttled! Detail: {err_body.get('detail')}")

        has_429 = 429 in rate_limit_responses
        results_data["rate_limiting"] = {
            "endpoint": "/api/v1/ingest",
            "requests_sent": len(rate_limit_responses),
            "status_codes": rate_limit_responses,
            "rate_limit_enforced": has_429,
        }

        # ----------------------------------------------------
        # 3. Prometheus Observability Demonstration
        # ----------------------------------------------------
        print("\n[3/3] Scraping Prometheus Metrics Endpoint (/metrics)...")
        metrics_resp = await client.get("/metrics")
        print(f"  -> HTTP Status: {metrics_resp.status_code}")
        print(f"  -> Content-Type: {metrics_resp.headers.get('content-type')}")

        sample_lines = []
        metrics_text = metrics_resp.text
        for line in metrics_text.splitlines():
            if line.startswith("rag_") and not line.startswith("rag_engine_info"):
                sample_lines.append(line)

        print(f"  -> Extracted {len(sample_lines)} domain metric telemetry lines:")
        for line in sample_lines[:8]:
            print(f"     {line}")

        results_data["prometheus_metrics"] = {
            "status_code": metrics_resp.status_code,
            "sample_telemetry": sample_lines[:15],
            "total_domain_metric_lines": len(sample_lines),
        }

    # Save results to /results
    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = results_dir / f"hardening_demo_{timestamp_str}.json"
    with open(out_file, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"\n[OK] Demonstration metrics written to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_hardening_demo())
