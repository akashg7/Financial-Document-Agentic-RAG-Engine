# Day 9 Learning Guide: Production Hardening — SSE Streaming, Rate Limiting & Observability

Welcome to Day 9! Today we transitioned our financial RAG system from a working prototype to an enterprise-grade production service by adding **real-time Server-Sent Events (SSE) streaming**, **protective multi-tiered rate limiting**, **domain-specific Prometheus observability**, and **distributed tracing**.

---

## 1. Plain English Summary

Imagine you are deploying an automated Bloomberg-style terminal for an investment bank's equity research desk:
1. **The UX Problem (The Waiting Blind Spot)**:
   A complex financial query takes 3 to 10 seconds because the agent performs hybrid search, neural reranking, confidence scoring, and possible query rewrites.
   - If the API is purely synchronous, the analyst stares at a frozen screen thinking the server crashed.
   - With **Server-Sent Events (SSE)**, the terminal streams immediate progress: *"Retrieving 10-K filings..."* $\to$ *"Found 3 relevant chunks"* $\to$ *"Confidence 0.94, synthesizing grounded answer..."* $\to$ followed by streaming text tokens. The perceived wait time drops from 10 seconds to under 200 milliseconds.

2. **The Abuse Problem (The Runaway Script)**:
   An analyst writes a Python script with an infinite loop or a DDoS attack hits your API. Without protection, your Groq and Gemini API keys would be depleted in seconds, incurring massive cloud bills or hitting provider rate limits and crashing the service for everyone.
   - With **Tiered Rate Limiting**, the system enforces boundaries per client IP: ingestion (heavy) is limited to 5/min, generation (expensive LLM) is limited to 15/min, and search is capped at 30/min. Any violation immediately receives an informative **HTTP 429 Too Many Requests** with `retry_after_seconds: 60`.

3. **The SRE Problem (Flying Blind in Production)**:
   How do you know if the agent is abstaining too frequently? How do you detect if retrieval latency spiked after a new collection was indexed?
   - With **Prometheus Observability**, the application automatically records every query, latency histogram, confidence score, and cache hit/miss across atomic in-memory counters. Any Prometheus/Grafana dashboard can scrape `GET /metrics` in real-time.

---

## 2. First-Principles Architecture

### A. Server-Sent Events (SSE) vs. WebSockets

```
┌─────────────────────────────────────────────────────────────┐
│                 Client-Server Communication                 │
├──────────────────────────────┬──────────────────────────────┤
│    Server-Sent Events (SSE)  │          WebSockets          │
├──────────────────────────────┼──────────────────────────────┤
│ Unidirectional (Server → Cl.)│ Full-duplex (Bidirectional)  │
│ Standard HTTP/1.1 or HTTP/2  │ Upgraded TCP protocol (ws://)│
│ Built-in auto-reconnect      │ Custom heartbeat & reconnect │
│ Passes proxies & firewalls   │ Needs special proxy config   │
│ UTF-8 text framing           │ Binary and text frames       │
│ Ideal for LLM token streams  │ Ideal for multiplayer/chat   │
└──────────────────────────────┴──────────────────────────────┘
```

#### Why SSE is the Superior Choice for RAG:
In an agentic query pipeline, the client sends a single question (`QueryRequest`), and the server generates a sequence of progress events followed by the answer. The client **never sends data back during generation**. Using WebSockets for this use case adds protocol negotiation complexity, stateful TCP connection pooling, and load-balancer sticky-session headaches for zero functional benefit.

#### Wire Specification:
Each event emitted follows the W3C EventSource standard:
```
event: status
data: {"stage": "retrieving", "message": "Retrieving documents (attempt 1)..."}

event: status
data: {"stage": "generating", "message": "Generating grounded answer..."}

event: chunk
data: {"text": "In fiscal year 2023, Apple Inc. reported..."}

event: done
data: {"query": "...", "answer": "...", "citations": [...], "confidence": 0.94, "decision": "generate"}
```

---

### B. Tiered Rate Limiting with SlowAPI

Rate limiting protects downstream resources from overload. However, a single global rate limit is inappropriate because different endpoints have fundamentally different computational costs:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Endpoint Rate Limiting Tiers                    │
├───────────────────────┬────────────┬───────────────────────────────────┤
│ Endpoint              │ Rate Limit │ Cost / Rationale                  │
├───────────────────────┼────────────┼───────────────────────────────────┤
│ POST /api/v1/ingest   │ 5 / min    │ Heavy PDF/HTML parse + embeddings │
│ POST /api/v1/query    │ 15 / min   │ Multi-stage agent + LLM inference │
│ POST /api/v1/generate │ 15 / min   │ Rerank + LLM prompt generation    │
│ POST /api/v1/search/* │ 30 / min   │ Qdrant vector & BM25 sparse search│
│ POST /api/v1/cache/*  │ 30 / min   │ Redis DAG invalidation operations │
│ GET /health, /metrics │ Unlimited  │ Health probes & Prometheus scrapes│
└───────────────────────┴────────────┴───────────────────────────────────┘
```

#### Implementation Architecture:
- Built with **SlowAPI** (wrapping `limits`), tracking client IP address via `get_remote_address`.
- Attached via `SlowAPIMiddleware` and decorated on route handlers.
- **Critical Architectural Detail**: Route functions accept `request: Request` explicitly so SlowAPI can extract client headers, while the request body is bound to `body: <PydanticModel>` without breaking serialization.
- Custom `RateLimitExceeded` handler:
  ```json
  {
    "error": "RateLimitExceeded",
    "detail": "Rate limit exceeded: 5 per 1 minute",
    "retry_after_seconds": 60
  }
  ```

---

### C. Domain-Specific Prometheus Observability

Standard HTTP instrumentators (e.g., `prometheus-fastapi-instrumentator`) only report generic HTTP response codes (`http_requests_total{status="200"}`). In an agentic RAG system, an HTTP 200 could mean:
1. High-confidence verified generation ($C = 0.94$).
2. A low-confidence retry loop rewrite ($C = 0.55$).
3. A disciplined financial abstention ($C = 0.20$).

Generic HTTP metrics treat all three as identical "successes"! We implemented **domain-specific Prometheus instrumentation** to monitor actual RAG health:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Prometheus Metric Registry                      │
├───────────────────────────────┬───────────┬────────────────────────────┤
│ Metric Name                   │ Type      │ Labels / Buckets           │
├───────────────────────────────┼───────────┼────────────────────────────┤
│ rag_queries_total             │ Counter   │ decision (gen/rew/abstain) │
│ rag_query_latency_seconds     │ Histogram │ 0.1s to 30.0s buckets      │
│ rag_confidence_score          │ Histogram │ 0.0 to 1.0 (0.1 step)      │
│ rag_cache_hits_total          │ Counter   │ tier (L1_exact/L2_semantic)│
│ rag_cache_misses_total        │ Counter   │ (none)                     │
│ rag_retrieval_latency_seconds │ Histogram │ 0.05s to 5.0s buckets      │
│ rag_generation_latency_seconds│ Histogram │ 0.1s to 10.0s buckets      │
│ rag_chunks_retrieved          │ Histogram │ 0 to 20 chunk buckets      │
│ rag_rate_limited_total        │ Counter   │ (none)                     │
│ rag_active_queries            │ Gauge     │ (tracks concurrent load)   │
│ rag_engine_info               │ Info      │ version, models, dims      │
└───────────────────────────────┴───────────┴────────────────────────────┘
```

---

## 3. Interview Defense Questions & Answers

### Q1: Why did you choose Server-Sent Events (SSE) over WebSockets for streaming answers?
> **Answer**: "The agentic financial QA interaction is inherently unidirectional: the client submits a structured question, and the server generates real-time pipeline status updates followed by incremental answer tokens. WebSockets are designed for bidirectional, full-duplex communication (like collaborative editing or chat rooms), which introduces unnecessary protocol complexity—including connection upgrade negotiation, stateful TCP socket pooling across instances, custom heartbeat/ping-pong handling, and firewall traversal issues. SSE runs over standard HTTP, natively supports auto-reconnection, works effortlessly through enterprise proxies and ALBs, and allows us to emit custom typed event streams (`status`, `chunk`, `done`) that frontends can cleanly consume using standard browser `EventSource` APIs."

### Q2: How did you implement rate limiting without hurting Prometheus scraping or health checks?
> **Answer**: "We implemented tiered, IP-based rate limiting using SlowAPI. Rather than applying a blunt global limit across all routes, we tiered endpoints based on computational and financial cost: heavy document ingestion is limited to 5 requests/minute, LLM generation and agentic queries are capped at 15 requests/minute, and vector search is allowed 30 requests/minute. Crucially, operational endpoints—specifically `/health` for Kubernetes liveness/readiness probes and `/metrics` for high-frequency Prometheus scrapers (every 15 seconds)—are completely exempt from rate limiting to prevent false-positive service outages."

### Q3: What is the difference between generic HTTP metrics and domain-specific RAG metrics?
> **Answer**: "In a standard web app, an HTTP 200 response indicates success. In an agentic RAG system, an HTTP 200 can mask critical system degradation—for instance, if the agent fails to find evidence and returns a structured abstention, or if the LLM enters an endless query-rewriting loop before degrading to a fallback response. Both return HTTP 200. We built custom Prometheus metrics tracking:
> 1. `rag_queries_total` partitioned by agent decision (`generate`, `rewrite`, `abstain`) to immediately detect sudden drops in retrieval recall.
> 2. `rag_confidence_score` histograms to monitor output quality distributions.
> 3. `rag_cache_hits_total` partitioned by tier (`L1_exact` vs. `L2_semantic`) to measure caching efficiency.
> 4. Phase-specific latency histograms separating retrieval latency from LLM generation latency to pinpoint upstream bottlenecks."

