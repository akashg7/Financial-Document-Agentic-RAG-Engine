"""
Prometheus metrics for the Financial Document Intelligence Engine.

WHY Prometheus rather than custom logging metrics:
- Prometheus is the industry standard for time-series observability. Any
  SRE or DevOps team can scrape /metrics and plug into existing dashboards
  without writing custom aggregation code.
- Histogram pre-computes quantiles server-side, which is more efficient
  than shipping raw latency values and computing percentiles in a logging
  pipeline.
- Labels (decision, cache_tier, retrieval_method) enable drill-down without
  separate counter families.

WHY we don't use prometheus_fastapi_instrumentator:
- It adds generic HTTP metrics (latency per route, status codes). We want
  RAG-domain-specific metrics: confidence distributions, cache tier hit rates,
  retrieval vs generation latency breakdown. Custom counters/histograms give
  us exactly the observability an interviewer would ask about.
"""

from prometheus_client import Counter, Histogram, Gauge, Info


# ============================================================
# Application Info
# ============================================================

APP_INFO = Info(
    "rag_engine",
    "Financial Document Intelligence Engine metadata",
)
APP_INFO.info({
    "version": "0.1.0",
    "embedding_model": "all-MiniLM-L6-v2",
    "reranker_model": "ms-marco-MiniLM-L-6-v2",
})


# ============================================================
# Query Metrics
# ============================================================

QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total queries processed by the agentic loop",
    ["decision"],  # generate | rewrite | abstain
)

QUERY_LATENCY = Histogram(
    "rag_query_latency_seconds",
    "End-to-end query latency including retrieval, generation, and caching",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0],
)

CONFIDENCE_SCORE = Histogram(
    "rag_confidence_score",
    "Distribution of confidence scores returned by the agent",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)


# ============================================================
# Cache Metrics
# ============================================================

CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Cache hits by tier",
    ["tier"],  # L1_exact | L2_semantic
)

CACHE_MISSES = Counter(
    "rag_cache_misses_total",
    "Cache misses (full pipeline execution required)",
)


# ============================================================
# Retrieval Metrics
# ============================================================

RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds",
    "Latency of the retrieve node (hybrid search + rerank)",
    buckets=[0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)

GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds",
    "Latency of the generate node (LLM call + citation validation)",
    buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
)

CHUNKS_RETRIEVED = Histogram(
    "rag_chunks_retrieved",
    "Number of chunks returned by the retrieve node per query",
    buckets=[0, 1, 2, 3, 5, 8, 10, 15, 20],
)


# ============================================================
# Rate Limiting Metrics
# ============================================================

RATE_LIMITED_REQUESTS = Counter(
    "rag_rate_limited_total",
    "Total requests rejected by rate limiter (HTTP 429)",
)


# ============================================================
# Ingestion Metrics
# ============================================================

INGESTION_JOBS = Counter(
    "rag_ingestion_jobs_total",
    "Total ingestion jobs started",
)

INGESTION_CHUNKS = Counter(
    "rag_ingestion_chunks_total",
    "Total chunks ingested",
    ["status"],  # stored | deduplicated
)


# ============================================================
# Active Queries Gauge
# ============================================================

ACTIVE_QUERIES = Gauge(
    "rag_active_queries",
    "Number of queries currently being processed",
)

