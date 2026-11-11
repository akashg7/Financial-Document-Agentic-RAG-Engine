"""
Observability package for Financial Document Intelligence Engine.

Provides:
- Prometheus domain-specific metrics (app.observability.metrics)
- LangSmith tracing configuration (app.observability.tracing)
"""

from app.observability.metrics import (
    QUERIES_TOTAL,
    QUERY_LATENCY,
    CONFIDENCE_SCORE,
    CACHE_HITS,
    CACHE_MISSES,
    RETRIEVAL_LATENCY,
    GENERATION_LATENCY,
    CHUNKS_RETRIEVED,
    RATE_LIMITED_REQUESTS,
    INGESTION_JOBS,
    INGESTION_CHUNKS,
    ACTIVE_QUERIES,
)
from app.observability.tracing import setup_langsmith

__all__ = [
    "QUERIES_TOTAL",
    "QUERY_LATENCY",
    "CONFIDENCE_SCORE",
    "CACHE_HITS",
    "CACHE_MISSES",
    "RETRIEVAL_LATENCY",
    "GENERATION_LATENCY",
    "CHUNKS_RETRIEVED",
    "RATE_LIMITED_REQUESTS",
    "INGESTION_JOBS",
    "INGESTION_CHUNKS",
    "ACTIVE_QUERIES",
    "setup_langsmith",
]
