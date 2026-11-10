"""
Data models for the Semantic Cache and Lineage subsystem (Day 8).
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.schemas import Citation, AnswerResponse


class CachedAnswer(BaseModel):
    """
    A cached query response containing complete citation binding and lineage metadata.

    WHY store full citation and confidence lineage:
    - We cannot simply cache plain text answers; financial compliance demands
      the caller always receives verifiable chunk IDs and quotes.
    - Storing dependent_chunk_ids and dependent_doc_ids creates the bipartite
      graph needed for surgical cache invalidation when filings are amended.
    """

    cache_key: str = Field(..., description="Unique key for the cached item (e.g. 'cache_a1b2c3d4')")
    query: str = Field(..., description="Canonical query that generated this cache entry")
    query_embedding: List[float] = Field(default_factory=list, description="Dense embedding vector of the query")
    answer: str = Field(..., description="Synthesized grounded answer")
    citations: List[Citation] = Field(default_factory=list, description="Source chunk citations bound to this answer")
    confidence: float = Field(..., description="Composite calibrated confidence score")
    decision: str = Field(default="generate", description="Agent decision that created this answer")
    ticker_filter: Optional[str] = Field(default=None, description="Scope filter: ticker symbol")
    doc_type_filter: Optional[str] = Field(default=None, description="Scope filter: document type")
    dependent_chunk_ids: List[str] = Field(default_factory=list, description="Chunk IDs cited or used as evidence")
    dependent_doc_ids: List[str] = Field(default_factory=list, description="Document IDs cited or used as evidence")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ttl_seconds: int = Field(default=86400, description="Time to live in seconds")
    hit_count: int = Field(default=0, description="Number of times this cache entry has been served")

    def to_answer_response(self, query: str, latency_ms: float = 0.5) -> AnswerResponse:
        """Convert cached entry into an AnswerResponse with cached=True."""
        return AnswerResponse(
            query=query,
            answer=self.answer,
            citations=self.citations,
            confidence=self.confidence,
            decision=self.decision,
            retries=0,
            latency_ms=round(latency_ms, 2),
            cached=True,
        )


class InvalidationResult(BaseModel):
    """Summary of cache invalidation actions performed."""

    target: str = Field(..., description="Target invalidated (e.g. document ID or chunk IDs)")
    evicted_cache_keys: List[str] = Field(default_factory=list, description="List of cache keys evicted")
    evicted_count: int = Field(default=0, description="Total number of cached queries evicted")
    affected_queries: List[str] = Field(default_factory=list, description="Original queries whose cache was invalidated")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class CacheStats(BaseModel):
    """Telemetry and operational statistics for the cache system."""

    total_entries: int = 0
    total_hits: int = 0
    l1_exact_hits: int = 0
    l2_semantic_hits: int = 0
    misses: int = 0
    hit_rate: float = 0.0
    storage_backend: str = "redis"

