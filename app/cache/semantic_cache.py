"""
Two-Tier Semantic Caching for the Financial RAG Engine (Day 8).

Tier 1 (L1): Fast hash-based exact string match (< 1ms).
Tier 2 (L2): Dense vector similarity match within scope filter (< 5ms).
"""

import time
import math
import hashlib
import logging
from typing import Optional, Tuple, List, Dict, Any

from app.schemas import AnswerResponse
from app.cache.models import CachedAnswer, CacheStats
from app.cache import backend
from app.retrieval.embedder import embed_query

logger = logging.getLogger(__name__)


def normalize_query(query: str) -> str:
    """Normalize query text for exact string cache lookup."""
    return " ".join(query.strip().lower().split())


def compute_query_hash(normalized_query: str) -> str:
    """Compute 16-character SHA-256 hash of normalized query."""
    return hashlib.sha256(normalized_query.encode("utf-8")).hexdigest()[:16]


def make_scope_key(ticker: Optional[str], doc_type: Optional[str]) -> str:
    """Create partition scope key, e.g. 'AAPL:10-K' or 'ALL:ALL'."""
    t = ticker.upper() if ticker else "ALL"
    d = doc_type.upper() if doc_type else "ALL"
    return f"{t}:{d}"


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Compute cosine similarity between two float vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0

    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b

    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


class SemanticCache:
    """
    Two-Tier Semantic Cache for Financial Document QA.

    L1: Exact query match via SHA-256 hash (< 1 ms latency).
    L2: Dense vector similarity match within matching ticker/doc_type scope (< 5 ms latency).
    """

    def __init__(self, default_similarity_threshold: float = 0.92):
        self.default_similarity_threshold = default_similarity_threshold
        self._stats = {
            "total_requests": 0,
            "l1_exact_hits": 0,
            "l2_semantic_hits": 0,
            "misses": 0,
        }

    def get(
        self,
        query: str,
        ticker_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Optional[CachedAnswer], str]:
        """
        Lookup answer in cache via L1 exact match or L2 semantic similarity.

        Returns:
            Tuple of (CachedAnswer or None, hit_type: 'exact' | 'semantic' | 'miss')
        """
        self._stats["total_requests"] += 1
        threshold = similarity_threshold if similarity_threshold is not None else self.default_similarity_threshold

        norm_q = normalize_query(query)
        q_hash = compute_query_hash(norm_q)
        scope = make_scope_key(ticker_filter, doc_type_filter)

        # ---------------------------------------------------------------------
        # Tier 1: L1 Exact Match (O(1) Hash Lookup)
        # ---------------------------------------------------------------------
        exact_cache_key = backend.get_cache_key_by_exact(scope, q_hash)
        if exact_cache_key:
            entry = backend.get_cache_entry(exact_cache_key)
            if entry:
                logger.info("L1 Exact Cache HIT for query '%s' (key: %s)", query, exact_cache_key)
                self._stats["l1_exact_hits"] += 1
                entry.hit_count += 1
                return entry, "exact"

        # ---------------------------------------------------------------------
        # Tier 2: L2 Semantic Similarity Match (Scoped Vector Search)
        # ---------------------------------------------------------------------
        candidate_keys = backend.get_cache_keys_in_scope(scope)
        if not candidate_keys:
            self._stats["misses"] += 1
            return None, "miss"

        # Embed query for cosine similarity comparison
        query_vec = embed_query(query).tolist()

        best_entry: Optional[CachedAnswer] = None
        best_score = -1.0

        for key in candidate_keys:
            candidate = backend.get_cache_entry(key)
            if not candidate or not candidate.query_embedding:
                continue

            sim = cosine_similarity(query_vec, candidate.query_embedding)
            if sim > best_score:
                best_score = sim
                best_entry = candidate

        if best_entry and best_score >= threshold:
            logger.info(
                "L2 Semantic Cache HIT for query '%s' ~ '%s' (similarity: %.4f >= %.4f, key: %s)",
                query,
                best_entry.query,
                best_score,
                threshold,
                best_entry.cache_key,
            )
            self._stats["l2_semantic_hits"] += 1
            best_entry.hit_count += 1
            return best_entry, "semantic"

        logger.info(
            "Cache MISS for query '%s' (best semantic score: %.4f < %.4f)",
            query,
            best_score,
            threshold,
        )
        self._stats["misses"] += 1
        return None, "miss"

    def set(
        self,
        query: str,
        answer_response: AnswerResponse,
        ticker_filter: Optional[str] = None,
        doc_type_filter: Optional[str] = None,
        ttl_seconds: int = 86400,
    ) -> str:
        """
        Store AnswerResponse in the semantic cache and register dependency lineage.

        Returns:
            Generated cache_key.
        """
        import uuid

        norm_q = normalize_query(query)
        q_hash = compute_query_hash(norm_q)
        scope = make_scope_key(ticker_filter, doc_type_filter)
        cache_key = f"cache_{uuid.uuid4().hex[:12]}"

        # Compute embedding vector for semantic matching
        query_vec = embed_query(query).tolist()

        # Extract citation chunk and document lineage
        chunk_ids = [c.chunk_id for c in answer_response.citations if c.chunk_id]
        doc_ids = list(set(c.doc_id for c in answer_response.citations if c.doc_id))

        cached_entry = CachedAnswer(
            cache_key=cache_key,
            query=query,
            query_embedding=query_vec,
            answer=answer_response.answer,
            citations=answer_response.citations,
            confidence=answer_response.confidence,
            decision=answer_response.decision,
            ticker_filter=ticker_filter,
            doc_type_filter=doc_type_filter,
            dependent_chunk_ids=chunk_ids,
            dependent_doc_ids=doc_ids,
            ttl_seconds=ttl_seconds,
        )

        # 1. Save entry and index in storage backend
        backend.save_cache_entry(cached_entry, scope=scope, query_hash=q_hash)

        # 2. Register lineage dependencies with LineageManager
        from app.cache.lineage import get_lineage_manager

        manager = get_lineage_manager()
        manager.register_lineage(
            cache_key=cache_key,
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
        )

        logger.info(
            "Cached answer for query '%s' (key: %s, chunks: %d, docs: %d)",
            query,
            cache_key,
            len(chunk_ids),
            len(doc_ids),
        )
        return cache_key

    def get_stats(self) -> CacheStats:
        """Return cache performance statistics."""
        total = self._stats["total_requests"]
        hits = self._stats["l1_exact_hits"] + self._stats["l2_semantic_hits"]
        rate = (hits / total) if total > 0 else 0.0

        return CacheStats(
            total_entries=len(backend._mem_entries),
            total_hits=hits,
            l1_exact_hits=self._stats["l1_exact_hits"],
            l2_semantic_hits=self._stats["l2_semantic_hits"],
            misses=self._stats["misses"],
            hit_rate=round(rate, 4),
            storage_backend="in-memory" if backend._use_fallback else "redis",
        )

    def clear(self) -> None:
        """Reset stats and clear storage backend."""
        self._stats = {
            "total_requests": 0,
            "l1_exact_hits": 0,
            "l2_semantic_hits": 0,
            "misses": 0,
        }
        backend.reset_cache_backend()


# Module-level singleton
_default_cache: Optional[SemanticCache] = None


def get_semantic_cache() -> SemanticCache:
    """Get or create singleton SemanticCache instance."""
    global _default_cache
    if _default_cache is None:
        _default_cache = SemanticCache()
    return _default_cache
