"""
Storage backend for the Semantic Cache and Lineage subsystem.

Supports Redis with an automated, transparent in-memory fallback for testing
or offline environments without Docker.
"""

import json
import logging
from typing import Optional, Set, List, Dict, Any

from app.cache.models import CachedAnswer

logger = logging.getLogger(__name__)

# Redis client singleton
_redis_client = None
_use_fallback = False

# In-memory stores (used when Redis is unavailable or forced in tests)
_mem_entries: Dict[str, str] = {}  # cache_key -> JSON string of CachedAnswer
_mem_exact: Dict[str, str] = {}  # f"{scope}:{query_hash}" -> cache_key
_mem_scopes: Dict[str, Set[str]] = {}  # scope -> set of cache_keys

# Lineage in-memory stores
_mem_lineage_chunk: Dict[str, Set[str]] = {}  # chunk_id -> set of cache_keys
_mem_lineage_doc: Dict[str, Set[str]] = {}  # doc_id -> set of cache_keys
_mem_doc_chunks: Dict[str, Set[str]] = {}  # doc_id -> set of chunk_ids
_mem_lineage_entry: Dict[str, Dict[str, List[str]]] = {}  # cache_key -> {chunk_ids, doc_ids}

# Redis key prefixes
PREFIX_ENTRY = "cache:entry:"
PREFIX_EXACT = "cache:exact:"
PREFIX_SCOPE = "cache:scope:"
PREFIX_LINEAGE_CHUNK = "lineage:chunk:"
PREFIX_LINEAGE_DOC = "lineage:doc:"
PREFIX_DOC_CHUNKS = "lineage:doc_chunks:"
PREFIX_LINEAGE_ENTRY = "lineage:entry:"


def _get_redis():
    """Lazily connect to Redis. Falls back to in-memory store if connection fails."""
    global _redis_client, _use_fallback

    if _use_fallback:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        from app.config import settings

        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
            socket_connect_timeout=1.0,
        )
        client.ping()
        _redis_client = client
        logger.info("Connected to Redis for semantic caching at %s:%d", settings.redis_host, settings.redis_port)
        return _redis_client
    except Exception as e:
        logger.warning(
            "Redis unavailable for cache (%s), falling back to in-memory cache backend.",
            e,
        )
        _use_fallback = True
        return None


def force_memory_fallback() -> None:
    """Force in-memory fallback mode for unit testing."""
    global _use_fallback, _redis_client
    _use_fallback = True
    _redis_client = None


def reset_cache_backend() -> None:
    """Clear all cache and lineage data across Redis and in-memory stores."""
    global _mem_entries, _mem_exact, _mem_scopes
    global _mem_lineage_chunk, _mem_lineage_doc, _mem_doc_chunks, _mem_lineage_entry
    global _use_fallback, _redis_client

    _mem_entries.clear()
    _mem_exact.clear()
    _mem_scopes.clear()
    _mem_lineage_chunk.clear()
    _mem_lineage_doc.clear()
    _mem_doc_chunks.clear()
    _mem_lineage_entry.clear()

    if _use_fallback:
        return

    r = _get_redis()
    if r is not None:
        try:
            prefixes = [
                PREFIX_ENTRY,
                PREFIX_EXACT,
                PREFIX_SCOPE,
                PREFIX_LINEAGE_CHUNK,
                PREFIX_LINEAGE_DOC,
                PREFIX_DOC_CHUNKS,
                PREFIX_LINEAGE_ENTRY,
            ]
            for prefix in prefixes:
                cursor = 0
                while True:
                    cursor, keys = r.scan(cursor, match=f"{prefix}*", count=200)
                    if keys:
                        r.delete(*keys)
                    if cursor == 0:
                        break
        except Exception as e:
            logger.error("Redis cache cleanup failed: %s", e)


# ============================================================================
# Cache Entry CRUD
# ============================================================================

def save_cache_entry(entry: CachedAnswer, scope: str, query_hash: str) -> None:
    """Save CachedAnswer and index it under exact match and scope sets."""
    entry_json = entry.model_dump_json()
    cache_key = entry.cache_key

    # Update in-memory fallback
    _mem_entries[cache_key] = entry_json
    _mem_exact[f"{scope}:{query_hash}"] = cache_key
    _mem_scopes.setdefault(scope, set()).add(cache_key)

    r = _get_redis()
    if r is None:
        return

    try:
        pipe = r.pipeline()
        pipe.setex(f"{PREFIX_ENTRY}{cache_key}", entry.ttl_seconds, entry_json)
        pipe.setex(f"{PREFIX_EXACT}{scope}:{query_hash}", entry.ttl_seconds, cache_key)
        pipe.sadd(f"{PREFIX_SCOPE}{scope}", cache_key)
        pipe.execute()
    except Exception as e:
        logger.error("Redis error saving cache entry %s: %s", cache_key, e)


def get_cache_entry(cache_key: str) -> Optional[CachedAnswer]:
    """Retrieve CachedAnswer by unique cache_key."""
    r = _get_redis()
    if r is None:
        data = _mem_entries.get(cache_key)
        return CachedAnswer.model_validate_json(data) if data else None

    try:
        data = r.get(f"{PREFIX_ENTRY}{cache_key}")
        if data is None:
            # Check in-memory fallback
            mem_data = _mem_entries.get(cache_key)
            return CachedAnswer.model_validate_json(mem_data) if mem_data else None
        return CachedAnswer.model_validate_json(data)
    except Exception as e:
        logger.error("Redis error reading cache entry %s: %s", cache_key, e)
        mem_data = _mem_entries.get(cache_key)
        return CachedAnswer.model_validate_json(mem_data) if mem_data else None


def get_cache_key_by_exact(scope: str, query_hash: str) -> Optional[str]:
    """Retrieve cache_key from L1 exact match index."""
    exact_key = f"{scope}:{query_hash}"
    r = _get_redis()
    if r is None:
        return _mem_exact.get(exact_key)

    try:
        val = r.get(f"{PREFIX_EXACT}{exact_key}")
        return val if val else _mem_exact.get(exact_key)
    except Exception as e:
        logger.error("Redis error checking exact cache for %s: %s", exact_key, e)
        return _mem_exact.get(exact_key)


def get_cache_keys_in_scope(scope: str) -> List[str]:
    """Retrieve all cache_keys indexed under a given filter scope."""
    r = _get_redis()
    if r is None:
        return list(_mem_scopes.get(scope, set()))

    try:
        keys = list(r.smembers(f"{PREFIX_SCOPE}{scope}"))
        if not keys and scope in _mem_scopes:
            keys = list(_mem_scopes[scope])
        return keys
    except Exception as e:
        logger.error("Redis error getting scope keys for %s: %s", scope, e)
        return list(_mem_scopes.get(scope, set()))


def delete_cache_entry(cache_key: str, scope: Optional[str] = None, query_hash: Optional[str] = None) -> None:
    """Evict a cache entry and remove its index pointers."""
    _mem_entries.pop(cache_key, None)
    if scope:
        if scope in _mem_scopes:
            _mem_scopes[scope].discard(cache_key)
        if query_hash:
            _mem_exact.pop(f"{scope}:{query_hash}", None)
    else:
        # Search and clean from all scopes
        for s, s_keys in _mem_scopes.items():
            s_keys.discard(cache_key)
        keys_to_del = [k for k, v in _mem_exact.items() if v == cache_key]
        for k in keys_to_del:
            _mem_exact.pop(k, None)

    r = _get_redis()
    if r is None:
        return

    try:
        pipe = r.pipeline()
        pipe.delete(f"{PREFIX_ENTRY}{cache_key}")
        if scope and query_hash:
            pipe.delete(f"{PREFIX_EXACT}{scope}:{query_hash}")
        if scope:
            pipe.srem(f"{PREFIX_SCOPE}{scope}", cache_key)
        pipe.execute()
    except Exception as e:
        logger.error("Redis error deleting cache entry %s: %s", cache_key, e)

