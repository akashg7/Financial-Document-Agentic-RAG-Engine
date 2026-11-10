"""
Reference Solution for Hand-Write Piece #4: Cache Lineage & Invalidation Manager
File: app/cache/lineage.py

The LineageManager maintains a bipartite DAG:
  Document (doc_id) -> Chunks (chunk_id) -> Cached Answers (cache_key)

When SEC documents are amended or re-ingested, LineageManager traverses reverse
indexes to evict exactly the affected cached answers, eliminating stale data
while preserving valid cache entries.
"""

import json
import logging
from typing import Optional, List, Dict, Set, Any

from app.cache import backend
from app.cache.backend import (
    _get_redis,
    _mem_lineage_chunk,
    _mem_lineage_doc,
    _mem_doc_chunks,
    _mem_lineage_entry,
    PREFIX_LINEAGE_CHUNK,
    PREFIX_LINEAGE_DOC,
    PREFIX_DOC_CHUNKS,
    PREFIX_LINEAGE_ENTRY,
)

logger = logging.getLogger(__name__)


class LineageManager:
    """
    Manages dependency lineage between SEC filings, chunks, and cached answers.
    Enables surgical, automated cache invalidation upon document updates or delta re-ingest.
    """

    def register_lineage(
        self,
        cache_key: str,
        chunk_ids: List[str],
        doc_ids: List[str],
    ) -> None:
        """
        Record dependency edges for a cached answer.

        Maintains:
        - cache_key -> {chunk_ids, doc_ids}
        - chunk_id -> {cache_keys}
        - doc_id -> {cache_keys}
        """
        # Update in-memory fallback
        _mem_lineage_entry[cache_key] = {
            "chunk_ids": list(chunk_ids),
            "doc_ids": list(doc_ids),
        }

        for cid in chunk_ids:
            _mem_lineage_chunk.setdefault(cid, set()).add(cache_key)

        for did in doc_ids:
            _mem_lineage_doc.setdefault(did, set()).add(cache_key)

        r = _get_redis()
        if r is None:
            return

        try:
            pipe = r.pipeline()
            entry_data = json.dumps({"chunk_ids": chunk_ids, "doc_ids": doc_ids})
            pipe.set(f"{PREFIX_LINEAGE_ENTRY}{cache_key}", entry_data)

            for cid in chunk_ids:
                pipe.sadd(f"{PREFIX_LINEAGE_CHUNK}{cid}", cache_key)

            for did in doc_ids:
                pipe.sadd(f"{PREFIX_LINEAGE_DOC}{did}", cache_key)

            pipe.execute()
        except Exception as e:
            logger.error("Redis error registering lineage for %s: %s", cache_key, e)

    def register_document_chunks(
        self,
        doc_id: str,
        chunk_ids: List[str],
    ) -> None:
        """Record the set of chunks that belong to a document."""
        if not chunk_ids:
            return

        _mem_doc_chunks.setdefault(doc_id, set()).update(chunk_ids)

        r = _get_redis()
        if r is None:
            return

        try:
            r.sadd(f"{PREFIX_DOC_CHUNKS}{doc_id}", *chunk_ids)
        except Exception as e:
            logger.error("Redis error registering doc chunks for %s: %s", doc_id, e)

    def invalidate_by_chunk_ids(
        self,
        chunk_ids: List[str],
    ) -> List[str]:
        """
        Invalidate all cached answers that depend on any of the specified chunk_ids.

        Returns:
            List of evicted cache_keys.
        """
        if not chunk_ids:
            return []

        affected_keys: Set[str] = set()

        # 1. Collect affected keys from in-memory reverse index
        for cid in chunk_ids:
            if cid in _mem_lineage_chunk:
                affected_keys.update(_mem_lineage_chunk[cid])

        # 2. Collect from Redis if available
        r = _get_redis()
        if r is not None:
            try:
                for cid in chunk_ids:
                    r_keys = r.smembers(f"{PREFIX_LINEAGE_CHUNK}{cid}")
                    if r_keys:
                        affected_keys.update(r_keys)
            except Exception as e:
                logger.error("Redis error looking up chunk lineage: %s", e)

        evicted: List[str] = []

        # 3. Evict each affected cache entry and clean reverse index pointers
        for key in affected_keys:
            lineage = self.get_lineage(key)
            c_ids = lineage.get("chunk_ids", [])
            d_ids = lineage.get("doc_ids", [])

            # Evict from backend (L1 exact index, scope set, and entry)
            backend.delete_cache_entry(key)
            evicted.append(key)

            # Clean in-memory lineage pointers
            _mem_lineage_entry.pop(key, None)
            for cid in c_ids:
                if cid in _mem_lineage_chunk:
                    _mem_lineage_chunk[cid].discard(key)
            for did in d_ids:
                if did in _mem_lineage_doc:
                    _mem_lineage_doc[did].discard(key)

            # Clean Redis lineage pointers
            if r is not None:
                try:
                    pipe = r.pipeline()
                    pipe.delete(f"{PREFIX_LINEAGE_ENTRY}{key}")
                    for cid in c_ids:
                        pipe.srem(f"{PREFIX_LINEAGE_CHUNK}{cid}", key)
                    for did in d_ids:
                        pipe.srem(f"{PREFIX_LINEAGE_DOC}{did}", key)
                    pipe.execute()
                except Exception as e:
                    logger.error("Redis error cleaning lineage for %s: %s", key, e)

        # Clean chunk keys themselves
        for cid in chunk_ids:
            _mem_lineage_chunk.pop(cid, None)
        if r is not None:
            try:
                for cid in chunk_ids:
                    r.delete(f"{PREFIX_LINEAGE_CHUNK}{cid}")
            except Exception as e:
                logger.error("Redis error deleting chunk index: %s", e)

        logger.info("Invalidated %d cache entries across %d chunks", len(evicted), len(chunk_ids))
        return sorted(evicted)

    def invalidate_by_document(
        self,
        doc_id: str,
    ) -> List[str]:
        """
        Invalidate all cached answers that depend directly on doc_id
        OR depend on any chunk belonging to doc_id.

        Returns:
            List of evicted cache_keys.
        """
        # 1. Discover all chunks belonging to doc_id
        doc_chunks: Set[str] = set()
        if doc_id in _mem_doc_chunks:
            doc_chunks.update(_mem_doc_chunks[doc_id])

        r = _get_redis()
        if r is not None:
            try:
                r_chunks = r.smembers(f"{PREFIX_DOC_CHUNKS}{doc_id}")
                if r_chunks:
                    doc_chunks.update(r_chunks)
            except Exception as e:
                logger.error("Redis error retrieving doc chunks for %s: %s", doc_id, e)

        # 2. Invalidate by chunks
        evicted_set = set(self.invalidate_by_chunk_ids(list(doc_chunks)))

        # 3. Check for any queries directly bound to doc_id
        direct_keys: Set[str] = set()
        if doc_id in _mem_lineage_doc:
            direct_keys.update(_mem_lineage_doc[doc_id])
        if r is not None:
            try:
                r_direct = r.smembers(f"{PREFIX_LINEAGE_DOC}{doc_id}")
                if r_direct:
                    direct_keys.update(r_direct)
            except Exception as e:
                logger.error("Redis error retrieving direct doc lineage for %s: %s", doc_id, e)

        for key in direct_keys:
            if key not in evicted_set:
                backend.delete_cache_entry(key)
                evicted_set.add(key)
                _mem_lineage_entry.pop(key, None)
                if r is not None:
                    try:
                        r.delete(f"{PREFIX_LINEAGE_ENTRY}{key}")
                        r.srem(f"{PREFIX_LINEAGE_DOC}{doc_id}", key)
                    except Exception as e:
                        logger.error("Redis error cleaning direct doc lineage for %s: %s", key, e)

        # 4. Clean document chunk index
        _mem_doc_chunks.pop(doc_id, None)
        _mem_lineage_doc.pop(doc_id, None)
        if r is not None:
            try:
                r.delete(f"{PREFIX_DOC_CHUNKS}{doc_id}")
                r.delete(f"{PREFIX_LINEAGE_DOC}{doc_id}")
            except Exception as e:
                logger.error("Redis error deleting doc index for %s: %s", doc_id, e)

        logger.info("Invalidated %d cache entries for document %s", len(evicted_set), doc_id)
        return sorted(list(evicted_set))

    def get_lineage(
        self,
        cache_key: str,
    ) -> Dict[str, List[str]]:
        """Return the upstream dependency graph (chunks and docs) for a cached answer."""
        if cache_key in _mem_lineage_entry:
            return _mem_lineage_entry[cache_key]

        r = _get_redis()
        if r is not None:
            try:
                data = r.get(f"{PREFIX_LINEAGE_ENTRY}{cache_key}")
                if data:
                    return json.loads(data)
            except Exception as e:
                logger.error("Redis error getting lineage for %s: %s", cache_key, e)

        return {"chunk_ids": [], "doc_ids": []}

    def get_affected_queries(
        self,
        chunk_id: Optional[str] = None,
        doc_id: Optional[str] = None,
    ) -> List[str]:
        """
        Impact analysis: returns the list of cached query strings that would be evicted
        if the specified chunk or document were modified/invalidated.
        """
        affected_keys: Set[str] = set()

        if chunk_id:
            if chunk_id in _mem_lineage_chunk:
                affected_keys.update(_mem_lineage_chunk[chunk_id])
            r = _get_redis()
            if r is not None:
                try:
                    r_keys = r.smembers(f"{PREFIX_LINEAGE_CHUNK}{chunk_id}")
                    if r_keys:
                        affected_keys.update(r_keys)
                except Exception as e:
                    logger.error("Redis error checking affected queries for chunk %s: %s", chunk_id, e)

        if doc_id:
            # Add keys from doc's chunks
            chunks = list(_mem_doc_chunks.get(doc_id, set()))
            r = _get_redis()
            if r is not None:
                try:
                    r_chunks = r.smembers(f"{PREFIX_DOC_CHUNKS}{doc_id}")
                    if r_chunks:
                        chunks.extend(r_chunks)
                except Exception as e:
                    logger.error("Redis error checking doc chunks: %s", e)

            for cid in chunks:
                if cid in _mem_lineage_chunk:
                    affected_keys.update(_mem_lineage_chunk[cid])
                if r is not None:
                    try:
                        r_k = r.smembers(f"{PREFIX_LINEAGE_CHUNK}{cid}")
                        if r_k:
                            affected_keys.update(r_k)
                    except Exception:
                        pass

            if doc_id in _mem_lineage_doc:
                affected_keys.update(_mem_lineage_doc[doc_id])
            if r is not None:
                try:
                    r_d = r.smembers(f"{PREFIX_LINEAGE_DOC}{doc_id}")
                    if r_d:
                        affected_keys.update(r_d)
                except Exception:
                    pass

        queries = []
        for key in affected_keys:
            entry = backend.get_cache_entry(key)
            if entry and entry.query:
                queries.append(entry.query)

        return sorted(list(set(queries)))

