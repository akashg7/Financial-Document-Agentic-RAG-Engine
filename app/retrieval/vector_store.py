"""
Qdrant vector store client for upserting and searching document chunks.

WHY Qdrant over Pinecone/Weaviate/Chroma:
- Free and open-source, runs locally in Docker — zero cost.
- Native support for both dense AND sparse vectors in the same collection,
  which we'll use in Day 3 for hybrid search.
- Payload filtering (filter by ticker, doc_type) without a separate metadata DB.
- gRPC port (6334) for fast bulk operations; REST port (6333) for debugging.

WHY cosine distance:
- all-MiniLM-L6-v2 outputs L2-normalized vectors (we normalize in embedder.py).
  For normalized vectors, cosine similarity = dot product, and both are equivalent
  for ranking. Qdrant's Cosine distance is 1 - cosine_similarity, so lower = better.
- We report scores as similarity (1 - distance) so higher = better for the user.

COLLECTION SCHEMA:
- Vector: "dense" — 384 dimensions, Cosine distance
- Payload fields:
    - chunk_id (keyword) — for dedup lookups
    - doc_id (keyword) — for document-level filtering
    - ticker (keyword) — for ticker-level filtering
    - doc_type (keyword) — for filing type filtering
    - content_hash (keyword) — for cross-batch dedup
    - text (text) — the chunk content (stored for retrieval, not indexed for search)
    - All other ChunkMetadata fields stored as payload
"""

import logging
import time
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PayloadSchemaType,
)

from app.config import settings
from app.schemas import Chunk, SearchResult
from app.retrieval.embedder import embed_texts, embed_query

logger = logging.getLogger(__name__)

# Module-level client singleton — initialized via init_qdrant_client()
_client: Optional[QdrantClient] = None

# Embedding dimension for all-MiniLM-L6-v2
EMBEDDING_DIM = 384


def get_qdrant_client() -> QdrantClient:
    """
    Get or create the Qdrant client singleton.

    WHY a singleton instead of creating a client per request:
    - QdrantClient maintains an HTTP connection pool internally.
      Creating a new client per request wastes connections and adds latency.
    - The singleton is initialized once at startup via init_qdrant_client().
    """
    global _client
    if _client is None:
        _client = QdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
        )
        logger.info(
            "Created Qdrant client connecting to %s:%d",
            settings.qdrant_host,
            settings.qdrant_port,
        )
    return _client


def set_qdrant_client(client: QdrantClient) -> None:
    """
    Inject a custom Qdrant client (used in tests to inject mocks).

    WHY dependency injection instead of monkeypatching:
    - Explicit is better than implicit. Tests call set_qdrant_client(mock_client)
      before running, and it's obvious what's happening.
    """
    global _client
    _client = client


def init_collection() -> None:
    """
    Create the Qdrant collection if it doesn't already exist.

    Called once during FastAPI lifespan startup. Idempotent — safe to call
    multiple times (e.g., on app restart).

    WHY we create indexes on keyword fields:
    - Payload indexes let Qdrant filter BEFORE vector search, not after.
      Without an index on 'ticker', a filtered search would scan all vectors,
      compute similarity, then filter — O(n). With an index, it scans only
      matching vectors — much faster for large collections.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    # Check if collection already exists
    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if collection_name in existing_names:
        logger.info("Qdrant collection '%s' already exists", collection_name)
        return

    # Create collection with dense vector config
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=EMBEDDING_DIM,
            distance=Distance.COSINE,
        ),
    )
    logger.info(
        "Created Qdrant collection '%s' (dim=%d, distance=Cosine)",
        collection_name,
        EMBEDDING_DIM,
    )

    # Create payload indexes for filtered search
    for field_name in ["ticker", "doc_type", "doc_id", "content_hash"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    logger.info("Created payload indexes on ticker, doc_type, doc_id, content_hash")


def upsert_chunks(chunks: list[Chunk], batch_size: int = 64) -> int:
    """
    Embed and upsert chunks into Qdrant.

    Steps:
    1. Extract text from each chunk.
    2. Batch-embed all texts using sentence-transformers.
    3. Build PointStruct objects with vectors + payload.
    4. Upsert to Qdrant in batches.

    Args:
        chunks: List of Chunk objects (text + metadata).
        batch_size: Number of chunks to embed and upsert at once.

    Returns:
        Number of points upserted.

    WHY upsert (not insert):
    - If we re-ingest a chunk with the same ID, upsert overwrites it cleanly.
      Insert would fail with a duplicate key error.
    - Combined with content-hash dedup, this means: new content gets inserted,
      identical content is skipped by dedup, and updated content (same chunk_id
      but different hash) gets upserted.
    """
    if not chunks:
        return 0

    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    total_upserted = 0

    start = time.time()

    # Process in batches to control memory usage
    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # Step 1: Extract texts
        texts = [chunk.text for chunk in batch]

        # Step 2: Embed the batch
        embeddings = embed_texts(texts, batch_size=batch_size)

        # Step 3: Build Qdrant points
        points = []
        for chunk, embedding in zip(batch, embeddings):
            # Use chunk_id as the Qdrant point ID (we need a deterministic ID
            # so re-upserts overwrite the same point)
            # Qdrant accepts string UUIDs or integers. We'll use the chunk_id hash
            # as a stable integer ID.
            point_id = _chunk_id_to_int(chunk.metadata.chunk_id)

            payload = {
                "text": chunk.text,
                "chunk_id": chunk.metadata.chunk_id,
                "doc_id": chunk.metadata.doc_id,
                "doc_type": chunk.metadata.doc_type.value,
                "ticker": chunk.metadata.ticker,
                "company_name": chunk.metadata.company_name,
                "filing_date": chunk.metadata.filing_date,
                "fiscal_year": chunk.metadata.fiscal_year,
                "fiscal_quarter": chunk.metadata.fiscal_quarter,
                "section": chunk.metadata.section,
                "page": chunk.metadata.page,
                "chunk_index": chunk.metadata.chunk_index,
                "content_hash": chunk.metadata.content_hash,
                "source_url": chunk.metadata.source_url,
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload=payload,
                )
            )

        # Step 4: Upsert to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=points,
        )
        total_upserted += len(points)
        logger.debug(
            "Upserted batch %d-%d (%d points)",
            batch_start,
            batch_start + len(batch),
            len(points),
        )

    elapsed = time.time() - start
    logger.info(
        "Upserted %d points to '%s' in %.2fs",
        total_upserted,
        collection_name,
        elapsed,
    )
    return total_upserted


def search_dense(
    query: str,
    top_k: int = 10,
    ticker_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
) -> list[SearchResult]:
    """
    Perform dense (semantic) vector search.

    Steps:
    1. Embed the query using the same model used for ingestion.
    2. Search Qdrant with optional payload filters.
    3. Convert results to SearchResult objects.

    Args:
        query: Natural language search query.
        top_k: Number of results to return.
        ticker_filter: If set, only return chunks from this ticker.
        doc_type_filter: If set, only return chunks of this document type.

    Returns:
        List of SearchResult objects sorted by relevance (highest first).

    WHY we filter via Qdrant payload filters instead of post-filtering:
    - Post-filtering: search top_k=100, then filter to AAPL → might get 3 results.
    - Payload filtering: Qdrant only searches AAPL vectors → guaranteed top_k results.
    - This is especially important when one ticker dominates the collection.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    # Build filter conditions
    must_conditions = []
    if ticker_filter:
        must_conditions.append(
            FieldCondition(key="ticker", match=MatchValue(value=ticker_filter))
        )
    if doc_type_filter:
        must_conditions.append(
            FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter))
        )

    query_filter = Filter(must=must_conditions) if must_conditions else None

    # Embed the query
    query_vector = embed_query(query)

    # Search Qdrant
    start = time.time()
    results = client.query_points(
        collection_name=collection_name,
        query=query_vector.tolist(),
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )
    elapsed = time.time() - start

    logger.info(
        "Dense search for '%s' returned %d results in %.2fms",
        query[:50],
        len(results.points),
        elapsed * 1000,
    )

    # Convert to our SearchResult schema
    search_results = []
    for point in results.points:
        payload = point.payload or {}
        search_results.append(
            SearchResult(
                chunk_id=payload.get("chunk_id", ""),
                text=payload.get("text", ""),
                score=point.score,
                metadata={
                    k: v
                    for k, v in payload.items()
                    if k != "text"  # Don't duplicate text in metadata
                },
            )
        )

    return search_results


def get_existing_hashes() -> set[str]:
    """
    Retrieve all content_hash values from Qdrant for cross-batch dedup.

    WHY we query Qdrant instead of maintaining an in-memory set:
    - The in-memory set from Day 1 is lost on restart. Qdrant is persistent.
    - This function is called once at the start of each ingestion job, so
      the overhead of one scroll query is acceptable.

    For large collections (>100K chunks), this would be replaced with a
    Bloom filter or Redis set for O(1) lookups.
    """
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name

    hashes = set()
    offset = None

    # Scroll through all points, fetching only the content_hash field
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=1000,
            offset=offset,
            with_payload=["content_hash"],
            with_vectors=False,
        )

        for point in results:
            if point.payload and "content_hash" in point.payload:
                hashes.add(point.payload["content_hash"])

        if offset is None:
            break

    logger.info("Retrieved %d existing content hashes from Qdrant", len(hashes))
    return hashes


def _chunk_id_to_int(chunk_id: str) -> int:
    """
    Convert a chunk_id string to a stable integer for use as Qdrant point ID.

    WHY integer IDs over UUIDs:
    - Qdrant supports both, but integer IDs are more memory-efficient.
    - We use a hash to get a deterministic mapping: same chunk_id always
      maps to the same integer, enabling idempotent upserts.
    - We mask to 63 bits to stay within Qdrant's unsigned 64-bit range
      while avoiding sign issues in Python.
    """
    import hashlib

    hash_bytes = hashlib.md5(chunk_id.encode()).digest()
    return int.from_bytes(hash_bytes[:8], "big") & 0x7FFFFFFFFFFFFFFF
