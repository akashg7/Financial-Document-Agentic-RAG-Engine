# Retrieval subpackage — dense, sparse, and hybrid search
#
# Day 2: embedder.py (sentence-transformers) + vector_store.py (Qdrant dense search)
# Day 3: sparse search (BM25 via Qdrant sparse vectors) + hybrid (RRF)

from app.retrieval.embedder import embed_texts, embed_query, get_embedding_dimension
from app.retrieval.vector_store import (
    init_collection,
    upsert_chunks,
    search_dense,
    get_existing_hashes,
)

__all__ = [
    "embed_texts",
    "embed_query",
    "get_embedding_dimension",
    "init_collection",
    "upsert_chunks",
    "search_dense",
    "get_existing_hashes",
]
