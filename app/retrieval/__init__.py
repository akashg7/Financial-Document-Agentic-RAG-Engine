# Retrieval subpackage — dense, sparse, and hybrid search
#
# Day 2: embedder.py (sentence-transformers) + vector_store.py (Qdrant dense search)
# Day 3: sparse.py (BM25 sparse vectors) + rrf.py (Reciprocal Rank Fusion) + hybrid search

from app.retrieval.embedder import embed_texts, embed_query, get_embedding_dimension
from app.retrieval.vector_store import (
    init_collection,
    upsert_chunks,
    search_dense,
    search_sparse,
    get_existing_hashes,
)
from app.retrieval.sparse import (
    fit_bm25,
    encode_sparse_document,
    encode_sparse_query,
    tokenize,
)
from app.retrieval.rrf import reciprocal_rank_fusion

__all__ = [
    "embed_texts",
    "embed_query",
    "get_embedding_dimension",
    "init_collection",
    "upsert_chunks",
    "search_dense",
    "search_sparse",
    "get_existing_hashes",
    "fit_bm25",
    "encode_sparse_document",
    "encode_sparse_query",
    "tokenize",
    "reciprocal_rank_fusion",
]
