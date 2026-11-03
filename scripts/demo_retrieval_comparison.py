"""
Demo script comparing Dense vs Sparse (BM25) vs Hybrid (RRF) retrieval.

This script demonstrates concrete scenarios where each retrieval strategy wins:
1. Scenario A — DENSE WINS:
   Query uses semantic concepts, synonyms, and paraphrasing with zero literal keyword overlap.
   Example: "top-line growth and expansion across cloud infrastructure"
   (BM25 fails because document says "Azure revenues surged by 29 percent")

2. Scenario B — SPARSE (BM25) WINS:
   Query uses exact regulatory codes, accounting terms, or specific identifiers.
   Example: "Item 1A risk factors related to cybersecurity"
   (Dense embedding can be vague or match general risk text; BM25 strictly matches "Item 1A")

3. Scenario C — HYBRID (RRF) WINS:
   Query contains both a specific keyword and a semantic question.
   Example: "iPhone sales slowdown due to foreign currency exchange headwinds"
   (Dense captures the macro headwind semantics; BM25 anchors the exact "iPhone" and "currency" entities)

Usage:
    python scripts/demo_retrieval_comparison.py
"""

import sys
import os

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qdrant_client import QdrantClient
from app.schemas import Chunk, ChunkMetadata, DocumentType
from app.retrieval.vector_store import (
    set_qdrant_client,
    init_collection,
    upsert_chunks,
    search_dense,
    search_sparse,
)
from app.retrieval.sparse import fit_bm25, encode_sparse_document


DEMO_DOCUMENTS = [
    {
        "id": "MSFT_10K_chunk_01",
        "ticker": "MSFT",
        "text": "Intelligent Cloud segment revenue was $87.9 billion, driven by Azure revenues surging 29 percent year-over-year.",
    },
    {
        "id": "MSFT_10K_chunk_02",
        "ticker": "MSFT",
        "text": "Item 1A. Risk Factors. Cybersecurity incidents, ransomware, and unauthorized network intrusions pose material operational risks.",
    },
    {
        "id": "AAPL_10K_chunk_01",
        "ticker": "AAPL",
        "text": "Total net sales expanded across international markets, though foreign currency exchange fluctuations created significant unfavorable top-line headwinds.",
    },
    {
        "id": "AAPL_10K_chunk_02",
        "ticker": "AAPL",
        "text": "iPhone net sales declined 2% due to foreign exchange rate volatility and supply chain component constraints.",
    },
    {
        "id": "GOOGL_10K_chunk_01",
        "ticker": "GOOGL",
        "text": "Google Cloud delivered sustained operating profitability with enterprise AI platform adoption accelerating across Fortune 500 customers.",
    },
]


def setup_demo_index():
    """Seed an in-memory Qdrant instance with sample financial documents."""
    client = QdrantClient(":memory:")
    set_qdrant_client(client)
    init_collection()

    chunks = []
    texts = [d["text"] for d in DEMO_DOCUMENTS]
    fit_bm25(texts)
    sparse_vecs = [encode_sparse_document(t) for t in texts]

    for i, doc in enumerate(DEMO_DOCUMENTS):
        chunk = Chunk(
            text=doc["text"],
            metadata=ChunkMetadata(
                chunk_id=doc["id"],
                doc_id=doc["id"].rsplit("_", 1)[0],
                doc_type=DocumentType.SEC_10K,
                ticker=doc["ticker"],
                chunk_index=i,
                content_hash=f"hash_{i}",
            ),
        )
        chunks.append(chunk)

    upsert_chunks(chunks, sparse_vectors=sparse_vecs)
    return client


def display_results(method_name: str, results: list):
    print(f"\n  [{method_name}] Top Results:")
    if not results:
        print("    (No results returned)")
        return
    for rank, r in enumerate(results[:3], start=1):
        print(f"    #{rank} [Score: {r.score:.4f}] [{r.chunk_id}] {r.text[:80]}...")


def run_demo():
    print("=" * 80)
    print("FINANCIAL DOCUMENT RAG: RETRIEVAL METHOD COMPARISON DEMO")
    print("Dense (all-MiniLM-L6-v2) vs Sparse (BM25) vs Hybrid (RRF)")
    print("=" * 80)

    setup_demo_index()

    test_queries = [
        {
            "scenario": "SCENARIO 1: Semantic Paraphrase (Dense Wins)",
            "query": "top-line growth and expansion across cloud infrastructure",
            "explanation": "Query uses 'expansion' and 'cloud infrastructure' without mentioning 'Azure' or 'surging 29 percent'. Dense captures semantic similarity; Sparse struggles without lexical match.",
        },
        {
            "scenario": "SCENARIO 2: Exact Regulatory Header / Code (Sparse Wins)",
            "query": "Item 1A risk factors cybersecurity",
            "explanation": "Query targets the specific SEC section header 'Item 1A'. BM25 locks onto exact term frequencies, while dense search may dilute the section identifier with generic risk passages.",
        },
        {
            "scenario": "SCENARIO 3: Keyword Entity + Semantic Context (Hybrid Wins)",
            "query": "iPhone sales slowdown due to foreign currency exchange headwinds",
            "explanation": "Needs exact entity matching for 'iPhone' (avoiding generic cloud chunks) and semantic alignment with currency headwinds. Hybrid RRF combines both strengths.",
        },
    ]

    for item in test_queries:
        print("\n" + "-" * 80)
        print(f"🔹 {item['scenario']}")
        print(f"   Query: \"{item['query']}\"")
        print(f"   Context: {item['explanation']}")

        dense_res = search_dense(item["query"], top_k=3)
        sparse_res = search_sparse(item["query"], top_k=3)

        display_results("DENSE SEARCH (Cosine)", dense_res)
        display_results("SPARSE SEARCH (BM25)", sparse_res)

        # Attempt Hybrid Search via RRF if implemented
        try:
            from app.retrieval.rrf import reciprocal_rank_fusion
            hybrid_res = reciprocal_rank_fusion([dense_res, sparse_res], k=60, top_n=3)
            display_results("HYBRID SEARCH (RRF)", hybrid_res)
        except NotImplementedError:
            print("\n  [HYBRID SEARCH (RRF)]")
            print("    ⚠️  app/retrieval/rrf.py is not yet implemented (Hand-write exercise #1).")
            print("    Run pytest tests/test_rrf.py after implementing to verify RRF fusion.")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_demo()

