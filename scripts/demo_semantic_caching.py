#!/usr/bin/env python3
"""
Demonstration and benchmark script for Day 8: Semantic Caching & Cache Lineage.

Evaluates:
1. Cold Cache Miss: Full LangGraph agentic loop execution (~800-1500 ms).
2. L1 Exact Match Hit: $O(1)$ SHA-256 hash lookup (< 1 ms latency).
3. L2 Semantic Match Hit: Dense cosine similarity vector match (< 5 ms latency).
4. Surgical Lineage Invalidation: Evicts dependent queries when filing/chunks are updated.

Saves empirical benchmark metrics to results/cache_benchmark_<timestamp>.json.

Usage:
    python scripts/demo_semantic_caching.py
"""

import sys
import time
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import SearchResult, Citation, AnswerResponse
from app.cache.semantic_cache import get_semantic_cache
from app.cache.lineage import get_lineage_manager
from app.agent.graph import run_financial_agent, build_agent_graph


def banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def run_benchmark():
    banner("DAY 8: SEMANTIC CACHING & LINEAGE INVALIDATION BENCHMARK")

    cache = get_semantic_cache()
    lineage = get_lineage_manager()
    cache.clear()

    graph = build_agent_graph()

    chunk_revenue = SearchResult(
        chunk_id="AAPL_10-K_2023_chunk_0042",
        text="Total net sales for fiscal year 2023 were $383,285 million, compared to $394,328 million in fiscal year 2022.",
        score=0.96,
        metadata={"ticker": "AAPL", "doc_id": "AAPL_10-K_2023", "doc_type": "10-K", "section": "Item 7"},
    )

    metrics = []

    with patch("app.agent.nodes.search_dense", return_value=[chunk_revenue]), \
         patch("app.agent.nodes.search_sparse", return_value=[chunk_revenue]), \
         patch("app.agent.nodes.rerank", return_value=[chunk_revenue]):

        # --------------------------------------------------------------------
        # Pass 1: Cold Cache Miss
        # --------------------------------------------------------------------
        banner("PASS 1: Cold Cache Miss (Full Agentic Execution)")
        q1 = "What was Apple's total net sales in fiscal 2023?"
        print(f"Query: \"{q1}\"")

        t0 = time.time()
        resp1 = run_financial_agent(query=q1, ticker_filter="AAPL", doc_type_filter="10-K", graph=graph, use_cache=True)
        cold_latency_ms = (time.time() - t0) * 1000

        print(f"[Cache Status]: {'HIT' if resp1.cached else 'MISS (Executed RAG)'}")
        print(f"[Latency]:      {cold_latency_ms:.2f} ms")
        print(f"[Answer]:       {resp1.answer[:90]}...")
        print(f"[Citations]:    {len(resp1.citations)} chunk citations bound")

        metrics.append({
            "pass": 1,
            "type": "Cold Miss",
            "query": q1,
            "cached": resp1.cached,
            "latency_ms": round(cold_latency_ms, 2),
            "speedup": "1.0x (baseline)",
        })

        # --------------------------------------------------------------------
        # Pass 2: L1 Exact String Match Hit (< 1 ms)
        # --------------------------------------------------------------------
        banner("PASS 2: L1 Exact String Match Cache Hit (< 1 ms)")
        q2 = "   what was apple's total net sales in fiscal 2023?   "
        print(f"Query (case/whitespace variant): \"{q2}\"")

        t0 = time.time()
        resp2 = run_financial_agent(query=q2, ticker_filter="AAPL", doc_type_filter="10-K", graph=graph, use_cache=True)
        l1_latency_ms = (time.time() - t0) * 1000

        l1_speedup = cold_latency_ms / max(l1_latency_ms, 0.01)
        print(f"[Cache Status]: {'HIT (L1 Exact Match)' if resp2.cached else 'MISS'}")
        print(f"[Latency]:      {l1_latency_ms:.2f} ms")
        print(f"[Speedup]:      {l1_speedup:.1f}x faster than cold RAG pipeline")
        print(f"[Answer]:       {resp2.answer[:90]}...")

        metrics.append({
            "pass": 2,
            "type": "L1 Exact Hit",
            "query": q2.strip(),
            "cached": resp2.cached,
            "latency_ms": round(l1_latency_ms, 2),
            "speedup": f"{l1_speedup:.1f}x",
        })

        # --------------------------------------------------------------------
        # Pass 3: L2 Semantic Similarity Match Hit (< 5 ms)
        # --------------------------------------------------------------------
        banner("PASS 3: L2 Semantic Similarity Cache Hit (< 5 ms)")
        q3 = "What was Apple total net sales in fiscal 2023?"
        print(f"Paraphrased Query: \"{q3}\"")

        t0 = time.time()
        resp3 = run_financial_agent(query=q3, ticker_filter="AAPL", doc_type_filter="10-K", graph=graph, use_cache=True)
        l2_latency_ms = (time.time() - t0) * 1000

        l2_speedup = cold_latency_ms / max(l2_latency_ms, 0.01)
        print(f"[Cache Status]: {'HIT (L2 Semantic Match)' if resp3.cached else 'MISS'}")
        print(f"[Latency]:      {l2_latency_ms:.2f} ms")
        print(f"[Speedup]:      {l2_speedup:.1f}x faster than cold RAG pipeline")
        print(f"[Answer]:       {resp3.answer[:90]}...")

        metrics.append({
            "pass": 3,
            "type": "L2 Semantic Hit",
            "query": q3,
            "cached": resp3.cached,
            "latency_ms": round(l2_latency_ms, 2),
            "speedup": f"{l2_speedup:.1f}x",
        })

        # --------------------------------------------------------------------
        # Pass 4: Surgical Lineage Invalidation (Filing Re-Ingestion)
        # --------------------------------------------------------------------
        banner("PASS 4: Surgical Lineage Invalidation (Delta Re-Ingestion)")
        doc_to_invalidate = "AAPL_10-K_2023"
        print(f"Triggering invalidation for document: '{doc_to_invalidate}'")

        affected = lineage.get_affected_queries(doc_id=doc_to_invalidate)
        print(f"Blast Radius Impact Analysis: {len(affected)} cached query dependent:")
        for q in affected:
            print(f"  - \"{q}\"")

        evicted = lineage.invalidate_by_document(doc_to_invalidate)
        print(f"✓ Evicted {len(evicted)} cache keys from Redis & memory backend.")

        # Re-querying must now result in a clean cache MISS
        t0 = time.time()
        resp4 = run_financial_agent(query=q1, ticker_filter="AAPL", doc_type_filter="10-K", graph=graph, use_cache=True)
        post_invalidation_latency_ms = (time.time() - t0) * 1000

        print(f"\n[Post-Invalidation Query]: \"{q1}\"")
        print(f"[Cache Status]:            {'HIT' if resp4.cached else 'MISS (Correctly Evicted - Re-executed)'}")
        print(f"[Latency]:                 {post_invalidation_latency_ms:.2f} ms")

        metrics.append({
            "pass": 4,
            "type": "Post-Invalidation Re-query",
            "query": q1,
            "cached": resp4.cached,
            "latency_ms": round(post_invalidation_latency_ms, 2),
            "evicted_keys": len(evicted),
        })

    # Cache Stats
    stats = cache.get_stats()
    banner("FINAL CACHE TELEMETRY")
    print(f"Total Requests:   {stats.total_hits + stats.misses}")
    print(f"L1 Exact Hits:    {stats.l1_exact_hits}")
    print(f"L2 Semantic Hits: {stats.l2_semantic_hits}")
    print(f"Misses:           {stats.misses}")
    print(f"Hit Rate:         {stats.hit_rate * 100:.1f}%")
    print(f"Storage Backend:  {stats.storage_backend}")

    # Save results
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = Path(f"results/cache_benchmark_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "metrics": metrics,
                "cache_stats": stats.model_dump(),
            },
            f,
            indent=2,
        )
    print(f"\n✓ Saved empirical cache benchmark results to {out_path}")


if __name__ == "__main__":
    run_benchmark()
