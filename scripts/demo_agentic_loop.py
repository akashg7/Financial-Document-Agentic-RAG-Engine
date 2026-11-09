#!/usr/bin/env python3
"""
Demonstration script for Day 7: LangGraph Agentic Loop.

Demonstrates the 3 core state machine trajectories:
1. Direct Generation: High-confidence retrieval terminates immediately with 0 retries.
2. Query Rewriting Cycle: Colloquial query triggers rewrite loop, maps GAAP terminology,
   and succeeds on the second retrieval iteration (1 retry).
3. Disciplined Abstention: Speculative/unanswerable query has insufficient evidence,
   triggering a clean abstention with 0 citations and no hallucinated claims.

Usage:
    python scripts/demo_agentic_loop.py
"""

import sys
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import SearchResult, Citation
from app.agent.graph import build_agent_graph, run_financial_agent


def banner(text: str):
    print("\n" + "=" * 80)
    print(f" {text}")
    print("=" * 80)


def format_citations(citations):
    if not citations:
        return "  (None - no citations generated)"
    lines = []
    for c in citations:
        lines.append(f"  * [{c.chunk_id}] ({c.doc_id}): \"{c.quote[:75]}...\"")
    return "\n".join(lines)


def run_demo():
    banner("FINANCIAL AGENTIC RAG ENGINE — DAY 7 LANGGRAPH LOOP DEMO")
    graph = build_agent_graph()
    print("✓ Successfully compiled LangGraph StateGraph (retrieve -> generate -> route -> [rewrite | abstain | END])")

    results_log = []

    # ------------------------------------------------------------------------
    # Scenario 1: High Confidence Direct Generation (0 Retries)
    # ------------------------------------------------------------------------
    banner("SCENARIO 1: High-Confidence Direct Generation (0 Retries)")
    q1 = "What was Apple's total net sales in fiscal year 2023?"
    print(f"User Query: {q1}")

    chunk_revenue = SearchResult(
        chunk_id="AAPL_10-K_2023_chunk_0042",
        text="Total net sales for fiscal year 2023 were $383,285 million, compared to $394,328 million in fiscal year 2022.",
        score=0.96,
        metadata={"ticker": "AAPL", "doc_id": "AAPL_10-K_2023", "doc_type": "10-K", "section": "Item 7"},
    )

    with patch("app.agent.nodes.search_dense", return_value=[chunk_revenue]), \
         patch("app.agent.nodes.search_sparse", return_value=[chunk_revenue]), \
         patch("app.agent.nodes.rerank", return_value=[chunk_revenue]):

        resp1 = run_financial_agent(query=q1, top_k=2, ticker_filter="AAPL", graph=graph)

    print(f"\n[Decision]:    {resp1.decision.upper()}")
    print(f"[Confidence]:  {resp1.confidence:.3f}")
    print(f"[Retries]:     {resp1.retries}")
    print(f"[Latency]:     {resp1.latency_ms:.1f} ms")
    print(f"[Answer]:\n{resp1.answer}")
    print(f"[Validated Citations]:\n{format_citations(resp1.citations)}")

    results_log.append({
        "scenario": 1,
        "name": "Direct Generation",
        "query": q1,
        "decision": resp1.decision,
        "confidence": resp1.confidence,
        "retries": resp1.retries,
        "latency_ms": resp1.latency_ms,
    })

    # ------------------------------------------------------------------------
    # Scenario 2: Ambiguous Retrieval -> Query Rewriting Cycle (1 Retry)
    # ------------------------------------------------------------------------
    banner("SCENARIO 2: Ambiguous Query -> GAAP Terminology Rewriting Loop (1 Retry)")
    q2 = "can you please tell me how much debt Apple had in 2023"
    print(f"Colloquial Query: \"{q2}\"")

    chunk_ambiguous = SearchResult(
        chunk_id="AAPL_10-K_2023_chunk_0012",
        text="Capital structure overview: Apple manages its capital structure through share repurchases and dividends.",
        score=0.62,
        metadata={"ticker": "AAPL", "doc_id": "AAPL_10-K_2023", "doc_type": "10-K"},
    )
    chunk_debt = SearchResult(
        chunk_id="AAPL_10-K_2023_chunk_0088",
        text="Total term debt was $105,960 million as of September 30, 2023, consisting of commercial paper and fixed-rate senior notes.",
        score=0.94,
        metadata={"ticker": "AAPL", "doc_id": "AAPL_10-K_2023", "doc_type": "10-K", "section": "Item 8 Note 10"},
    )

    call_count = 0

    def mock_answer_cycle(query, context_chunks):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # Pass 1: retrieval was conversational/ambiguous, low confidence triggers rewrite
            return MagicMock(
                answer="Apple mentions capital structure but specific debt obligations are ambiguous.",
                citations=[],
                confidence=0.55,
                decision="rewrite",
            )
        else:
            # Pass 2: rewritten GAAP query retrieves exact Note 10 debt figures
            return MagicMock(
                answer="Total term debt was $105,960 million as of September 30, 2023, consisting of commercial paper and fixed-rate senior notes [Chunk: AAPL_10-K_2023_chunk_0088].",
                citations=[
                    Citation(
                        chunk_id="AAPL_10-K_2023_chunk_0088",
                        doc_id="AAPL_10-K_2023",
                        quote="Total term debt was $105,960 million as of September 30, 2023",
                    )
                ],
                confidence=0.89,
                decision="generate",
            )

    with patch("app.agent.nodes.search_dense", return_value=[chunk_debt]), \
         patch("app.agent.nodes.search_sparse", return_value=[chunk_debt]), \
         patch("app.agent.nodes.rerank", return_value=[chunk_debt]), \
         patch("app.agent.nodes.get_llm_client") as mock_client:

        mock_instance = MagicMock()
        mock_instance.groq_api_key = None
        mock_instance.answer_query.side_effect = mock_answer_cycle
        mock_client.return_value = mock_instance

        resp2 = run_financial_agent(query=q2, top_k=2, ticker_filter="AAPL", max_retries=2, graph=graph)

    print(f"\n[Decision]:    {resp2.decision.upper()}")
    print(f"[Confidence]:  {resp2.confidence:.3f}")
    print(f"[Retries]:     {resp2.retries} (Triggered cyclical rewrite edge)")
    print(f"[Latency]:     {resp2.latency_ms:.1f} ms")
    print(f"[Answer]:\n{resp2.answer}")
    print(f"[Validated Citations]:\n{format_citations(resp2.citations)}")

    results_log.append({
        "scenario": 2,
        "name": "Query Rewriting Cycle",
        "query": q2,
        "decision": resp2.decision,
        "confidence": resp2.confidence,
        "retries": resp2.retries,
        "latency_ms": resp2.latency_ms,
    })

    # ------------------------------------------------------------------------
    # Scenario 3: Disciplined Abstention on Missing Evidence (Abstain)
    # ------------------------------------------------------------------------
    banner("SCENARIO 3: Disciplined Abstention on Speculative Query (Abstain)")
    q3 = "What are Apple's revenue and gross margin targets for fiscal 2050?"
    print(f"Speculative Query: \"{q3}\"")

    with patch("app.agent.nodes.search_dense", return_value=[]), \
         patch("app.agent.nodes.search_sparse", return_value=[]), \
         patch("app.agent.nodes.rerank", return_value=[]):

        resp3 = run_financial_agent(query=q3, top_k=2, ticker_filter="AAPL", graph=graph)

    print(f"\n[Decision]:    {resp3.decision.upper()}")
    print(f"[Confidence]:  {resp3.confidence:.3f}")
    print(f"[Retries]:     {resp3.retries}")
    print(f"[Latency]:     {resp3.latency_ms:.1f} ms")
    print(f"[Answer]:\n{resp3.answer}")
    print(f"[Validated Citations]:\n{format_citations(resp3.citations)}")

    results_log.append({
        "scenario": 3,
        "name": "Disciplined Abstention",
        "query": q3,
        "decision": resp3.decision,
        "confidence": resp3.confidence,
        "retries": resp3.retries,
        "latency_ms": resp3.latency_ms,
    })

    # Save timestamped results
    banner("SAVING BENCHMARK EXECUTION LOG")
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = Path(f"results/agentic_loop_demo_{timestamp}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(
            {
                "timestamp": timestamp,
                "agent_framework": "LangGraph StateGraph",
                "scenarios": results_log,
            },
            f,
            indent=2,
        )
    print(f"✓ Saved demo results to {out_path}")


if __name__ == "__main__":
    run_demo()
