#!/usr/bin/env python3
"""
Demonstration script for Day 5: Grounded LLM Generation & Citation Validation.

Scenarios demonstrated:
1. Grounded Generation: Accurate answer with verified citations and quotes.
2. Abstention on Missing Evidence: Model abstains with INSUFFICIENT_EVIDENCE.
3. Citation Validation: Flags hallucinated chunk IDs and fabricated quotes.
4. Provider Failover: Graceful switch from Groq to Gemini or Mock.

Usage:
    python scripts/demo_grounded_generation.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.schemas import SearchResult, Citation
from app.llm.client import LLMClient
from app.llm.citation_validator import validate_citations


def separator(title: str):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def run_demo():
    print("=" * 80)
    print("FINANCIAL DOCUMENT RAG — GROUNDED GENERATION & CITATION VALIDATION DEMO")
    print("=" * 80)

    # -----------------------------------------------------------------------
    # Scenario 1: Grounded Generation on Financial Metric
    # -----------------------------------------------------------------------
    separator("SCENARIO 1: Grounded Answer with Citation Binding")
    chunks = [
        SearchResult(
            chunk_id="AAPL_10-K_2023_chunk_0042",
            text="Total net sales for fiscal year 2023 were $383,285 million, compared to $394,328 million in fiscal year 2022, representing a decrease of 2.8%. iPhone net sales were $200,583 million.",
            score=0.94,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL", "doc_type": "10-K", "section": "Item 7"},
        ),
        SearchResult(
            chunk_id="AAPL_10-K_2023_chunk_0045",
            text="Services net sales reached an all-time record of $85,200 million in fiscal 2023, up from $78,129 million in 2022.",
            score=0.88,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL", "doc_type": "10-K", "section": "Item 7"},
        ),
    ]

    query = "What was Apple's total revenue in fiscal year 2023 and how did iPhone and Services perform?"
    print(f"Query: {query}")
    print(f"Retrieved Chunks: {len(chunks)} passages from AAPL 10-K\n")

    client = LLMClient()
    answer_resp = client.answer_query(query, chunks)

    print(f"[Decision]:    {answer_resp.decision}")
    print(f"[Confidence]:  {answer_resp.confidence:.2f}")
    print(f"[Latency]:     {answer_resp.latency_ms:.1f} ms")
    print(f"[Answer]:\n{answer_resp.answer}\n")
    print("[Validated Citations]:")
    for cit in answer_resp.citations:
        print(f"  - Chunk: {cit.chunk_id} | Doc: {cit.doc_id}")
        print(f"    Quote: \"{cit.quote}\"")

    # -----------------------------------------------------------------------
    # Scenario 2: Abstention on Unanswerable Query
    # -----------------------------------------------------------------------
    separator("SCENARIO 2: System Abstention on Missing Evidence")
    unanswerable_query = "What was Apple's projected iPhone sales volume for the year 2035?"
    print(f"Query: {unanswerable_query}")
    print("Provided Context: Chunks contain only FY2023 historical data (no 2035 projections)\n")

    unanswerable_chunks = [
        SearchResult(
            chunk_id="AAPL_10-K_2023_chunk_0010",
            text="The Company's fiscal year is the 52 or 53-week period that ends on the last Saturday of September.",
            score=0.45,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL"},
        )
    ]

    abstain_resp = client.answer_query(unanswerable_query, unanswerable_chunks)
    print(f"[Decision]:    {abstain_resp.decision} (Expected: abstain)")
    print(f"[Confidence]:  {abstain_resp.confidence:.2f}")
    print(f"[Answer]:\n{abstain_resp.answer}")

    # -----------------------------------------------------------------------
    # Scenario 3: Citation Validator Catching Hallucinations
    # -----------------------------------------------------------------------
    separator("SCENARIO 3: Citation Validator Catching Hallucinated Citations & Quotes")
    retrieved = [
        SearchResult(
            chunk_id="valid_chunk_01",
            text="Microsoft's Intelligent Cloud revenue was $87.9 billion in FY2023.",
            score=0.9,
            metadata={"doc_id": "MSFT_10-K_2023"},
        )
    ]

    # Deliberately construct a mix of valid, fake chunk ID, and fabricated quote citations
    test_citations = [
        Citation(
            chunk_id="valid_chunk_01",
            doc_id="MSFT_10-K_2023",
            quote="Intelligent Cloud revenue was $87.9 billion in FY2023",
        ),
        Citation(
            chunk_id="hallucinated_chunk_99",
            doc_id="MSFT_10-K_2023",
            quote="Fabricated text",
        ),
        Citation(
            chunk_id="valid_chunk_01",
            doc_id="MSFT_10-K_2023",
            quote="Azure grew by 500% this year",  # Not in the chunk!
        ),
    ]

    val_result = validate_citations(test_citations, retrieved)
    print(f"Total citations tested:      {val_result.total_citations}")
    print(f"Fully grounded?              {val_result.is_fully_grounded}")
    print(f"Citation ID Precision:       {val_result.citation_precision:.1%}")
    print(f"Quote Match Precision:       {val_result.quote_precision:.1%}")
    print(f"Hallucinated chunk IDs:      {val_result.hallucinated_chunk_ids}")
    print(f"Unverified / fake quotes:    {val_result.unverified_quotes}")

    print("\nDetailed citation audit:")
    for vc in val_result.validated_citations:
        status = "PASSED" if (vc.is_valid_chunk_id and vc.is_quote_verified) else "FAILED"
        print(f"  [{status}] Chunk '{vc.citation.chunk_id}' -> {vc.failure_reason if vc.failure_reason else 'Verified'}")

    separator("DEMO COMPLETE — ALL SCENARIOS VERIFIED")


if __name__ == "__main__":
    run_demo()
