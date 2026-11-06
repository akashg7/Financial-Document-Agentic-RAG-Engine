"""
Unit tests for citation validation and quote verification (Day 5).
"""

from app.schemas import Citation, SearchResult
from app.llm.citation_validator import validate_citations


def test_validate_citations_empty():
    """Empty citations list is valid."""
    res = validate_citations([], [])
    assert res.is_fully_grounded is True
    assert res.total_citations == 0
    assert res.citation_precision == 1.0


def test_validate_citations_valid_exact_match():
    """Valid chunk ID and exact quote substring match pass verification."""
    chunks = [
        SearchResult(
            chunk_id="chunk_apple_01",
            text="Apple Inc. reported total revenue of $383.3 billion for the fiscal year 2023.",
            score=0.95,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL"},
        )
    ]

    citations = [
        Citation(
            chunk_id="chunk_apple_01",
            doc_id="AAPL_10-K_2023",
            quote="total revenue of $383.3 billion for the fiscal year 2023",
        )
    ]

    res = validate_citations(citations, chunks)
    assert res.is_fully_grounded is True
    assert res.total_citations == 1
    assert res.valid_citations_count == 1
    assert res.verified_quotes_count == 1
    assert res.citation_precision == 1.0
    assert res.quote_precision == 1.0
    assert len(res.hallucinated_chunk_ids) == 0
    assert len(res.unverified_quotes) == 0


def test_validate_citations_hallucinated_chunk_id():
    """Citing a chunk ID not in the retrieved pool is flagged as hallucinated."""
    chunks = [
        SearchResult(
            chunk_id="real_chunk_01",
            text="Real chunk content here.",
            score=0.9,
            metadata={},
        )
    ]

    citations = [
        Citation(
            chunk_id="hallucinated_chunk_99",
            doc_id="doc_fake",
            quote="Fake quote",
        )
    ]

    res = validate_citations(citations, chunks)
    assert res.is_fully_grounded is False
    assert res.total_citations == 1
    assert res.valid_citations_count == 0
    assert res.citation_precision == 0.0
    assert "hallucinated_chunk_99" in res.hallucinated_chunk_ids


def test_validate_citations_hallucinated_quote():
    """Valid chunk ID but quote text not in chunk is flagged as unverified."""
    chunks = [
        SearchResult(
            chunk_id="chunk_real",
            text="Tesla delivered 1.81 million vehicles in 2023.",
            score=0.88,
            metadata={"doc_id": "TSLA_10-K_2023"},
        )
    ]

    citations = [
        Citation(
            chunk_id="chunk_real",
            doc_id="TSLA_10-K_2023",
            quote="Tesla projected 5 million deliveries for 2024",
        )
    ]

    res = validate_citations(citations, chunks)
    assert res.is_fully_grounded is False
    assert res.valid_citations_count == 1
    assert res.verified_quotes_count == 0
    assert res.citation_precision == 1.0
    assert res.quote_precision == 0.0
    assert len(res.unverified_quotes) == 1


def test_validate_citations_fuzzy_quote_match():
    """Slight whitespace or casing differences in quote still verify successfully."""
    chunks = [
        SearchResult(
            chunk_id="c1",
            text="Alphabet's Google Cloud   operating income was  positive for the full year.",
            score=0.9,
            metadata={"doc_id": "GOOGL_2023"},
        )
    ]

    citations = [
        Citation(
            chunk_id="c1",
            doc_id="GOOGL_2023",
            quote="Google Cloud operating income was positive",
        )
    ]

    res = validate_citations(citations, chunks)
    assert res.is_fully_grounded is True
    assert res.verified_quotes_count == 1
