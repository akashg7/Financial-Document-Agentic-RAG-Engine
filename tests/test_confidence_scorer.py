"""
Unit tests for Day 6 Hand-write Piece #2: Confidence Scorer.
"""

import pytest
from app.schemas import SearchResult, Citation
from app.llm.citation_validator import CitationValidationResult, ValidatedCitation
from app.confidence.scorer import score_confidence


@pytest.fixture
def high_quality_chunk():
    return SearchResult(
        chunk_id="AAPL_chunk_001",
        text="Apple reported total revenue of $383.3 billion in fiscal year 2023.",
        score=3.5,  # High cross-encoder logit
        metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL"},
    )


@pytest.fixture
def second_chunk():
    return SearchResult(
        chunk_id="AAPL_chunk_002",
        text="iPhone sales were $200.6 billion.",
        score=1.0,
        metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL"},
    )


@pytest.fixture
def low_quality_chunk():
    return SearchResult(
        chunk_id="AAPL_chunk_099",
        text="General accounting policies and definitions.",
        score=-3.0,  # Negative cross-encoder logit
        metadata={"doc_id": "AAPL_10-K_2023"},
    )


def test_confidence_scorer_insufficient_evidence_override():
    """Explicit INSUFFICIENT_EVIDENCE in answer results in 0.0 confidence and abstain."""
    breakdown = score_confidence(
        query="What was Apple revenue in 2040?",
        answer="INSUFFICIENT_EVIDENCE: The filings do not contain projections for 2040.",
        retrieved_chunks=[],
        validation_result=None,
    )
    assert breakdown.overall_confidence == 0.0
    assert breakdown.decision == "abstain"
    assert "missing or explicitly insufficient evidence" in breakdown.reasoning.lower()


def test_confidence_scorer_empty_chunks_abstain():
    """Empty retrieved chunks results in 0.0 confidence and abstain."""
    breakdown = score_confidence(
        query="What was revenue?",
        answer="Apple reported revenue of $383 billion.",
        retrieved_chunks=[],
        validation_result=None,
    )
    assert breakdown.overall_confidence == 0.0
    assert breakdown.decision == "abstain"


def test_confidence_scorer_high_confidence_generate(high_quality_chunk, second_chunk):
    """High retrieval score, 100% verified citations, and confident language yields 'generate'."""
    val_result = CitationValidationResult(
        is_fully_grounded=True,
        total_citations=1,
        valid_citations_count=1,
        verified_quotes_count=1,
        citation_precision=1.0,
        quote_precision=1.0,
        validated_citations=[
            ValidatedCitation(
                citation=Citation(
                    chunk_id="AAPL_chunk_001",
                    doc_id="AAPL_10-K_2023",
                    quote="total revenue of $383.3 billion in fiscal year 2023",
                ),
                is_valid_chunk_id=True,
                is_quote_verified=True,
            )
        ],
    )

    breakdown = score_confidence(
        query="What was Apple's total revenue in fiscal year 2023?",
        answer="Apple reported total revenue of $383.3 billion in fiscal year 2023 [Chunk: AAPL_chunk_001].",
        retrieved_chunks=[high_quality_chunk, second_chunk],
        validation_result=val_result,
    )

    assert breakdown.overall_confidence >= 0.70
    assert breakdown.decision == "generate"
    assert breakdown.retrieval_signal > 0.70
    assert breakdown.grounding_signal == 1.0
    assert breakdown.linguistic_signal == 1.0


def test_confidence_scorer_hedging_penalty(high_quality_chunk, second_chunk):
    """Hedging phrases in answer penalize the linguistic signal."""
    val_result = CitationValidationResult(
        is_fully_grounded=True,
        total_citations=1,
        valid_citations_count=1,
        verified_quotes_count=1,
        citation_precision=1.0,
        quote_precision=1.0,
    )

    breakdown_confident = score_confidence(
        query="What was revenue?",
        answer="Apple reported revenue of $383 billion.",
        retrieved_chunks=[high_quality_chunk, second_chunk],
        validation_result=val_result,
    )

    breakdown_hedged = score_confidence(
        query="What was revenue?",
        answer="Apple might be reporting around $383 billion, but it is possibly unclear and could be different.",
        retrieved_chunks=[high_quality_chunk, second_chunk],
        validation_result=val_result,
    )

    assert breakdown_hedged.linguistic_signal < breakdown_confident.linguistic_signal
    assert breakdown_hedged.overall_confidence < breakdown_confident.overall_confidence


def test_confidence_scorer_hallucinated_chunk_id_penalty(high_quality_chunk):
    """Hallucinated chunk ID caps grounding signal and lowers overall score."""
    val_result = CitationValidationResult(
        is_fully_grounded=False,
        total_citations=1,
        valid_citations_count=0,
        verified_quotes_count=0,
        citation_precision=0.0,
        quote_precision=0.0,
        hallucinated_chunk_ids=["fake_chunk_id"],
    )

    breakdown = score_confidence(
        query="What was revenue?",
        answer="Apple revenue was $383 billion [Chunk: fake_chunk_id].",
        retrieved_chunks=[high_quality_chunk],
        validation_result=val_result,
    )

    assert breakdown.grounding_signal <= 0.30
    assert breakdown.decision != "generate"


def test_confidence_scorer_moderate_confidence_rewrite(second_chunk):
    """Moderate retrieval and incomplete evidence triggers 'rewrite' decision."""
    # Chunk with near-zero cross-encoder score
    moderate_chunk = SearchResult(
        chunk_id="chunk_mod",
        text="Discussion of general market conditions.",
        score=0.2,
        metadata={},
    )
    val_result = CitationValidationResult(
        is_fully_grounded=True,
        total_citations=1,
        valid_citations_count=1,
        verified_quotes_count=1,
        citation_precision=1.0,
        quote_precision=1.0,
    )

    breakdown = score_confidence(
        query="What were the gross margins?",
        answer="Market conditions were discussed [Chunk: chunk_mod].",
        retrieved_chunks=[moderate_chunk, second_chunk],
        validation_result=val_result,
        threshold_generate=0.85,  # Higher generate threshold to test rewrite band
        threshold_abstain=0.40,
    )

    assert 0.40 <= breakdown.overall_confidence < 0.85
    assert breakdown.decision == "rewrite"


def test_confidence_scorer_low_confidence_abstain(low_quality_chunk):
    """Negative retrieval scores and poor grounding trigger 'abstain'."""
    val_result = CitationValidationResult(
        is_fully_grounded=False,
        total_citations=1,
        valid_citations_count=0,
        verified_quotes_count=0,
        citation_precision=0.0,
        quote_precision=0.0,
        hallucinated_chunk_ids=["hallucinated_id"],
    )

    breakdown = score_confidence(
        query="What is the forecast?",
        answer="It might be possible that accounting policies apply.",
        retrieved_chunks=[low_quality_chunk],
        validation_result=val_result,
    )

    assert breakdown.overall_confidence < 0.40
    assert breakdown.decision == "abstain"


def test_confidence_scorer_custom_thresholds(high_quality_chunk):
    """Custom thresholds are respected by the decision logic."""
    val_result = CitationValidationResult(
        is_fully_grounded=True,
        total_citations=1,
        valid_citations_count=1,
        verified_quotes_count=1,
        citation_precision=1.0,
        quote_precision=1.0,
    )

    # With very strict generate threshold of 0.99
    breakdown_strict = score_confidence(
        query="What was revenue?",
        answer="Apple reported revenue of $383 billion.",
        retrieved_chunks=[high_quality_chunk],
        validation_result=val_result,
        threshold_generate=0.99,
        threshold_abstain=0.50,
    )
    assert breakdown_strict.decision == "rewrite"

    # With very permissive generate threshold of 0.50
    breakdown_permissive = score_confidence(
        query="What was revenue?",
        answer="Apple reported revenue of $383 billion.",
        retrieved_chunks=[high_quality_chunk],
        validation_result=val_result,
        threshold_generate=0.50,
        threshold_abstain=0.20,
    )
    assert breakdown_permissive.decision == "generate"

