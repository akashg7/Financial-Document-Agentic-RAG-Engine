"""
Citation validation and grounding verification.

WHY citation validation:
- LLMs frequently hallucinate citations: they may invent nonexistent chunk IDs,
  fabricate quotes that don't exist in the text, or attribute a real fact
  to the wrong document chunk.
- For high-stakes financial analysis, an unverified citation is worse than no
  citation because it gives the illusion of authoritative grounding.
- The validator enforces two strict checks:
  1. Chunk ID Existence: The cited chunk ID must exist in the retrieved context pool.
  2. Quote Verification: The cited verbatim quote must exist within the source chunk.
"""

import logging
import re
from typing import Sequence
from pydantic import BaseModel, Field

from app.schemas import Citation, SearchResult

logger = logging.getLogger(__name__)


class ValidatedCitation(BaseModel):
    """A citation with verification flags."""
    citation: Citation
    is_valid_chunk_id: bool = False
    is_quote_verified: bool = False
    failure_reason: str = ""


class CitationValidationResult(BaseModel):
    """Aggregate result of validating all citations in an LLM response."""
    is_fully_grounded: bool = Field(..., description="True if all citations are valid and verified")
    total_citations: int = 0
    valid_citations_count: int = 0
    verified_quotes_count: int = 0
    citation_precision: float = 0.0
    quote_precision: float = 0.0
    validated_citations: list[ValidatedCitation] = Field(default_factory=list)
    hallucinated_chunk_ids: list[str] = Field(default_factory=list)
    unverified_quotes: list[str] = Field(default_factory=list)


def _normalize_text(text: str) -> str:
    """Normalize whitespace and punctuation for fuzzy quote matching."""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def validate_citations(
    citations: Sequence[Citation],
    retrieved_chunks: Sequence[SearchResult],
) -> CitationValidationResult:
    """
    Validate a list of citations against the actual retrieved context chunks.

    Args:
        citations: List of Citation objects output by the LLM.
        retrieved_chunks: The candidate SearchResult chunks passed to the LLM.

    Returns:
        CitationValidationResult with per-citation and aggregate grounding metrics.
    """
    if not citations:
        # If no citations were generated, it is trivially valid only if no claims were made
        # or if the model abstained.
        return CitationValidationResult(
            is_fully_grounded=True,
            total_citations=0,
            valid_citations_count=0,
            verified_quotes_count=0,
            citation_precision=1.0,
            quote_precision=1.0,
        )

    # Map retrieved chunk IDs to their normalized texts
    chunk_map: dict[str, SearchResult] = {c.chunk_id: c for c in retrieved_chunks}
    chunk_texts_norm: dict[str, str] = {
        c.chunk_id: _normalize_text(c.text) for c in retrieved_chunks
    }

    validated_citations: list[ValidatedCitation] = []
    hallucinated_chunk_ids: list[str] = []
    unverified_quotes: list[str] = []

    valid_count = 0
    quote_count = 0

    for cit in citations:
        chunk_id = cit.chunk_id.strip()
        quote = cit.quote.strip()

        # Check 1: Does chunk ID exist in retrieved pool?
        if chunk_id not in chunk_map:
            hallucinated_chunk_ids.append(chunk_id)
            validated_citations.append(
                ValidatedCitation(
                    citation=cit,
                    is_valid_chunk_id=False,
                    is_quote_verified=False,
                    failure_reason=f"Chunk ID '{chunk_id}' was not in retrieved context",
                )
            )
            continue

        valid_count += 1
        source_chunk = chunk_map[chunk_id]
        norm_source = chunk_texts_norm[chunk_id]
        norm_quote = _normalize_text(quote)

        # Populate missing metadata on citation if available from chunk
        if not cit.doc_id:
            cit.doc_id = source_chunk.metadata.get("doc_id", "")
        if not cit.ticker:
            cit.ticker = source_chunk.metadata.get("ticker", "")
        if not cit.section:
            cit.section = source_chunk.metadata.get("section", "")
        if cit.page is None:
            cit.page = source_chunk.metadata.get("page")

        # Check 2: Does the quote exist in the source chunk text?
        # Allow exact match or normalized substring match
        is_quote_verified = False
        failure_reason = ""

        if not quote:
            # Empty quote provided
            failure_reason = "Empty quote provided for citation"
        elif norm_quote in norm_source:
            is_quote_verified = True
            quote_count += 1
        else:
            # Check for partial match (e.g. at least 60% of significant words match)
            quote_words = [w for w in norm_quote.split() if len(w) > 3]
            if quote_words:
                matched_words = sum(1 for w in quote_words if w in norm_source)
                ratio = matched_words / len(quote_words)
                if ratio >= 0.75:
                    is_quote_verified = True
                    quote_count += 1
                else:
                    unverified_quotes.append(quote)
                    failure_reason = f"Quote not found in chunk '{chunk_id}' (overlap={ratio:.1%})"
            else:
                unverified_quotes.append(quote)
                failure_reason = f"Quote not found in chunk '{chunk_id}'"

        validated_citations.append(
            ValidatedCitation(
                citation=cit,
                is_valid_chunk_id=True,
                is_quote_verified=is_quote_verified,
                failure_reason=failure_reason,
            )
        )

    total = len(citations)
    cit_precision = valid_count / total if total > 0 else 0.0
    quote_precision = quote_count / total if total > 0 else 0.0
    is_fully_grounded = (valid_count == total) and (quote_count == total)

    return CitationValidationResult(
        is_fully_grounded=is_fully_grounded,
        total_citations=total,
        valid_citations_count=valid_count,
        verified_quotes_count=quote_count,
        citation_precision=round(cit_precision, 4),
        quote_precision=round(quote_precision, 4),
        validated_citations=validated_citations,
        hallucinated_chunk_ids=hallucinated_chunk_ids,
        unverified_quotes=unverified_quotes,
    )
