"""
Confidence Scorer for Financial Document Intelligence Engine.

WHY multi-signal confidence scoring (Day 6):
- Relying on a single metric (e.g., top vector similarity) is brittle: a passage
  can have high cosine similarity due to repeated financial terms while still
  failing to answer the specific question.
- Multi-signal fusion combines:
  1. Retrieval Quality (top reranker score, score margin between #1 and #2, candidate relevance)
  2. Grounding Fidelity (citation precision, verbatim quote verification)
  3. Linguistic Certainty (absence of hedging, explicit insufficiency signals)
- Calibrated thresholds allow the agent (Day 7) to decide:
  - 'generate': Evidence is authoritative and citations verify.
  - 'rewrite': Evidence is ambiguous or marginal; reformulating the query may find better chunks.
  - 'abstain': Evidence is absent or unanswerable; declining to answer prevents hallucination.
"""

import math
from typing import Sequence, Optional
from pydantic import BaseModel, Field

from app.schemas import SearchResult
from app.llm.citation_validator import CitationValidationResult

# Common hedging phrases indicative of epistemic uncertainty in financial NLP
HEDGING_PHRASES = [
    "might be",
    "possibly",
    "unclear",
    "not explicitly stated",
    "likely",
    "could be",
    "it appears",
    "cannot be determined",
    "may have",
    "suggests that",
]


class ConfidenceBreakdown(BaseModel):
    """Detailed breakdown of individual confidence signals."""
    retrieval_signal: float = Field(..., ge=0.0, le=1.0, description="Quality of retrieved passages")
    grounding_signal: float = Field(..., ge=0.0, le=1.0, description="Fidelity of citations & quotes")
    linguistic_signal: float = Field(..., ge=0.0, le=1.0, description="Certainty of answer phrasing")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Calibrated composite score")
    decision: str = Field(..., description="Action: 'generate', 'rewrite', or 'abstain'")
    reasoning: str = Field(default="", description="Human-readable justification for the decision")


def _sigmoid(x: float) -> float:
    """Standard logistic sigmoid to map unbounded cross-encoder logits to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def score_confidence(
    query: str,
    answer: str,
    retrieved_chunks: Sequence[SearchResult],
    validation_result: Optional[CitationValidationResult] = None,
    threshold_generate: float = 0.70,
    threshold_abstain: float = 0.40,
) -> ConfidenceBreakdown:
    """
    Compute a calibrated multi-signal confidence score and determine agent action.

    Args:
        query: The user query string.
        answer: The generated answer text from the LLM.
        retrieved_chunks: Chunks retrieved and reranked for this query.
        validation_result: CitationValidationResult from the citation validator.
        threshold_generate: Minimum score required to accept answer (default: 0.70).
        threshold_abstain: Score below which agent abstains (default: 0.40).

    Returns:
        ConfidenceBreakdown with individual signals, overall score, decision, and reasoning.
    """
    ans_clean = (answer or "").strip()
    ans_lower = ans_clean.lower()

    # 1. HARD OVERRIDES: Explicit insufficiency or no chunks retrieved
    # If the model explicitly stated evidence is missing, or retrieval returned 0 chunks,
    # we must immediately abstain with 0 confidence to prevent hallucination.
    if not retrieved_chunks or "insufficient_evidence" in ans_lower or "insufficient evidence" in ans_lower:
        return ConfidenceBreakdown(
            retrieval_signal=0.0,
            grounding_signal=0.0,
            linguistic_signal=0.0,
            overall_confidence=0.0,
            decision="abstain",
            reasoning="System abstains due to missing or explicitly insufficient evidence.",
        )

    # 2. RETRIEVAL SIGNAL (0.0 to 1.0)
    top_chunk = retrieved_chunks[0]
    top_raw_score = top_chunk.score

    # Map cross-encoder logit (-10 to +10) via sigmoid
    # Scores: < -2 (irrelevant), 0 (neutral), > 2 (highly relevant)
    top_score_norm = _sigmoid(top_raw_score)

    # Margin check between #1 and #2 candidate
    # A wide margin indicates a clear, unambiguous best passage.
    if len(retrieved_chunks) > 1:
        second_score = retrieved_chunks[1].score
        margin = max(0.0, top_raw_score - second_score)
        margin_bonus = min(0.3, margin * 0.1)
    else:
        margin_bonus = 0.1

    retrieval_signal = min(1.0, max(0.0, 0.7 * top_score_norm + margin_bonus))

    # 3. GROUNDING SIGNAL (0.0 to 1.0)
    if validation_result is None or validation_result.total_citations == 0:
        # If citations were expected but absent, penalize heavily
        grounding_signal = 0.2
    else:
        # Combine quote precision (70%) and citation ID precision (30%)
        g_raw = 0.70 * validation_result.quote_precision + 0.30 * validation_result.citation_precision
        # If any hallucinated chunk IDs exist, hard cap grounding at 0.3
        if validation_result.hallucinated_chunk_ids:
            grounding_signal = min(0.3, g_raw)
        else:
            grounding_signal = min(1.0, max(0.0, g_raw))

    # 4. LINGUISTIC SIGNAL (0.0 to 1.0)
    # Deduct 0.15 for each hedging phrase found in the answer
    hedge_count = sum(1 for phrase in HEDGING_PHRASES if phrase in ans_lower)
    linguistic_signal = max(0.0, 1.0 - (hedge_count * 0.15))

    # 5. COMPOSITE FUSION
    # 45% grounding fidelity, 40% retrieval relevance, 15% linguistic certainty
    overall = (
        0.45 * grounding_signal
        + 0.40 * retrieval_signal
        + 0.15 * linguistic_signal
    )
    overall_confidence = round(min(1.0, max(0.0, overall)), 4)

    # 6. DECISION THRESHOLDS
    if overall_confidence >= threshold_generate and grounding_signal >= 0.50:
        decision = "generate"
        reasoning = f"High confidence ({overall_confidence:.2f}) with verified citations."
    elif overall_confidence >= threshold_abstain:
        decision = "rewrite"
        reasoning = (
            f"Moderate confidence ({overall_confidence:.2f}). Evidence is ambiguous or citations "
            "are incomplete; rewriting the query may retrieve better passages."
        )
    else:
        decision = "abstain"
        reasoning = (
            f"Low confidence ({overall_confidence:.2f}) below threshold ({threshold_abstain:.2f}). "
            "Abstaining to prevent financial hallucination."
        )

    return ConfidenceBreakdown(
        retrieval_signal=round(retrieval_signal, 4),
        grounding_signal=round(grounding_signal, 4),
        linguistic_signal=round(linguistic_signal, 4),
        overall_confidence=overall_confidence,
        decision=decision,
        reasoning=reasoning,
    )
