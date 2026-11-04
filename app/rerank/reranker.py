"""
Cross-encoder reranker for two-stage retrieval.

WHY cross-encoder reranking (Day 4):
- Bi-encoders (used in dense search) encode query and document INDEPENDENTLY,
  then compare with cosine similarity. This is fast but misses fine-grained
  query-document interaction.
- Cross-encoders encode the (query, document) PAIR jointly through all
  transformer attention layers, producing a single relevance score.
  This captures word-level interactions (e.g., negation, coreference)
  that bi-encoders miss.
- Tradeoff: cross-encoders are ~100x slower than bi-encoders, so we can't
  run them over the full corpus. Instead, we use a two-stage pipeline:
    Stage 1: Hybrid search (dense + sparse + RRF) → top 50 candidates
    Stage 2: Cross-encoder reranks those 50 → final top K

WHY ms-marco-MiniLM-L-6-v2:
- Trained on MS MARCO passage ranking (real web search queries).
- Small enough to run on CPU (~80 MB), fast inference.
- Strong accuracy despite its size — well-suited to zero-cost constraint.
"""

import logging
from typing import Optional

from sentence_transformers import CrossEncoder

from app.config import settings
from app.schemas import SearchResult

logger = logging.getLogger(__name__)

# Module-level singleton — loaded lazily on first call.
# WHY lazy: avoids loading a ~80 MB model at import time, which would slow
# down startup of every test/script that imports any app module.
_cross_encoder: Optional[CrossEncoder] = None


def _get_model() -> CrossEncoder:
    """Load or return the cached cross-encoder model."""
    global _cross_encoder
    if _cross_encoder is None:
        logger.info("Loading cross-encoder model: %s", settings.reranker_model_name)
        _cross_encoder = CrossEncoder(settings.reranker_model_name)
        logger.info("Cross-encoder model loaded.")
    return _cross_encoder


def rerank(
    query: str,
    results: list[SearchResult],
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Rerank search results using a cross-encoder model.

    The cross-encoder scores each (query, chunk_text) pair jointly,
    producing more accurate relevance estimates than bi-encoder cosine
    similarity alone.

    Args:
        query: The user's natural language query.
        results: Candidate search results from first-stage retrieval.
        top_k: Number of top results to return after reranking.

    Returns:
        Reranked list of SearchResult, ordered by cross-encoder score
        (highest first), truncated to top_k.
    """
    if not results:
        return []

    # Nothing to rerank if fewer results than requested
    if len(results) <= 1:
        return results[:top_k]

    model = _get_model()

    # Build (query, passage) pairs for the cross-encoder.
    # WHY pairs: cross-encoders require both query and passage as input —
    # they compute attention across the concatenated sequence.
    pairs = [[query, result.text] for result in results]

    # Predict relevance scores — returns a numpy array of floats.
    # Higher = more relevant. Scores are NOT probabilities; they can be
    # negative or > 1. We use them only for ordering.
    scores = model.predict(pairs)

    # Attach cross-encoder scores to results
    scored_results = []
    for result, score in zip(results, scores):
        scored_results.append(
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=float(score),
                metadata={
                    **result.metadata,
                    "rerank_score": float(score),
                    # Preserve original retrieval score for debugging
                    "retrieval_score": result.score,
                },
            )
        )

    # Sort by cross-encoder score descending, take top_k
    scored_results.sort(key=lambda r: r.score, reverse=True)
    return scored_results[:top_k]


def reset_model() -> None:
    """Reset the cached model. Useful for testing."""
    global _cross_encoder
    _cross_encoder = None

