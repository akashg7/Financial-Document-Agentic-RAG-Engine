"""
REFERENCE SOLUTION for Reciprocal Rank Fusion.

⚠️  DO NOT open this file until you have tried implementing it yourself!

This is the reference implementation for the hand-write piece in
app/retrieval/rrf.py. Copy the function body (not the whole file)
into rrf.py to make the tests pass.
"""

from app.schemas import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_n: int = 20,
) -> list[SearchResult]:
    """See rrf.py for the full docstring."""

    # Step 1: Accumulate RRF scores per chunk_id
    # We track two things:
    #   rrf_scores: {chunk_id → total RRF score}
    #   best_result: {chunk_id → (best_rank, SearchResult)} to pick the best copy
    rrf_scores: dict[str, float] = {}
    best_result: dict[str, tuple[int, SearchResult]] = {}

    for ranked_list in ranked_lists:
        seen_in_this_list: set[str] = set()

        for rank_0based, result in enumerate(ranked_list):
            cid = result.chunk_id

            # Skip duplicates within the same list (only first occurrence counts)
            if cid in seen_in_this_list:
                continue
            seen_in_this_list.add(cid)

            # 1-based rank for the RRF formula
            rank = rank_0based + 1
            rrf_contribution = 1.0 / (k + rank)

            # Accumulate the score
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + rrf_contribution

            # Track the best (lowest rank number) SearchResult for this chunk
            if cid not in best_result or rank < best_result[cid][0]:
                best_result[cid] = (rank, result)

    # Step 2: Build the fused result list
    fused = []
    for cid, score in rrf_scores.items():
        _, result = best_result[cid]
        fused.append(
            SearchResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=score,
                metadata=result.metadata,
            )
        )

    # Step 3: Sort by RRF score descending, truncate to top_n
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused[:top_n]

