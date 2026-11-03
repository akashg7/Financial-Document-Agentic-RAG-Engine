"""
Reciprocal Rank Fusion (RRF) — combines ranked lists from multiple retrieval methods.

THIS IS A HAND-WRITE PIECE: the function signature, docstring, and tests are
provided. YOU implement the body. The reference solution is in:
    /docs/learn/solutions/rrf_solution.py
Do NOT open it until you have tried implementing it yourself.

WHAT IS RRF?
    Given multiple ranked lists of search results (e.g., one from dense search,
    one from sparse/BM25 search), RRF produces a single fused ranking by
    assigning each result a score based on its RANK (position) in each list,
    not its raw score.

    For each result appearing in any list:
        rrf_score = sum over all lists of: 1 / (k + rank_in_list)
    
    where k is a constant (typically 60) that prevents top-ranked results from
    dominating too aggressively.

WHY RRF OVER OTHER FUSION METHODS:
    - Score normalization is hard: dense cosine scores (0.3–0.9) and BM25 scores
      (0.1–25.0) live on completely different scales. Naive score addition or
      averaging would let the higher-magnitude method dominate.
    - RRF uses RANK position only, so it's scale-invariant by design.
    - Simple to implement (~15 lines), no hyperparameters to tune beyond k.
    - Proven to outperform individual methods in benchmarks (Cormack et al., 2009).
"""

from app.schemas import SearchResult


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    k: int = 60,
    top_n: int = 20,
) -> list[SearchResult]:
    """
    Fuse multiple ranked result lists using Reciprocal Rank Fusion.

    Algorithm:
        1. For each result in each ranked list, compute: 1 / (k + rank)
           where rank is the 1-based position in that list.
        2. Sum the RRF scores for each unique result across all lists.
        3. Sort by total RRF score descending.
        4. Return the top_n results with the RRF score set in their .score field.

    Args:
        ranked_lists: A list of ranked result lists. Each inner list is ordered
                      by relevance (best first). Results are identified by chunk_id.
        k: Smoothing constant. Higher k = less aggressive boosting of top ranks.
           Default 60 is the standard from the original RRF paper.
        top_n: Maximum number of results to return after fusion.

    Returns:
        A list of SearchResult objects sorted by fused RRF score (highest first).
        Each result's .score field is set to its RRF score.
        If the same chunk_id appears in multiple lists, the SearchResult from
        the list where it ranked highest is used (preserving its text/metadata).

    Example:
        dense_results = [chunk_A (score=0.9), chunk_B (score=0.7), chunk_C (score=0.5)]
        sparse_results = [chunk_B (score=15.2), chunk_D (score=8.1), chunk_A (score=3.0)]

        RRF with k=60:
          chunk_A: 1/(60+1) + 1/(60+3) = 0.01639 + 0.01587 = 0.03226
          chunk_B: 1/(60+2) + 1/(60+1) = 0.01613 + 0.01639 = 0.03252
          chunk_C: 1/(60+3) + 0         = 0.01587
          chunk_D: 0         + 1/(60+2) = 0.01613

        Fused ranking: [chunk_B, chunk_A, chunk_D, chunk_C]
        Note: chunk_B wins because it ranked well in BOTH lists.

    Edge Cases:
        - Empty ranked_lists → return []
        - Single ranked list → equivalent to just re-scoring with RRF weights
        - Duplicate chunk_ids within a single list → only first occurrence counts
    """
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

