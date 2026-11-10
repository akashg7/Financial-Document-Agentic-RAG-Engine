"""
API routes for the Financial Document Intelligence Engine.

Day 1: health check and ingestion endpoints.
Day 2: added POST /search/dense for semantic vector search,
       swapped in-memory job store for Redis-backed store.
Day 3: added POST /search/sparse for BM25 keyword search,
       added POST /search/hybrid for RRF-fused dense+sparse search.
Day 4: added POST /search/rerank for hybrid + cross-encoder reranking.
Day 5: added POST /generate for grounded question answering with verified citations.
"""

import time
import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import (
    IngestionRequest,
    IngestionResponse,
    IngestionStatus,
    JobStatus,
    SearchRequest,
    SearchResponse,
    GenerateRequest,
    QueryRequest,
    AnswerResponse,
    InvalidateDocumentRequest,
    InvalidateChunksRequest,
)
from app.cache.models import InvalidationResult, CacheStats
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.redis_job_store import get_job, set_job

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse, tags=["ingestion"])
async def ingest_documents(
    request: IngestionRequest,
    background_tasks: BackgroundTasks,
):
    """
    Start an async ingestion job.

    WHY async with BackgroundTasks instead of synchronous:
    - Ingesting thousands of documents takes minutes. We don't want to hold
      an HTTP connection open that long — the client would timeout.
    - BackgroundTasks is simpler than Celery/RQ for our scale. If we need
      distributed workers later, we can swap in a task queue without changing
      this endpoint's interface (just the dispatch mechanism).

    Returns a job_id the client can poll via GET /job/{job_id}.
    """
    job_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Initialize job status in Redis (falls back to in-memory if Redis is down)
    job = JobStatus(
        job_id=job_id,
        status=IngestionStatus.PENDING,
        created_at=now,
        updated_at=now,
    )
    set_job(job_id, job)

    # Dispatch the actual work to a background thread
    background_tasks.add_task(run_ingestion_pipeline, job_id, request)

    logger.info("Created ingestion job %s for source=%s", job_id, request.source)
    return IngestionResponse(job_id=job_id, status=IngestionStatus.PENDING)


@router.get("/job/{job_id}", response_model=JobStatus, tags=["ingestion"])
async def get_job_status(job_id: str):
    """
    Check the status of an ingestion job.

    WHY polling instead of WebSockets for job status:
    - Simpler to implement and test. Job status is a low-frequency check
      (every few seconds), not a real-time stream.
    - Day 9 adds SSE for query streaming, which is a different use case.
    """
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job


@router.post("/search/dense", response_model=SearchResponse, tags=["search"])
async def dense_search(request: SearchRequest):
    """
    Perform dense (semantic) vector search over ingested documents.

    HOW IT WORKS:
    1. The query string is embedded using the same sentence-transformers model
       used during ingestion (all-MiniLM-L6-v2, 384 dims).
    2. The embedding is sent to Qdrant, which computes cosine similarity
       against all stored chunk vectors.
    3. Qdrant returns the top_k most similar chunks, optionally filtered
       by ticker or document type.

    WHY a separate /search/dense endpoint instead of a single /search:
    - Day 3 adds sparse search (BM25) and hybrid search (RRF fusion).
      Having separate endpoints makes it easy to A/B test retrieval methods
      and lets the evaluation framework compare them directly.
    - The /query endpoint (Day 6+) will use hybrid search internally,
      but the /search/* endpoints are useful for debugging and benchmarking.
    """
    from app.retrieval.vector_store import search_dense

    start = time.time()

    results = search_dense(
        query=request.query,
        top_k=request.top_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
    )

    latency_ms = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        results=results,
        retrieval_method="dense",
        latency_ms=round(latency_ms, 2),
    )


@router.post("/search/sparse", response_model=SearchResponse, tags=["search"])
async def sparse_search(request: SearchRequest):
    """
    Perform sparse (BM25 keyword) search over ingested documents.

    HOW IT WORKS:
    1. The query is tokenized and encoded as a BM25-weighted sparse vector
       where each dimension corresponds to a vocabulary term (hashed to an int),
       and the value is the term's IDF weight.
    2. Qdrant computes dot product of this query sparse vector against all
       stored document sparse vectors → this is equivalent to BM25 scoring.
    3. Returns the top_k most relevant chunks.

    WHY sparse search?
    - Dense search excels at semantic similarity but can miss exact terms.
    - Sparse/BM25 search captures exact keyword matches, which is critical
      for financial jargon: "EBITDA", "Item 1A", "10-K/A", ticker symbols.
    - Day 3's hybrid endpoint fuses both for the best of both worlds.
    """
    from app.retrieval.vector_store import search_sparse

    start = time.time()

    results = search_sparse(
        query=request.query,
        top_k=request.top_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
    )

    latency_ms = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        results=results,
        retrieval_method="sparse",
        latency_ms=round(latency_ms, 2),
    )


@router.post("/search/hybrid", response_model=SearchResponse, tags=["search"])
async def hybrid_search(request: SearchRequest):
    """
    Perform hybrid search: dense + sparse, fused with Reciprocal Rank Fusion.

    HOW IT WORKS:
    1. Run dense (semantic) search → top_k * 2 results.
    2. Run sparse (BM25) search → top_k * 2 results.
    3. Fuse both ranked lists using RRF (rank-based, scale-invariant).
    4. Return the top_k fused results.

    WHY over-retrieve (top_k * 2) before fusion:
    - RRF rewards results that appear in BOTH lists. If we only retrieve
      top_k from each, a result at rank k+1 in both lists would be missed
      even though it should rank high in the fused output.
    - Retrieving 2x gives RRF enough candidates to find overlap.

    WHY RRF over weighted score averaging:
    - Dense scores (cosine similarity: 0.3–0.9) and sparse scores (BM25: 0.1–25.0)
      live on completely different scales. Averaging would let BM25 dominate.
    - RRF uses RANK position only, so it's scale-invariant.
    """
    from app.retrieval.vector_store import search_dense, search_sparse
    from app.retrieval.rrf import reciprocal_rank_fusion

    start = time.time()

    # Over-retrieve from both methods to give RRF enough candidates
    retrieval_k = min(request.top_k * 2, 50)

    dense_results = search_dense(
        query=request.query,
        top_k=retrieval_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
    )

    sparse_results = search_sparse(
        query=request.query,
        top_k=retrieval_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
    )

    # Fuse with RRF
    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        top_n=request.top_k,
    )

    latency_ms = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        results=fused_results,
        retrieval_method="hybrid",
        latency_ms=round(latency_ms, 2),
    )



@router.post("/search/rerank", response_model=SearchResponse, tags=["search"])
async def rerank_search(request: SearchRequest):
    """
    Two-stage retrieval: hybrid search (dense + sparse + RRF) → cross-encoder rerank.

    HOW IT WORKS:
    1. Hybrid search retrieves top 50 candidates (over-retrieval pool).
    2. Cross-encoder scores each (query, chunk) pair jointly through all
       attention layers — much more accurate than bi-encoder similarity.
    3. Return the top_k reranked results.

    WHY 50 candidates for reranking:
    - Cross-encoder inference scales linearly with candidate count.
      50 pairs on the ms-marco-MiniLM model takes ~50-200 ms on CPU,
      which keeps total latency under 1 second.
    - Beyond 50, we see diminishing recall gains but linear cost increase.

    WHY a separate endpoint instead of a flag on /search/hybrid:
    - Keeps the benchmark script clean: each retrieval strategy has its own
      endpoint, making A/B comparison straightforward.
    - The /query endpoint (Day 7) will use rerank internally.
    """
    from app.retrieval.vector_store import search_dense, search_sparse
    from app.retrieval.rrf import reciprocal_rank_fusion
    from app.rerank.reranker import rerank

    start = time.time()

    # Stage 1: Hybrid search — over-retrieve 50 candidates for the reranker
    retrieval_k = 50

    dense_results = search_dense(
        query=request.query,
        top_k=retrieval_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
    )

    sparse_results = search_sparse(
        query=request.query,
        top_k=retrieval_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=request.doc_type_filter.value if request.doc_type_filter else None,
    )

    # Fuse with RRF — keep all 50 candidates for the cross-encoder
    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        top_n=retrieval_k,
    )

    # Stage 2: Cross-encoder reranking — precise scoring of each candidate
    reranked_results = rerank(
        query=request.query,
        results=fused_results,
        top_k=request.top_k,
    )

    latency_ms = (time.time() - start) * 1000

    return SearchResponse(
        query=request.query,
        results=reranked_results,
        retrieval_method="hybrid+rerank",
        latency_ms=round(latency_ms, 2),
    )



@router.post("/generate", response_model=AnswerResponse, tags=["generation"])
async def generate_answer(request: GenerateRequest):
    """
    Generate a grounded answer to a financial question with citation binding.

    HOW IT WORKS:
    1. Retrieve candidates via hybrid search (dense + sparse + RRF).
    2. Rerank candidates with the cross-encoder to select the highest-precision chunks.
    3. Construct a grounded context prompt with explicit chunk IDs.
    4. Call the LLM (Groq primary with automatic Gemini fallback).
    5. Validate all output citations: verify chunk ID exists and quote matches verbatim.
    6. Return AnswerResponse with verified citations and confidence score.

    WHY validate citations at the API boundary:
    - Guarantees the caller never receives fabricated chunk IDs or hallucinated quotes.
    - If evidence is weak or missing, the response explicitly signals "abstain".
    """
    from app.retrieval.vector_store import search_dense, search_sparse
    from app.retrieval.rrf import reciprocal_rank_fusion
    from app.rerank.reranker import rerank
    from app.llm.client import get_llm_client

    start = time.time()

    # Stage 1: Over-retrieve candidates via hybrid search
    retrieval_k = min(max(request.top_k * 4, 20), 50)
    ticker_val = request.ticker_filter
    doc_type_val = request.doc_type_filter.value if request.doc_type_filter else None

    dense_results = search_dense(
        query=request.query,
        top_k=retrieval_k,
        ticker_filter=ticker_val,
        doc_type_filter=doc_type_val,
    )

    sparse_results = search_sparse(
        query=request.query,
        top_k=retrieval_k,
        ticker_filter=ticker_val,
        doc_type_filter=doc_type_val,
    )

    fused_results = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        top_n=retrieval_k,
    )

    # Stage 2: Neural rerank to extract the best top_k chunks
    top_chunks = rerank(
        query=request.query,
        results=fused_results,
        top_k=request.top_k,
    )

    # Stage 3: Grounded LLM generation & citation validation
    client = get_llm_client()
    answer_resp = client.answer_query(
        query=request.query,
        context_chunks=top_chunks,
    )

    total_latency_ms = (time.time() - start) * 1000
    answer_resp.latency_ms = round(total_latency_ms, 2)

    return answer_resp


@router.post("/query", response_model=AnswerResponse, tags=["agent"])
async def query_agent(request: QueryRequest):
    """
    Execute the agentic financial QA loop (LangGraph StateGraph).

    HOW IT WORKS:
    1. Enters the compiled LangGraph state machine.
    2. Retrieve node performs hybrid search (dense + sparse + RRF) and cross-encoder rerank.
    3. Generate node synthesizes grounded answer, validates citations, and calculates
       calibrated composite confidence.
    4. Conditional edge evaluates confidence:
       - confidence >= 0.70 ('generate') -> terminates and returns verified answer.
       - 0.40 <= confidence < 0.70 ('rewrite') -> rewrites query and loops back to retrieve (up to max_retries).
       - confidence < 0.40 or retries exhausted -> routes to abstain node.
    5. Returns AnswerResponse with final answer, verified citations, confidence, and retries.
    """
    from app.agent.graph import run_financial_agent

    doc_type_val = request.doc_type_filter.value if request.doc_type_filter else None
    return run_financial_agent(
        query=request.query,
        top_k=request.top_k,
        ticker_filter=request.ticker_filter,
        doc_type_filter=doc_type_val,
        max_retries=request.max_retries,
    )


# ============================================================
# Semantic Cache & Lineage Endpoints (Day 8)
# ============================================================

@router.post("/cache/invalidate/document", response_model=InvalidationResult, tags=["cache"])
async def invalidate_document_cache(request: InvalidateDocumentRequest):
    """
    Surgically invalidate all cached answers that depend on a specific document.

    Traverses the lineage DAG:
    1. Finds all chunks belonging to doc_id.
    2. Evicts all cached queries citing those chunks from L1 exact and L2 semantic indices.
    3. Cleans up reverse lineage pointers.
    """
    from app.cache.lineage import get_lineage_manager

    mgr = get_lineage_manager()
    affected_queries = mgr.get_affected_queries(doc_id=request.doc_id)
    evicted_keys = mgr.invalidate_by_document(request.doc_id)

    return InvalidationResult(
        target=f"doc:{request.doc_id}",
        evicted_cache_keys=evicted_keys,
        evicted_count=len(evicted_keys),
        affected_queries=affected_queries,
    )


@router.post("/cache/invalidate/chunks", response_model=InvalidationResult, tags=["cache"])
async def invalidate_chunks_cache(request: InvalidateChunksRequest):
    """
    Surgically invalidate all cached answers that depend on specific chunk IDs.
    """
    from app.cache.lineage import get_lineage_manager

    mgr = get_lineage_manager()
    all_affected = []
    for cid in request.chunk_ids:
        all_affected.extend(mgr.get_affected_queries(chunk_id=cid))

    evicted_keys = mgr.invalidate_by_chunk_ids(request.chunk_ids)

    return InvalidationResult(
        target=f"chunks:{len(request.chunk_ids)}",
        evicted_cache_keys=evicted_keys,
        evicted_count=len(evicted_keys),
        affected_queries=sorted(list(set(all_affected))),
    )


@router.get("/cache/stats", response_model=CacheStats, tags=["cache"])
async def get_cache_statistics():
    """Retrieve operational telemetry and hit-rate statistics for the semantic cache."""
    from app.cache.semantic_cache import get_semantic_cache

    cache = get_semantic_cache()
    return cache.get_stats()


