"""
API routes for the Financial Document Intelligence Engine.

Day 1: health check and ingestion endpoints.
Day 2: added POST /search/dense for semantic vector search,
       swapped in-memory job store for Redis-backed store.
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
)
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
