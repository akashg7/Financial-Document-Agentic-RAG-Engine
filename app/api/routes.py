"""
API routes for the Financial Document Intelligence Engine.

Day 1: health check and ingestion endpoints.
Later days add /search/dense, /search/hybrid, /query, etc.
"""

import uuid
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException

from app.schemas import (
    IngestionRequest,
    IngestionResponse,
    IngestionStatus,
    JobStatus,
)
from app.ingestion.pipeline import run_ingestion_pipeline
from app.ingestion.job_store import job_store

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

    # Initialize job status in the in-memory store (Day 2: move to Redis)
    job_store[job_id] = JobStatus(
        job_id=job_id,
        status=IngestionStatus.PENDING,
        created_at=now,
        updated_at=now,
    )

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
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job_store[job_id]
