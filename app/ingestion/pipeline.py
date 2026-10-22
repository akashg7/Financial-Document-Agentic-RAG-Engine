"""
Ingestion pipeline: orchestrates loading, chunking, dedup, and (Day 2) embedding.

This module is the "conductor" — it doesn't do the heavy lifting itself,
but calls the right modules in the right order and tracks job status.

FLOW:
1. Load documents from the requested source (SEC EDGAR or HuggingFace)
2. Chunk each document with metadata
3. Deduplicate against existing chunk hashes
4. (Day 2) Embed and upsert to Qdrant
5. Update job status throughout

WHY a separate pipeline module instead of inlining in the route handler:
- Testability: we can test the pipeline without spinning up FastAPI.
- Reusability: the same pipeline can be called from CLI, tests, or a worker queue.
- Separation of concerns: the route handles HTTP; the pipeline handles business logic.
"""

import logging
from datetime import datetime, timezone

from app.config import settings
from app.schemas import (
    IngestionRequest,
    IngestionStatus,
    JobStatus,
    DocumentType,
)
from app.ingestion.job_store import job_store
from app.ingestion.loader import (
    load_sec_edgar_documents,
    load_huggingface_earnings_transcripts,
)
from app.ingestion.chunker import chunk_document, deduplicate_chunks

logger = logging.getLogger(__name__)

# Global set of content hashes we've already ingested.
# Day 2: move this to Qdrant payload queries or a Redis set for persistence.
_existing_hashes: set[str] = set()


def _update_job(job_id: str, **kwargs):
    """Update job status fields atomically."""
    if job_id not in job_store:
        return
    job = job_store[job_id]
    for key, value in kwargs.items():
        setattr(job, key, value)
    job.updated_at = datetime.now(timezone.utc).isoformat()


def run_ingestion_pipeline(job_id: str, request: IngestionRequest):
    """
    Main ingestion pipeline. Runs as a background task.

    This function is called by FastAPI BackgroundTasks in a thread pool.
    It must not raise — any exception is caught and recorded in job status.
    """
    global _existing_hashes

    logger.info("Starting ingestion pipeline for job %s", job_id)
    _update_job(job_id, status=IngestionStatus.PROCESSING)

    try:
        # ── Step 1: Load documents ──────────────────────────────────
        documents = _load_documents(request)

        if not documents:
            _update_job(
                job_id,
                status=IngestionStatus.COMPLETED,
                total_documents=0,
                errors=["No documents found for the given parameters"],
            )
            return

        _update_job(job_id, total_documents=len(documents))
        logger.info("Loaded %d documents for job %s", len(documents), job_id)

        # ── Step 2: Chunk and deduplicate ───────────────────────────
        all_chunks_to_insert = []
        total_skipped = 0

        for i, (text, doc_metadata) in enumerate(documents):
            chunks = chunk_document(
                text=text,
                doc_metadata=doc_metadata,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )

            to_insert, skipped = deduplicate_chunks(chunks, _existing_hashes)
            all_chunks_to_insert.extend(to_insert)
            total_skipped += len(skipped)

            _update_job(
                job_id,
                processed_documents=i + 1,
                total_chunks=len(all_chunks_to_insert),
                skipped_chunks=total_skipped,
            )

        logger.info(
            "Job %s: %d chunks to insert, %d skipped (dedup)",
            job_id,
            len(all_chunks_to_insert),
            total_skipped,
        )

        # ── Step 3: Embed and upsert (Day 2) ───────────────────────
        # For now, we just log the chunks. Day 2 adds embedding + Qdrant upsert.
        # The chunks are fully prepared with metadata and content hashes.
        logger.info(
            "Job %s: Day 1 — chunking complete. Embedding will be added in Day 2.",
            job_id,
        )

        _update_job(
            job_id,
            status=IngestionStatus.COMPLETED,
            total_chunks=len(all_chunks_to_insert),
            skipped_chunks=total_skipped,
        )
        logger.info("Ingestion job %s completed successfully", job_id)

    except Exception as e:
        logger.exception("Ingestion job %s failed: %s", job_id, e)
        _update_job(
            job_id,
            status=IngestionStatus.FAILED,
            errors=[str(e)],
        )


def _load_documents(request: IngestionRequest) -> list:
    """
    Dispatch to the correct loader based on the request source.

    WHY a dispatcher instead of if/else in the pipeline:
    - Adding a new source (e.g., local files) only requires adding a
      new branch here, not modifying the pipeline logic.
    """
    if request.source == "sec_edgar":
        tickers = request.tickers or list(
            # Default: use first 5 tickers from our CIK map
            __import__("app.ingestion.loader", fromlist=["TICKER_TO_CIK"]).TICKER_TO_CIK.keys()
        )[:5]
        return load_sec_edgar_documents(
            tickers=tickers,
            filing_types=request.filing_types,
            max_per_ticker=max(1, request.max_documents // len(tickers)),
        )

    elif request.source == "huggingface":
        return load_huggingface_earnings_transcripts(
            max_documents=request.max_documents,
            tickers=request.tickers or None,
        )

    else:
        logger.error("Unknown source: %s", request.source)
        return []
