"""
Redis-backed job status store.

WHY Redis over Postgres/SQLite for job status:
- Job status is ephemeral, TTL-based data — perfect for Redis.
  We don't need ACID transactions or complex queries.
- Redis is already in our stack (needed for caching in Day 5).
  Using it for job status avoids adding another service.
- JSON serialization of Pydantic models is simple and human-readable
  (you can `redis-cli GET job:<id>` and see the full status).

WHY a fallback to in-memory:
- Tests shouldn't require a running Redis instance.
- Local development can start without Docker if needed.
- The fallback logs a warning so it's obvious when it's active.

INTERFACE:
- Same dict-like pattern as the Day 1 in-memory store.
- get_job(job_id) → JobStatus | None
- set_job(job_id, status) → None
- update_job(job_id, **kwargs) → None
"""

import json
import logging
from typing import Optional

from app.schemas import JobStatus, IngestionStatus

logger = logging.getLogger(__name__)

# Redis client singleton — initialized on first use
_redis_client = None
_use_fallback = False

# In-memory fallback store (used when Redis is unavailable)
_memory_store: dict[str, JobStatus] = {}

# Job status TTL in seconds (24 hours)
JOB_TTL_SECONDS = 86400

# Redis key prefix for job status entries
KEY_PREFIX = "job:"


def _get_redis():
    """
    Lazily connect to Redis. Falls back to in-memory store if connection fails.

    WHY lazy connection instead of connecting at import time:
    - Same reason as lazy model loading: tests that don't use Redis
      shouldn't fail because Redis isn't running.
    - The fallback is logged clearly so it's never a silent failure.
    """
    global _redis_client, _use_fallback

    if _use_fallback:
        return None

    if _redis_client is not None:
        return _redis_client

    try:
        import redis
        from app.config import settings

        _redis_client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,  # Return str instead of bytes
        )
        # Test the connection
        _redis_client.ping()
        logger.info(
            "Connected to Redis at %s:%d", settings.redis_host, settings.redis_port
        )
        return _redis_client
    except Exception as e:
        logger.warning(
            "Redis unavailable (%s), falling back to in-memory job store. "
            "Job status will be lost on restart.",
            e,
        )
        _use_fallback = True
        return None


def _serialize_job(job: JobStatus) -> str:
    """Serialize a JobStatus to JSON string for Redis storage."""
    return job.model_dump_json()


def _deserialize_job(data: str) -> JobStatus:
    """Deserialize a JSON string from Redis into a JobStatus."""
    return JobStatus.model_validate_json(data)


def get_job(job_id: str) -> Optional[JobStatus]:
    """
    Retrieve job status by ID.

    Returns None if the job doesn't exist (or has expired).
    """
    r = _get_redis()

    if r is None:
        # In-memory fallback
        return _memory_store.get(job_id)

    try:
        data = r.get(f"{KEY_PREFIX}{job_id}")
        if data is None:
            return None
        return _deserialize_job(data)
    except Exception as e:
        logger.error("Redis GET failed for job %s: %s", job_id, e)
        return _memory_store.get(job_id)


def set_job(job_id: str, job: JobStatus) -> None:
    """
    Store job status. Sets a TTL so old jobs are automatically cleaned up.

    WHY TTL instead of manual cleanup:
    - We never have to write a cleanup cron job.
    - Old job IDs become 404 naturally after 24 hours.
    - If a client loses a job_id, the storage cost is bounded.
    """
    r = _get_redis()

    # Always update in-memory store (serves as backup)
    _memory_store[job_id] = job

    if r is None:
        return

    try:
        r.setex(
            f"{KEY_PREFIX}{job_id}",
            JOB_TTL_SECONDS,
            _serialize_job(job),
        )
    except Exception as e:
        logger.error("Redis SET failed for job %s: %s", job_id, e)


def update_job(job_id: str, **kwargs) -> None:
    """
    Update specific fields of a job status.

    This is the equivalent of the Day 1 _update_job() function,
    but backed by Redis.
    """
    job = get_job(job_id)
    if job is None:
        logger.warning("Tried to update non-existent job %s", job_id)
        return

    for key, value in kwargs.items():
        if hasattr(job, key):
            setattr(job, key, value)

    from datetime import datetime, timezone

    job.updated_at = datetime.now(timezone.utc).isoformat()

    set_job(job_id, job)


def reset_store() -> None:
    """
    Clear all job data. Used in tests for isolation.

    WHY a reset function instead of just clearing _memory_store:
    - Tests need a clean state between test cases.
    - This function handles both Redis and in-memory cleanup.
    """
    global _memory_store, _redis_client, _use_fallback
    _memory_store.clear()

    # Don't try to flush Redis in tests — just reset the fallback
    if _use_fallback:
        return

    r = _get_redis()
    if r is not None:
        try:
            # Delete only our keys, not the entire Redis DB
            cursor = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"{KEY_PREFIX}*", count=100)
                if keys:
                    r.delete(*keys)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error("Redis cleanup failed: %s", e)


def force_memory_fallback() -> None:
    """Force in-memory fallback mode. Used in tests to avoid Redis dependency."""
    global _use_fallback, _redis_client
    _use_fallback = True
    _redis_client = None
