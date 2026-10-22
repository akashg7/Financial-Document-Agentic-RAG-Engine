"""
In-memory job status store.

WHY in-memory first instead of jumping straight to Redis:
- Day 1 focuses on getting the pipeline working end-to-end. An in-memory
  dict is zero-config and lets us iterate fast.
- Day 2 swaps this for Redis so job status survives API restarts. The
  interface (dict-like access) stays the same thanks to Python's duck typing.

LIMITATION: job status is lost if the API process restarts. Acceptable for
Day 1 since we're running a single-worker dev server.
"""

from app.schemas import JobStatus

# Simple dict-based store. Keys are job_id strings, values are JobStatus objects.
# Thread-safe for our use case: FastAPI's BackgroundTasks runs in a thread pool,
# and Python's GIL protects dict operations from race conditions on CPython.
job_store: dict[str, JobStatus] = {}
