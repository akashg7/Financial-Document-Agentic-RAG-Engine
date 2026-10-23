"""
Tests for the Redis-backed job store.

These tests use the in-memory fallback mode (no Redis needed) to verify:
1. Job creation and retrieval
2. Job updates (partial field updates)
3. Serialization/deserialization of JobStatus
4. Fallback behavior when Redis is unavailable
5. Store reset for test isolation

WHY we test the fallback path:
- The in-memory fallback IS the path used in tests and local dev.
  We need to ensure it behaves identically to Redis for all operations.
- The Redis-specific path is tested implicitly in integration tests
  (Docker Compose + pytest).
"""

import pytest

from app.schemas import IngestionStatus, JobStatus
from app.ingestion.redis_job_store import (
    get_job,
    set_job,
    update_job,
    reset_store,
    force_memory_fallback,
    _serialize_job,
    _deserialize_job,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture(autouse=True)
def clean_store():
    """Reset the job store before each test and force in-memory mode."""
    force_memory_fallback()
    reset_store()
    yield
    reset_store()


def _make_job(job_id: str = "test-job-123", status: IngestionStatus = IngestionStatus.PENDING) -> JobStatus:
    """Helper to create a test JobStatus."""
    return JobStatus(
        job_id=job_id,
        status=status,
        created_at="2024-01-01T00:00:00Z",
        updated_at="2024-01-01T00:00:00Z",
    )


# ============================================================
# Basic CRUD Tests
# ============================================================

class TestJobStoreCRUD:
    """Tests for basic create/read operations."""

    def test_set_and_get(self):
        """Setting a job and getting it back should return the same data."""
        job = _make_job()
        set_job("test-job-123", job)
        retrieved = get_job("test-job-123")
        assert retrieved is not None
        assert retrieved.job_id == "test-job-123"
        assert retrieved.status == IngestionStatus.PENDING

    def test_get_nonexistent(self):
        """Getting a non-existent job should return None."""
        assert get_job("nonexistent") is None

    def test_overwrite(self):
        """Setting a job with the same ID should overwrite."""
        job1 = _make_job(status=IngestionStatus.PENDING)
        job2 = _make_job(status=IngestionStatus.COMPLETED)
        set_job("test-job-123", job1)
        set_job("test-job-123", job2)
        retrieved = get_job("test-job-123")
        assert retrieved.status == IngestionStatus.COMPLETED


# ============================================================
# Update Tests
# ============================================================

class TestJobUpdate:
    """Tests for partial field updates."""

    def test_update_status(self):
        """update_job should change specific fields."""
        set_job("j1", _make_job("j1"))
        update_job("j1", status=IngestionStatus.PROCESSING)
        job = get_job("j1")
        assert job.status == IngestionStatus.PROCESSING

    def test_update_multiple_fields(self):
        """update_job should handle multiple fields at once."""
        set_job("j1", _make_job("j1"))
        update_job("j1", total_documents=5, processed_documents=3, total_chunks=42)
        job = get_job("j1")
        assert job.total_documents == 5
        assert job.processed_documents == 3
        assert job.total_chunks == 42

    def test_update_sets_updated_at(self):
        """update_job should refresh the updated_at timestamp."""
        set_job("j1", _make_job("j1"))
        old_time = get_job("j1").updated_at
        update_job("j1", status=IngestionStatus.COMPLETED)
        new_time = get_job("j1").updated_at
        assert new_time != old_time

    def test_update_nonexistent_job(self):
        """Updating a non-existent job should not raise."""
        update_job("nonexistent", status=IngestionStatus.FAILED)
        # Should silently do nothing
        assert get_job("nonexistent") is None

    def test_update_errors_list(self):
        """update_job should handle list fields like errors."""
        set_job("j1", _make_job("j1"))
        update_job("j1", errors=["Something went wrong"])
        job = get_job("j1")
        assert job.errors == ["Something went wrong"]


# ============================================================
# Serialization Tests
# ============================================================

class TestSerialization:
    """Tests for JSON serialization/deserialization of JobStatus."""

    def test_roundtrip(self):
        """Serializing and deserializing should produce equivalent objects."""
        original = JobStatus(
            job_id="rt-test",
            status=IngestionStatus.PROCESSING,
            total_documents=10,
            processed_documents=5,
            total_chunks=100,
            skipped_chunks=3,
            errors=["Warning: slow network"],
            created_at="2024-01-01T00:00:00Z",
            updated_at="2024-01-01T12:00:00Z",
        )
        json_str = _serialize_job(original)
        restored = _deserialize_job(json_str)

        assert restored.job_id == original.job_id
        assert restored.status == original.status
        assert restored.total_documents == original.total_documents
        assert restored.processed_documents == original.processed_documents
        assert restored.total_chunks == original.total_chunks
        assert restored.skipped_chunks == original.skipped_chunks
        assert restored.errors == original.errors

    def test_serialize_produces_json_string(self):
        """_serialize_job should return a valid JSON string."""
        import json
        job = _make_job()
        json_str = _serialize_job(job)
        parsed = json.loads(json_str)
        assert parsed["job_id"] == "test-job-123"


# ============================================================
# Reset Tests
# ============================================================

class TestStoreReset:
    """Tests for store cleanup."""

    def test_reset_clears_all_jobs(self):
        """reset_store should remove all stored jobs."""
        set_job("j1", _make_job("j1"))
        set_job("j2", _make_job("j2"))
        reset_store()
        assert get_job("j1") is None
        assert get_job("j2") is None
