"""
Tests for the configuration module.

Verifies that:
1. Default values are set correctly
2. Environment variables override defaults
3. The settings singleton is accessible
"""

import os
import pytest

from app.config import Settings


class TestSettings:
    """Tests for the Settings configuration."""

    def test_default_values(self):
        """Settings should have sensible defaults even without .env."""
        s = Settings(
            _env_file=None,  # Don't read any .env file during tests
        )
        assert s.qdrant_host == "localhost"
        assert s.qdrant_port == 6333
        assert s.redis_host == "localhost"
        assert s.redis_port == 6379
        assert s.chunk_size == 512
        assert s.chunk_overlap == 64
        assert s.embedding_model_name == "all-MiniLM-L6-v2"

    def test_env_override(self, monkeypatch):
        """Environment variables should override defaults."""
        monkeypatch.setenv("QDRANT_HOST", "my-qdrant-server")
        monkeypatch.setenv("CHUNK_SIZE", "256")
        s = Settings(_env_file=None)
        assert s.qdrant_host == "my-qdrant-server"
        assert s.chunk_size == 256

    def test_chunk_overlap_less_than_size(self):
        """Overlap should be less than chunk size (sanity check)."""
        s = Settings(_env_file=None)
        assert s.chunk_overlap < s.chunk_size

    def test_settings_singleton_import(self):
        """The settings singleton should be importable."""
        from app.config import settings
        assert settings is not None
        assert isinstance(settings.qdrant_port, int)
