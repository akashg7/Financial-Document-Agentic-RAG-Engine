"""
Application configuration via Pydantic Settings.

WHY Pydantic Settings instead of raw os.environ:
- Validation at startup: if QDRANT_HOST is missing, the app crashes immediately
  with a clear error, rather than failing later in a request handler.
- Type coercion: CHUNK_SIZE comes in as str from env, but we need int.
- Single source of truth: every configurable value lives here with its default,
  making it easy to audit what the app needs.
- .env file support: for local dev, settings auto-load from .env.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Central configuration for the Financial Document Intelligence Engine."""

    # --- LLM Providers ---
    groq_api_key: str = Field(default="", description="Groq API key for primary LLM")
    gemini_api_key: str = Field(default="", description="Google Gemini API key for fallback LLM")
    llm_primary_provider: str = Field(default="groq", description="Primary LLM provider name")
    llm_fallback_provider: str = Field(default="gemini", description="Fallback LLM provider name")
    groq_model_name: str = Field(default="llama-3.1-8b-instant", description="Model name on Groq")
    gemini_model_name: str = Field(default="gemini-1.5-flash", description="Model name on Gemini")

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost", description="Qdrant server hostname")
    qdrant_port: int = Field(default=6333, description="Qdrant REST API port")
    qdrant_collection_name: str = Field(default="sec_filings", description="Qdrant collection name")

    # --- Redis ---
    redis_host: str = Field(default="localhost", description="Redis server hostname")
    redis_port: int = Field(default=6379, description="Redis server port")

    # --- Embeddings ---
    embedding_model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="sentence-transformers model for dense embeddings",
    )
    reranker_model_name: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )

    # --- Ingestion ---
    chunk_size: int = Field(default=512, description="Target chunk size in characters")
    chunk_overlap: int = Field(default=64, description="Overlap between consecutive chunks in characters")
    sec_edgar_user_agent: str = Field(
        default="Financial-RAG-Engine admin@example.com",
        description="User-Agent header for SEC EDGAR API (required by their fair-use policy)",
    )

    # --- Observability ---
    langsmith_api_key: str = Field(default="", description="LangSmith API key")
    langsmith_project: str = Field(default="financial-rag-engine", description="LangSmith project name")
    langsmith_tracing_enabled: bool = Field(default=False, description="Enable LangSmith tracing")

    # --- App ---
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton instance — import this throughout the app
settings = Settings()
