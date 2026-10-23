"""
FastAPI application entry point.

WHY a factory-style lifespan instead of @app.on_event:
- @app.on_event("startup") is deprecated in FastAPI 0.100+.
- The lifespan context manager gives us clean setup/teardown with
  proper resource cleanup even if startup partially fails.

DAY 2 CHANGES:
- Added Qdrant collection initialization on startup.
- Qdrant init is wrapped in try/except so the app starts even if
  Qdrant is temporarily unavailable (useful for local dev without Docker).
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import settings
from app.api.routes import router as api_router


logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan: runs once at startup (before yield) and
    once at shutdown (after yield).

    Startup tasks:
    1. Initialize Qdrant collection (create if not exists, set up indexes)
    """
    logger.info("Starting Financial Document Intelligence Engine v%s", "0.1.0")
    logger.info("Environment: %s", settings.app_env)

    # Initialize Qdrant collection
    try:
        from app.retrieval.vector_store import init_collection
        init_collection()
        logger.info("Qdrant collection initialized successfully")
    except Exception as e:
        logger.warning(
            "Could not initialize Qdrant collection: %s. "
            "The app will start, but ingestion/search will fail until Qdrant is available.",
            e,
        )

    yield
    logger.info("Shutting down...")


app = FastAPI(
    title="Financial Document Intelligence Engine",
    description=(
        "Agentic RAG system for SEC filings (10-K, 10-Q) and earnings call transcripts. "
        "Every claim is grounded in a source. If evidence is weak, the system retries or abstains."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}
