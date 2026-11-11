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

DAY 9 CHANGES:
- Configured SlowAPI rate limiter with SlowAPIMiddleware and custom 429 handler.
- Initialized optional LangSmith distributed tracing in lifespan.
- Exposed GET /metrics endpoint for Prometheus scrapers with domain-specific metrics.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import settings
from app.api.routes import router as api_router
from app.limiter import limiter
from app.observability.tracing import setup_langsmith
from app.observability.metrics import RATE_LIMITED_REQUESTS


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
    1. Configure LangSmith distributed tracing if enabled.
    2. Initialize Qdrant collection (create if not exists, set up indexes).
    """
    logger.info("Starting Financial Document Intelligence Engine v%s", "0.1.0")
    logger.info("Environment: %s", settings.app_env)

    # 1. Initialize LangSmith tracing
    try:
        setup_langsmith()
    except Exception as e:
        logger.warning("Could not initialize LangSmith tracing: %s", e)

    # 2. Initialize Qdrant collection
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

# Attach rate limiter to FastAPI state & register middleware
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    """
    Custom handler for rate limit violations.

    Tracks rejected requests in Prometheus and returns a structured JSON payload
    with 429 status code and human-readable detail.
    """
    RATE_LIMITED_REQUESTS.inc()
    logger.warning("Rate limit exceeded for %s on %s: %s", request.client.host if request.client else "unknown", request.url.path, exc.detail)
    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "detail": f"Rate limit exceeded: {exc.detail}",
            "retry_after_seconds": 60,
        },
    )


app.include_router(api_router, prefix="/api/v1")


@app.get("/health", tags=["system"])
async def health_check():
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/metrics", tags=["observability"])
async def metrics():
    """
    Prometheus scraping endpoint.

    Exposes real-time application metrics including query latencies,
    confidence distributions, cache hit/miss counters, retrieval statistics,
    and rate limiting counters formatted according to Prometheus standards.
    """
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )
