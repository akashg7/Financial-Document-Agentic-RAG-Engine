# Day 1: Foundation — Repo, Docker, Config, Loaders, Chunker, Dedup

## What Was Implemented

### Repository Structure
Created the full project skeleton:
- `/app` — FastAPI application with subpackages for each system component
- `/tests` — pytest test suite
- `/eval` — golden set and benchmark scripts
- `/results` — timestamped benchmark outputs
- `/docs` — architecture, decisions, version logs, learning materials

### Docker Compose Stack
Three services running with one command:
- **FastAPI** (port 8000) — application server with hot reload
- **Qdrant** (port 6333) — vector database for dense + sparse vectors
- **Redis** (port 6379) — cache, job status, session memory

All services have health checks. API waits for Qdrant and Redis to be healthy.

### Configuration (`app/config.py`)
Pydantic Settings with:
- Environment variable loading from `.env`
- Type validation at startup
- Defaults for all configurable values

### Document Loaders (`app/ingestion/loader.py`)
Two data sources:
1. **SEC EDGAR** — fetches 10-K/10-Q filings via submissions API, downloads HTML, extracts text with BeautifulSoup. Polite rate limiting (0.15s between requests).
2. **HuggingFace** — loads `Rogersurf/earnings-call-transcripts` via `datasets` library.

### Chunker (`app/ingestion/chunker.py`)
- Recursive character splitting: paragraph → newline → sentence → space → forced split
- Configurable chunk_size (default 512) and overlap (default 64)
- Content-hash deduplication via SHA-256
- Full metadata propagation from document to chunk

### Ingestion Pipeline (`app/ingestion/pipeline.py`)
- Orchestrates: load → chunk → dedup → (Day 2: embed)
- Runs as FastAPI BackgroundTask
- Tracks job status in in-memory store

### API Endpoints
- `GET /health` — liveness check
- `POST /api/v1/ingest` — start async ingestion job
- `GET /api/v1/job/{job_id}` — poll job status

## Files Added/Changed

| File | Purpose |
|---|---|
| `docker-compose.yml` | Three-service stack with health checks |
| `Dockerfile` | Python 3.11-slim with layer-cached deps |
| `requirements.txt` | Pinned dependencies |
| `Makefile` | Dev shortcuts (up, down, test, ingest, clean) |
| `.env.example` | Config template |
| `.gitignore` | Python + Docker + data exclusions |
| `app/__init__.py` | Package root |
| `app/config.py` | Pydantic Settings |
| `app/schemas.py` | All Pydantic models |
| `app/main.py` | FastAPI app with lifespan |
| `app/api/routes.py` | Ingest + job status endpoints |
| `app/ingestion/loader.py` | SEC EDGAR + HuggingFace loaders |
| `app/ingestion/chunker.py` | Recursive splitter + dedup |
| `app/ingestion/pipeline.py` | Pipeline orchestrator |
| `app/ingestion/job_store.py` | In-memory job status |
| `tests/test_chunker.py` | 15 tests for chunking + dedup |
| `tests/test_api.py` | 5 tests for API endpoints |
| `tests/test_config.py` | 4 tests for configuration |
| `eval/golden_set.jsonl` | Seed with 3 example questions |
| `docs/DECISIONS.md` | 6 architecture decisions |
| `docs/architecture.md` | Mermaid diagrams |

## New Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/api/v1/ingest` | Start ingestion job |
| GET | `/api/v1/job/{job_id}` | Poll job status |

## Config Changes
New `.env.example` with all configurable values. Copy to `.env` and fill in API keys.

## How to Run and Verify

```bash
# 1. Copy env config
cp .env.example .env

# 2. Start the stack
docker compose up --build -d
# or: make up

# 3. Verify services are running
curl http://localhost:8000/health
# → {"status":"healthy","version":"0.1.0"}

# 4. Check Qdrant is up
curl http://localhost:6333/readyz
# → ok

# 5. Check Redis is up
docker compose exec redis redis-cli ping
# → PONG

# 6. Run tests (locally, with pip install -r requirements.txt first)
pytest tests/ -v

# 7. Test ingestion (will chunk but not embed yet — Day 2)
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "sec_edgar", "tickers": ["AAPL"], "filing_types": ["10-K"], "max_documents": 1}'

# 8. Check job status
curl http://localhost:8000/api/v1/job/<job_id_from_above>
```

## Known Limitations

1. **No embedding yet** — chunks are created but not stored in Qdrant. Day 2.
2. **In-memory job store** — job status lost on restart. Day 2 moves to Redis.
3. **No search endpoints** — Day 2 adds `/search/dense`.
4. **HuggingFace dataset** — not yet verified; may need field name adjustments when tested live.
5. **SEC EDGAR CIK map** — hardcoded for 20 tickers. Production would use EDGAR's company search.

## Metrics Measured Today

- **Test results**: See results from `pytest tests/ -v` (run locally)
- No retrieval or latency metrics yet — those start at Day 2/Day 4.
