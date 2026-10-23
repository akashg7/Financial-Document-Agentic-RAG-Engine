# Day 2: Embeddings, Qdrant Vector Store, Dense Search, and Redis Job Store

## What Was Implemented

### 1. Local Embedding Service (`app/retrieval/embedder.py`)
- Loaded `sentence-transformers` model `all-MiniLM-L6-v2` locally (384-dimensional output).
- Singleton pattern: model is loaded once into memory on first request or startup, not reloaded per batch.
- Batch encoding (`embed_texts`) with configurable batch size (`settings.embedding_batch_size`).
- Query encoding (`embed_query`) producing 1D normalized numpy vectors for similarity search.
- Zero-cost: runs completely local on CPU/GPU without external API keys or rate limits.

### 2. Qdrant Vector Store Integration (`app/retrieval/vector_store.py`)
- Connection management to Qdrant server (`http://qdrant:6333` in Docker, `http://localhost:6333` locally) with automatic in-memory fallback for lightweight testing without Docker.
- Idempotent collection initialization (`init_collection`) configured with Cosine distance.
- Payload indexing on `ticker` and `doc_type` for high-performance filtered retrieval.
- Batch upserting (`upsert_chunks`) converting chunk models to Qdrant points with SHA-256 derived deterministic UUIDs and rich metadata payloads.
- Dense semantic search (`search_dense`) with query vectorization, payload filtering, latency measurement, and typed result mapping (`SearchResult`).
- Deduplication hash lookup (`get_existing_hashes`) directly from Qdrant to enable cross-batch and persistent deduplication across service restarts.

### 3. Redis-Backed Job Store (`app/ingestion/redis_job_store.py`)
- Persistent asynchronous ingestion status tracking backed by Redis (`redis://redis:6379/0`).
- Replaced Day 1's ephemeral in-memory dictionary.
- Automated TTL expiration (default 24 hours / 86400s) to prevent memory leak from historical job records.
- Seamless, fault-tolerant in-memory fallback if Redis is unavailable during local development or unit testing.
- JSON serialization/deserialization for full fidelity with `JobStatus` Pydantic models.

### 4. Wired Ingestion Pipeline (`app/ingestion/pipeline.py`)
- Replaced Day 1 embedding placeholder with real embedding generation and Qdrant point upserts.
- Integrated `get_existing_hashes()` to query existing Qdrant chunk hashes, ensuring amendments or duplicate filings don't re-embed known content.
- Updated pipeline state management to write through the Redis job store with real-time chunk counts, embedded counts, and errors.

### 5. Dense Semantic Search API (`POST /api/v1/search/dense`)
- New REST endpoint accepting query text, `top_k`, and optional `ticker` / `doc_type` filters.
- Returns ranked chunks with similarity scores, provenance metadata (doc_id, ticker, fiscal_year, chunk_index), and retrieval latency in milliseconds.

## Files Added/Changed

| File | Purpose |
|---|---|
| `app/retrieval/embedder.py` | Local sentence-transformers embedding service |
| `app/retrieval/vector_store.py` | Qdrant client, collection setup, indexing, upsert, and search |
| `app/retrieval/__init__.py` | Retrieval subpackage exports |
| `app/ingestion/redis_job_store.py` | Redis-backed job store with TTL and in-memory fallback |
| `app/ingestion/pipeline.py` | Wired real embeddings, Qdrant upsert, and Redis job tracking |
| `app/api/routes.py` | Swapped in Redis job store; added `POST /api/v1/search/dense` |
| `app/main.py` | Lifespan collection initialization on app startup |
| `tests/test_embedder.py` | Tests for embedding dimensions, batch consistency, semantic similarity |
| `tests/test_vector_store.py` | Tests for Qdrant init, batch upsert, dense search, filters, dedup hashes |
| `tests/test_redis_job_store.py` | Tests for Redis job store CRUD, TTL, fallback, and serialization |
| `tests/test_api.py` | End-to-end endpoint tests for dense search and Redis-backed ingestion |
| `docs/DECISIONS.md` | Architecture decisions 7 through 10 |
| `docs/versions/v2-day2.md` | Day 2 version summary and changelog |
| `docs/learn/day2-learn.md` | Comprehensive learning guide for Day 2 concepts |

## New Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/search/dense` | Dense vector semantic search with optional metadata filters |

## How to Run and Verify

```bash
# 1. Run full test suite locally
pytest

# 2. Start the multi-container stack
docker compose up --build -d

# 3. Ingest a document
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "doc_type": "10-K", "year": 2023}'

# 4. Search dense vectors
curl -X POST http://localhost:8000/api/v1/search/dense \
  -H "Content-Type: application/json" \
  -d '{"query": "What was the total revenue?", "top_k": 3, "ticker": "AAPL"}'
```
