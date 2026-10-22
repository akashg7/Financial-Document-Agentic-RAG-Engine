# Architecture Decisions Log

Every major tradeoff is recorded here with: the decision, why, what alternatives existed, and when I would change it.

---

## Decision 1: Recursive Character Splitting over LangChain's Built-in Splitter

**Decision**: Implemented recursive character splitting from scratch in `app/ingestion/chunker.py`.

**Why**: 
- For a portfolio project, I need to understand and explain every line. LangChain's `RecursiveCharacterTextSplitter` is a black box.
- The algorithm is the same: try paragraph breaks first → newlines → sentence boundaries → spaces → forced character split.
- Our version is ~80 lines. LangChain's has 200+ lines of generalized code we don't need.

**Alternatives**:
- `langchain.text_splitter.RecursiveCharacterTextSplitter` — more battle-tested, but opaque.
- `tiktoken`-based splitting (split by token count) — better for LLM context windows, but our chunks are for retrieval, not generation. Character-based is simpler and good enough.
- Semantic chunking (split by topic/embedding similarity) — more accurate boundaries but O(n²) in embedding calls. Overkill for Day 1.

**When I'd change it**: If retrieval quality benchmarks (Day 4) show that chunk boundaries are hurting recall, I'd try semantic chunking or experiment with different chunk sizes.

---

## Decision 2: Content-Hash (SHA-256) Deduplication

**Decision**: Each chunk is hashed by its text content. On re-ingestion, chunks with known hashes are skipped.

**Why**:
- SEC filings get amended (10-K/A). An amendment may change only a few paragraphs. Re-embedding the entire document wastes compute.
- Content-hash catches identical content even if chunk boundaries shift (e.g., a new paragraph was inserted before an existing one).
- SHA-256 is collision-resistant, stdlib, and fast enough (hashing is negligible vs. embedding).

**Alternatives**:
- Chunk-ID dedup (skip if `chunk_id` already exists) — fails when chunk boundaries shift.
- Embedding-similarity dedup (skip if a very similar embedding exists) — more flexible but requires vector search during ingestion, which is slower.
- Minhash/LSH for near-duplicate detection — good for fuzzy dedup but adds complexity.

**When I'd change it**: If we see many near-duplicate chunks (e.g., same paragraph with a minor typo fix), I'd add Minhash on top of exact hashing.

---

## Decision 3: SEC EDGAR + HuggingFace as Data Sources

**Decision**: Use SEC EDGAR direct download for 10-K/10-Q filings and `Rogersurf/earnings-call-transcripts` from HuggingFace for earnings calls.

**Why**:
- SEC EDGAR is the authoritative source for filings. Public domain, no license issues.
- No single HuggingFace dataset has both filings + transcripts with clean metadata.
- `Rogersurf/earnings-call-transcripts` is cleaned, has ticker/quarter/date fields, and is formatted for NLP/RAG.

**Alternatives**:
- `astr010/sec-10k-markdown-compressed` — clean markdown but 10-K only, compressed format adds complexity.
- `khaihernlow/financial-reports-sec` — sentence-level with sentiment, but restructured away from original doc structure.
- SEC EDGAR full-text search API — could replace HuggingFace, but earnings calls aren't on EDGAR.

**When I'd change it**: If the HuggingFace dataset has quality issues (missing tickers, truncated transcripts), I'd scrape earnings calls from company IR pages with proper attribution.

---

## Decision 4: In-Memory Job Store (Day 1 only)

**Decision**: Job status is stored in a Python dict (`job_store`). Will move to Redis in Day 2.

**Why**: Zero-config for Day 1. Gets the pipeline working end-to-end without adding Redis complexity on day one.

**Limitation**: Job status is lost if the API process restarts.

**When I'd change it**: Day 2 — already planned.

---

## Decision 5: FastAPI BackgroundTasks over Celery/RQ

**Decision**: Use FastAPI's built-in `BackgroundTasks` for async ingestion.

**Why**:
- Single-worker dev server. No need for distributed task execution yet.
- Zero additional infrastructure (no Redis broker, no separate worker process).
- Same interface as a task queue: fire-and-forget with status polling.

**Alternatives**:
- Celery + Redis — production-grade, supports retries, but heavy for a portfolio project.
- RQ (Redis Queue) — lighter than Celery, but still needs a separate worker.
- asyncio tasks — would work but harder to track status and handle failures.

**When I'd change it**: If we need parallel ingestion of many documents, or if the API worker runs out of thread pool capacity under load.

---

## Decision 6: Docker Compose with Health Checks

**Decision**: All three services (API, Qdrant, Redis) run via Docker Compose with health checks and `depends_on` conditions.

**Why**:
- One-command startup (`docker compose up`).
- Health checks prevent the API from starting before Qdrant/Redis are ready, avoiding connection errors on first request.
- Volumes persist data across restarts.

**Alternatives**:
- Run services natively (no Docker) — works but "it runs on my machine" problem.
- Kubernetes — overkill for a portfolio project.
- Docker without Compose — more manual, no dependency ordering.

**When I'd change it**: Never for this project. Docker Compose is the right tool for local multi-service development.
