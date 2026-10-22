# Day 1 — Learning Guide

## 1. What We Built Today in Plain English

We set up the foundation for a system that reads SEC financial filings (10-K, 10-Q) and earnings call transcripts, then breaks them into small searchable pieces ("chunks") that we can later embed as vectors and retrieve with AI.

Think of it like building a library: today we built the shelves (Docker infrastructure), created a system for cataloging books (metadata schemas), hired two librarians who know where to find documents (SEC EDGAR loader, HuggingFace loader), built a machine that cuts books into index cards (chunker), and added a stamp system so we never file the same card twice (content-hash dedup).

Tomorrow we'll add the "card catalog search" — embedding these chunks into Qdrant so we can find them by meaning.

---

## 2. Why Each Design Decision Was Made

### Why Docker Compose instead of running services natively?
**Decision**: All three services (API, Qdrant, Redis) run in Docker containers orchestrated by Docker Compose.

**Why**: "It works on my machine" is the #1 onboarding killer. Docker Compose guarantees that anyone cloning this repo gets the exact same environment — same Python version, same Qdrant version, same Redis version. Health checks in the compose file ensure the API doesn't start until its dependencies are ready.

**Alternative**: Install Python, Qdrant, and Redis natively. Faster dev cycle (no container overhead), but version mismatches cause subtle bugs. For a portfolio project that interviewers might clone, Docker is non-negotiable.

### Why Pydantic Settings instead of `os.getenv()`?
**Decision**: All configuration lives in one `Settings` class that validates at startup.

**Why**: If you misspell `QDRANT_HOST` in your .env, `os.getenv("QDRANT_HOST")` silently returns `None` — and your app crashes later with a cryptic `ConnectionRefusedError`. Pydantic Settings fails **immediately** with a clear error: "Field 'qdrant_host' is required." This "fail fast" principle saves hours of debugging.

**Alternative**: `python-dotenv` with manual `os.getenv()` calls. Works, but you lose type coercion (everything is a string) and centralized documentation of what the app needs.

### Why recursive character splitting instead of fixed-size windows?
**Decision**: We split text by trying paragraph breaks first, then newlines, then sentences, then spaces.

**Why**: A fixed 512-character window cuts mid-sentence. The chunk "...reported revenue of $383.3" is useless without "billion for fiscal year 2023." Recursive splitting respects natural language boundaries. The hierarchy (paragraph > newline > sentence > space) means we get the biggest coherent unit that fits in the chunk size.

**Alternative**: Token-based splitting (split every 128 tokens). Better if your downstream LLM has a strict token limit, but our chunks are for *retrieval*, not *generation*. Character-based is simpler and adequate.

### Why SHA-256 for content hashing?
**Decision**: Each chunk is hashed by its text content, and we skip chunks whose hash already exists.

**Why**: SEC filings get amended. A 10-K/A might change one risk factor paragraph out of 200. Without content-hash dedup, re-ingesting the amendment re-embeds 200 chunks (wasting ~$0 since we use local models, but wasting 30+ minutes of compute). With it, we re-embed only the changed chunks.

**Alternative**: Use the chunk_id (e.g., "AAPL_10-K_2023_chunk_0042") for dedup. Fails when chunk boundaries shift — inserting a paragraph before chunk 42 makes the old chunk 42 become chunk 43, so the "same content" gets a new ID and is treated as new.

### Why two separate loaders instead of one generic loader?
**Decision**: `load_sec_edgar_documents()` and `load_huggingface_earnings_transcripts()` are separate functions.

**Why**: SEC EDGAR serves HTML via REST API with pagination. HuggingFace serves Parquet via the `datasets` library with streaming. The data formats, authentication, rate limits, and error modes are completely different. Forcing them into one class would mean a lot of `if source == "edgar": ... else: ...` branching. Separate functions are simpler, testable independently, and follow the Single Responsibility Principle.

---

## 3. Key Concepts Explained from First Principles

### What is a "chunk" and why do we need chunking?

Imagine you have a 100-page 10-K filing (roughly 50,000 words). You want to find the paragraph that mentions "revenue from services segment." You can't feed the entire 50,000-word document into a vector database — it would become one giant vector that represents the *average* meaning of everything, losing the specific details.

**Chunking** breaks the document into smaller pieces (typically 200-1000 characters each). Each chunk gets its own vector. Now when someone asks about "services revenue," the vector search finds the specific chunk that discusses it, not the whole document.

**Trade-off**: Chunk too small → fragments lose context ("$85 billion" without knowing what it refers to). Chunk too large → vectors become too general, reducing retrieval precision.

```python
# Example: a 1000-char text split into ~200-char chunks with 50-char overlap

text = "Apple reported total revenue of $383B... [200 chars] ...Services grew 9%... [200 chars] ...iPhone was $200B..."

chunks = [
    "Apple reported total revenue of $383B...",        # chunk 0
    "...383B for fiscal 2023. Services grew 9%...",    # chunk 1 (overlap with 0)
    "...grew 9% year over year. iPhone was $200B...",  # chunk 2 (overlap with 1)
]
```

### What is content-addressable hashing?

A content-addressable hash is a fingerprint of data computed from the data itself. The same input always produces the same hash. Different inputs (almost certainly) produce different hashes.

```python
import hashlib

# Same input → same hash (deterministic)
hashlib.sha256(b"Revenue was $383B").hexdigest()
# → "a1b2c3d4..."  (always this exact value)

# Different input → different hash
hashlib.sha256(b"Revenue was $384B").hexdigest()
# → "e5f6g7h8..."  (completely different)
```

**Why "content-addressable"?** Because you can use the hash to *address* (find) the content. If you know a chunk's hash, you know whether you've seen that exact text before — without comparing the full text.

**Used in**: Git (every commit is a SHA-1 hash of its contents), Docker images, IPFS, and our dedup system.

### What is the SEC EDGAR API?

SEC EDGAR (Electronic Data Gathering, Analysis, and Retrieval) is the U.S. government's database of all public company filings. Every public company must file:

- **10-K**: Annual report. Comprehensive overview of financials, risks, strategy. ~100 pages.
- **10-Q**: Quarterly report. Financial update, shorter than 10-K. ~40 pages.
- **8-K**: Current report. Material events (acquisitions, CEO changes). Variable length.

The API is free, public domain, but has a fair-use policy:
1. Send a `User-Agent` header with your name + email.
2. Stay under 10 requests/second.
3. Don't scrape the full database.

```
# Example: get all Apple filings
GET https://data.sec.gov/submissions/CIK0000320193.json
User-Agent: MyApp admin@example.com
```

### What is Pydantic and why does it matter?

Pydantic is Python's most popular data validation library. It lets you define **schemas** — structured descriptions of what your data looks like:

```python
from pydantic import BaseModel, Field

class ChunkMetadata(BaseModel):
    chunk_id: str                              # Required string
    page: int | None = None                    # Optional integer
    score: float = Field(ge=0.0, le=1.0)       # Float between 0 and 1

# Pydantic validates automatically:
meta = ChunkMetadata(chunk_id="abc", page="42", score=0.8)
print(meta.page)   # 42 (converted from string to int!)
print(type(meta.page))  # <class 'int'>

ChunkMetadata(chunk_id="abc", score=1.5)  # ❌ ValidationError: score must be ≤ 1.0
```

In FastAPI, Pydantic schemas serve triple duty:
1. **Request validation**: reject malformed inputs before your code runs.
2. **Response serialization**: convert Python objects to JSON automatically.
3. **API documentation**: OpenAPI/Swagger docs are generated from schema annotations.

---

## 4. Step-by-Step Walkthrough of a Real Request

### Scenario: POST /api/v1/ingest with `{"source": "sec_edgar", "tickers": ["AAPL"], "filing_types": ["10-K"], "max_documents": 1}`

**Step 1: Request hits FastAPI** (`app/main.py` → `app/api/routes.py`)
- FastAPI deserializes the JSON body into an `IngestionRequest` Pydantic model.
- Pydantic validates: `source` is a string ✓, `tickers` is a list of strings ✓, `filing_types` contains valid `DocumentType` enums ✓.
- If validation fails, FastAPI returns 422 automatically.

**Step 2: Route handler creates a job** (`app/api/routes.py::ingest_documents`)
- Generates a UUID: `job_id = "a1b2c3d4-..."`
- Creates a `JobStatus(status="pending")` in `job_store` (a dict).
- Calls `background_tasks.add_task(run_ingestion_pipeline, job_id, request)`.
- Returns `{"job_id": "a1b2c3d4-...", "status": "pending"}` to the client immediately.

**Step 3: Background thread runs the pipeline** (`app/ingestion/pipeline.py::run_ingestion_pipeline`)
- Updates status to `"processing"`.
- Calls `_load_documents(request)`, which dispatches to `load_sec_edgar_documents()`.

**Step 4: SEC EDGAR loader fetches filings** (`app/ingestion/loader.py::load_sec_edgar_documents`)
- Looks up "AAPL" → CIK "0000320193" in the static map.
- `GET https://data.sec.gov/submissions/CIK0000320193.json` with polite User-Agent.
- Parses the JSON to find the most recent 10-K filing.
- Downloads the primary document HTML.
- BeautifulSoup extracts clean text: `soup.get_text(separator="\n", strip=True)`.
- Returns `[(text, DocumentMetadata(doc_id="AAPL_10-K_2023-10-27", ...))]`.

**Step 5: Chunker splits the text** (`app/ingestion/chunker.py::chunk_document`)
- `recursive_character_split(text, chunk_size=512, chunk_overlap=64)` runs.
- Tries splitting on `\n\n` (paragraph breaks). Each segment is checked against 512 chars.
- Segments still too long are recursively split on `\n`, then `. `, then ` `.
- Small segments are merged back up to chunk_size.
- Each chunk gets: `chunk_id = "AAPL_10-K_2023-10-27_chunk_0042"`, `content_hash = SHA256(text)`.

**Step 6: Deduplication** (`app/ingestion/chunker.py::deduplicate_chunks`)
- Compare each chunk's `content_hash` against `_existing_hashes` (empty on first run).
- First run: all chunks are new → all inserted. Hashes added to the set.
- Second run (re-ingest): unchanged chunks are skipped.

**Step 7: Job completes** (`app/ingestion/pipeline.py`)
- Updates `job_store[job_id]` with `status="completed"`, `total_chunks=N`, `skipped_chunks=0`.
- Client polls `GET /api/v1/job/a1b2c3d4-...` and sees the final status.

---

## 5. Interview Questions with Strong Answers

### Q1: Why did you choose recursive character splitting over token-based splitting?

**A**: Our chunks are for *retrieval*, not *generation*. Token-based splitting matters when you need to fit chunks into an LLM's context window with exact token counts. For retrieval, we want semantically coherent chunks — splitting on paragraph and sentence boundaries preserves meaning better than splitting every 128 tokens, which might cut "reported revenue of $383" from "billion." Our character-based approach is simpler, has no tokenizer dependency, and produces chunks that read naturally.

### Q2: What happens if you ingest the same document twice?

**A**: Nothing is duplicated. Each chunk is hashed by its text content (SHA-256). On the second ingestion, every chunk's hash matches an existing hash, so it's skipped. The job completes with `skipped_chunks = N` and `total_chunks = 0` new inserts. This is content-hash dedup, not ID-based dedup — it catches identical content even if chunk boundaries shift due to formatting changes.

### Q3: Why not use LangChain's RecursiveCharacterTextSplitter?

**A**: I wanted to understand and be able to explain every line. LangChain's splitter does essentially the same thing (try separators in order, merge small chunks, add overlap), but it's wrapped in 200+ lines of generalized code handling transformers, token counting, metadata, and multiple inheritance. My implementation is ~80 lines doing exactly what I need. For a portfolio project, clarity > code reuse.

### Q4: How does your system handle an amended SEC filing (10-K/A)?

**A**: When a 10-K/A is re-ingested, the chunker produces new chunks. Content-hash dedup compares each chunk's SHA-256 against the existing set. Unchanged chunks (the majority) are skipped. Only truly modified chunks get new hashes and are embedded + upserted to Qdrant. This is important because a 10-K amendment might change one risk factor paragraph out of 200 — re-embedding all 200 would waste compute.

### Q5: Why FastAPI BackgroundTasks instead of Celery?

**A**: At our scale (single worker, ~2000 documents), BackgroundTasks is the right tool. It runs tasks in a thread pool within the same process — no broker, no separate worker, no serialization overhead. Celery adds Redis as a broker, a separate worker process, task serialization, and retry infrastructure. That's valuable at scale, but overkill here. The interface is identical: fire-and-forget with status polling. If we needed distributed ingestion, swapping in Celery only changes the dispatch mechanism, not the pipeline logic.

### Q6: What's the purpose of the `chunk_overlap` parameter?

**A**: Overlap ensures that information at chunk boundaries isn't lost. Consider a sentence that spans the boundary between chunk 41 and chunk 42: "The company's total revenue was $383 | billion for fiscal year 2023." Without overlap, neither chunk has the complete fact. With 64-character overlap, the last 64 chars of chunk 41 appear at the start of chunk 42, so the complete sentence exists in at least one chunk. The trade-off: more overlap = more redundant data = more storage and slightly slower search.

### Q7: Why use Pydantic models instead of plain dictionaries?

**A**: Three reasons. First, **validation**: if a chunk is missing its `doc_id`, I get a clear error at creation time, not a `KeyError` deep in the retrieval pipeline. Second, **documentation**: FastAPI auto-generates OpenAPI docs from Pydantic schemas — every field has a description, type, and constraints. Third, **type safety**: my IDE can autocomplete `chunk.metadata.ticker` instead of `chunk["metadata"]["ticker"]`, catching typos before runtime.

### Q8: How would you scale ingestion to 100,000 documents?

**A**: Three changes. (1) Replace BackgroundTasks with Celery + Redis broker for distributed workers. (2) Batch embedding calls — instead of embedding one chunk at a time, batch 64 chunks per GPU call (sentence-transformers supports this). (3) Use Qdrant's batch upsert API with async uploads. The chunking and dedup logic doesn't change — it's already O(n) in document size. The bottleneck is embedding, which is GPU-bound and parallelizable.

### Q9: What's the trade-off between chunk_size=256 vs chunk_size=1024?

**A**: Smaller chunks (256) are more precise — they match narrow queries like "What was iPhone revenue?" well. But they lose context — the chunk might not mention "Apple" or "fiscal 2023." Larger chunks (1024) have more context but dilute the signal — a chunk about iPhone revenue that also discusses iPad and Mac will match queries about all three, reducing precision. I chose 512 as a middle ground, and Day 4's benchmark will test whether 256 or 1024 performs better on our golden set.

### Q10: Why health checks in Docker Compose?

**A**: Without health checks, `depends_on` only waits for the *container* to start, not for the *service* to be ready. Qdrant might take 3 seconds to initialize its index after the container starts. If the API starts and immediately tries to connect, it gets `ConnectionRefusedError`. Health checks (`curl -f http://localhost:6333/readyz`) verify the service is actually accepting connections. `depends_on: { condition: service_healthy }` blocks the API container until the check passes.

---

## 6. "Explain It Back" Self-Checks

1. **Draw the ingestion pipeline** on paper: which module calls which, in what order? What does each module return?

2. **Trace a re-ingestion**: if you ingest AAPL 10-K, then ingest the same document again, what happens at each step? How many chunks are embedded the second time?

3. **Explain recursive splitting** to a non-technical friend: why can't you just cut every 512 characters?

4. **Without looking at the code**, write the `compute_content_hash` function. What library do you use? What encoding?

5. **Explain the difference** between the `IngestionRequest` and `IngestionResponse` schemas. Why are they separate models instead of one model with optional fields?

---

## 7. Common Failure Modes and How to Debug

| Failure | Symptom | Debug Steps |
|---|---|---|
| SEC EDGAR rate limit | HTTP 403 or 429 from EDGAR | Check `User-Agent` header. Increase sleep between requests. Check if IP is temporarily blocked. |
| Docker services not ready | `ConnectionRefusedError` on startup | `docker compose ps` to check health. `docker compose logs qdrant` for errors. |
| Empty chunks after splitting | No results from search | Log chunk count in pipeline. Check if input text was empty (loader issue). |
| Pydantic validation error on ingest | 422 response from API | Read the error detail — it tells you exactly which field failed and why. |
| HuggingFace dataset field mismatch | `KeyError` in loader | Print `ds.column_names` to see actual field names. May differ from documentation. |
| Job stays "pending" forever | Job never transitions to "processing" | BackgroundTask may have thrown before updating status. Check API logs. |

---

## 8. Metrics Measured Today

No retrieval or quality metrics today — those start at Day 4.

**What we verified**:
- All tests pass (`pytest tests/ -v`)
- Docker Compose starts all 3 services
- Health check returns 200
- API accepts and tracks ingestion jobs

---

## 9. Recommended Reading for Day 2

1. **sentence-transformers documentation** — understand how `all-MiniLM-L6-v2` encodes text into 384-dimensional vectors. Focus on `model.encode()` batching.
2. **Qdrant documentation: Collections** — learn about `VectorParams`, `Distance` metrics (Cosine vs Dot), and `PointStruct` for upsert.
3. **FastAPI Background Tasks** — read the official docs on how tasks run in the thread pool and their limitations.
4. **Understanding Cosine Similarity** — why it's preferred over Euclidean distance for text embeddings (normalization-invariant, angle-based).
5. **Redis data structures** — `HSET`/`HGET` for job status storage (Day 2 swaps in-memory → Redis).
