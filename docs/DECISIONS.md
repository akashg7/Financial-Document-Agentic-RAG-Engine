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

---

## Decision 7: sentence-transformers (`all-MiniLM-L6-v2`) Local Embeddings

**Decision**: Run `sentence-transformers/all-MiniLM-L6-v2` locally inside the application runtime rather than calling OpenAI `text-embedding-3-small` or using `fastembed`.

**Why**:
- Zero cost and privacy compliant: SEC filings and earnings data are processed without network egress to commercial model providers, satisfying the zero-cost constraint.
- Standard and ubiquitous: `sentence-transformers` is universally recognized in industry ML/NLP teams.
- Compact & performant: 384 dimensions balances semantic resolution with low memory and fast similarity search (~80MB model size, easily runs on CPU in sub-15ms for queries).
- Normalized embeddings out-of-the-box: dot product equals cosine similarity.

**Alternatives**:
- `text-embedding-3-small` (OpenAI): Higher dimensionality (1536) and slightly better benchmark retrieval, but violates zero-cost constraint and introduces external network latency, rate limits, and egress privacy issues.
- `fastembed` (Qdrant ONNX runtime): Lighter runtime dependencies, but less transparent model customizability and fewer community examples than `sentence-transformers`.
- `BAAI/bge-small-en-v1.5`: Marginally higher MTEB score, but `all-MiniLM-L6-v2` is more standard for cross-platform CPU inference.

**When I'd change it**: If Day 4 retrieval evaluation demonstrates specific domain failures on financial jargon, I would evaluate fine-tuned financial embedding models (e.g., `FinBERT` embeddings or `bge-base-financial`).

---

## Decision 8: Cosine Distance for Dense Vector Similarity

**Decision**: Configure Qdrant collection with `Distance.COSINE` for vector indexing and distance measurement.

**Why**:
- Normalized vector geometry: Cosine distance evaluates directional angle rather than vector magnitude, eliminating bias toward longer text chunks that might have larger magnitude before normalization.
- Consistent thresholding: Similarity scores bounded in `[-1.0, 1.0]` (or `[0.0, 1.0]` for normalized text embeddings), making relevance cutoff tuning straightforward for agentic gating.

**Alternatives**:
- `Distance.DOT`: Equivalent to cosine when vectors are unit-normalized ($L_2 = 1.0$), and computationally faster. However, Cosine distance in Qdrant provides guardrails if an unnormalized vector is ever passed.
- `Distance.EUCLID`: Sensitive to absolute length differences; poor choice for variable-length semantic text chunks.

**When I'd change it**: If we switch to an embedding model specifically trained for dot-product or maximum inner product search (MIPS) where magnitude conveys document authority or prior probability.

---

## Decision 9: In-Memory Qdrant Client Fallback for Integration Testing

**Decision**: Provide `location=":memory:"` support in `get_qdrant_client()` during test fixtures, enabling realistic vector search testing without requiring Docker or network services.

**Why**:
- High developer velocity: Tests run in <10 seconds locally and in lightweight CI pipelines without starting Docker containers.
- True API fidelity: Qdrant's embedded in-memory engine executes real HNSW indexing and payload filtering code, unlike mocking where mocked queries do not validate Qdrant filter schemas.
- Idempotent and isolated: Each test run gets a pristine database instance.

**Alternatives**:
- Testcontainers: Spawns real Docker container during pytest. Reliable, but adds 15-30s overhead per test session and requires Docker daemon availability.
- Pure `unittest.mock.MagicMock`: Doesn't test actual vector math, collection creation, or Qdrant query syntax errors (which caught a `FieldCondition` argument mismatch).

**When I'd change it**: If testing distributed cluster clustering, snapshot restorations, or specific storage engine disk I/O.

---

## Decision 10: Redis Hashes with TTL for Ingestion Job Tracking & Graceful Fallback

**Decision**: Migrate ingestion job state from in-memory dictionary to Redis Hashes with 24-hour TTL (`settings.redis_job_ttl_seconds`), with automatic fallback to an in-memory dictionary if Redis is offline.

**Why**:
- Process resilience: If the FastAPI container restarts or crashes during a long ingestion batch, job metadata and status history are preserved in Redis.
- Horizontal scaling: Multiple API replicas can query and update the same job IDs.
- Automatic cleanup: TTL guarantees Redis memory does not grow unbounded from old completed or failed ingestion jobs.
- Graceful degradation: Unit tests and standalone local runs work seamlessly without requiring a running Redis instance.

**Alternatives**:
- PostgreSQL / SQLite task table: Good durability, but heavier schema migrations and slower for high-frequency progress polling.
- Celery / Flower: Heavier orchestration framework with extra worker containers; overkill for single-worker background tasks.

**When I'd change it**: If we move to distributed workers requiring persistent job logs and complex DAG dependencies across documents.

---

## Decision 11: BM25 as Qdrant Named Sparse Vectors vs. External Search Engine (Elasticsearch / OpenSearch)

**Decision**: Store BM25 term weights directly as named sparse vectors (`sparse`) within the same Qdrant collection alongside dense vectors (`dense`), rather than deploying an external search engine.

**Why**:
- Operational simplicity and zero cost: Running Elasticsearch / OpenSearch requires 2-4GB+ RAM just for JVM overhead, violating lightweight local dev and zero-cost constraints.
- Unified storage & atomic operations: Every chunk's dense embedding, sparse weights, and metadata payload are written, updated, and deleted in a single Qdrant transaction.
- Single-stage filtering: Qdrant applies payload filters (`ticker`, `doc_type`) uniformly across both dense and sparse vector queries.

**Alternatives**:
- Elasticsearch / OpenSearch: Traditional industry choice for BM25. Better out-of-the-box tokenizers and stemming analyzers, but introduces dual-store synchronization bugs, duplicate data storage, and heavy resource footprint.
- SQLite FTS5 / PostgreSQL tsvector: Built-in full-text search, but cannot be easily queried in parallel with vector ANN search in the same database engine.

**When I'd change it**: If the document corpus scales past 10 million documents where inverted index sharding and distributed keyword queries require specialized text cluster infra.

---

## Decision 12: Reciprocal Rank Fusion (RRF) vs. Weighted Linear Score Combination

**Decision**: Fuse ranked candidate lists from dense and sparse retrieval using Reciprocal Rank Fusion (RRF) with $k=60$, rather than linear weighted score combination ($\alpha \cdot \text{Score}_{\text{dense}} + (1-\alpha) \cdot \text{Score}_{\text{sparse}}$).

**Why**:
- Scale invariance: Dense cosine similarities live strictly in $[-1.0, 1.0]$ (or $[0.3, 0.9]$ in practice), while BM25 scores are unbounded positive numbers ($[0.0, 30.0+]$ depending on term rarity and document length). Combining raw scores requires min-max or z-score normalization that shifts wildly across different query types.
- Robustness: RRF operates exclusively on rank positions ($1 / (k + r)$), making it impervious to score distribution anomalies.
- Parameter stability: The smoothing constant $k=60$ has been validated across extensive academic and industrial IR benchmarks (Cormack et al., 2009) as a reliable default without requiring per-dataset hyperparameter grid searches.

**Alternatives**:
- Linear weighted combination with Min-Max normalization: Can work well if weights are tuned carefully, but is extremely fragile when one query has an outlier BM25 score.
- Relative Score Fusion (RSF): Normalizes scores relative to the top result in each list before blending; still sensitive to score distribution shapes.
- Learned cross-attention reranker only: Extremely high retrieval quality, but running cross-encoders across top-50 from each engine directly is computationally expensive. Combining top candidates with RRF first yields a higher-quality candidate pool for downstream reranking (Day 4).

**When I'd change it**: If golden set evaluation (Day 4) demonstrates that one modality consistently dominates certain question classes, prompting a learned classifier or dynamic gating.

---

## Decision 13: Hash-based Term Indexing for Stateless Sparse Vectors vs. Global Vocabulary Dictionary

**Decision**: Map vocabulary terms to sparse vector dimension indices using 32-bit deterministic hashing (`int.from_bytes(md5(term)[:4])`) instead of maintaining a centralized, sequential vocabulary dictionary.

**Why**:
- Stateless and distributed: Any worker or API thread can compute the sparse vector index for any query or document term independently without coordinating with a central vocabulary registry or database lookup.
- Low collision probability: A 32-bit integer space spans over 4.29 billion index slots. Across a domain vocabulary of ~50,000 unique financial terms, birthday paradox collision rate is $< 0.03\%$, which has negligible impact on BM25 retrieval ranking.

**Alternatives**:
- Sequential dictionary (`vocab[term] = current_id++`): Guarantees zero collisions, but requires persisting and synchronizing vocabulary tables across all workers and restart cycles.
- Tokenizer vocabularies (e.g. BERT/WordPiece vocab IDs): Fixed vocabulary size, but does not capture domain-specific financial tickers, section codes ("10-K/A"), or rare numbers.

**When I'd change it**: If collision rates in massive corpora (>10M unique terms) degrade precision, in which case we would migrate to 64-bit sparse vector indices if supported by the vector database.


---

## Decision 14: Cross-Encoder Model Selection (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

**Decision**: Use `cross-encoder/ms-marco-MiniLM-L-6-v2` as our local CPU reranker rather than proprietary API rerankers (Cohere Rerank, Voyage Rerank) or heavier cross-encoder models (e.g., `cross-encoder/ms-marco-electra-base`, BGE-Reranker-Large).

**Why**:
- Strict zero-cost constraint: No API tokens, keys, or rate limits. Fully self-hosted and offline-capable.
- Memory and latency budget: Model size is ~80MB, easily residing in RAM alongside the dense embedding model (`all-MiniLM-L6-v2`).
- Inference efficiency on CPU: Evaluates a 50-pair candidate batch in ~30–60ms on modern CPU without requiring a dedicated CUDA GPU.
- High empirical quality: Fine-tuned on the 500k+ query MS MARCO passage ranking dataset; captures deep token-to-token cross-attention between query and candidate passage.

**Alternatives**:
- Cohere Rerank API: Industry gold standard for reranking quality, but incurs monetary cost ($1/1,000 searches) and network egress latency (150–300ms network round-trip).
- `BAAI/bge-reranker-large`: Exceptional performance on MTEB benchmarks, but ~1.3GB model size causes substantial CPU latency (>1.5s per batch of 50) without a GPU.
- ColBERTv2 (Late Interaction): Fast token-level max-sim scoring, but requires specialized index storage (substantially larger than standard vector stores) and complex deployment.

**When I'd change it**: If moving to a production GPU cluster with dedicated inference endpoints (e.g. Triton/vLLM/TEI) where `bge-reranker-large` or `mixedbread-ai/mxbai-rerank-large` can run in under 20ms.

---

## Decision 15: Two-Stage Retrieval with Top-50 Candidate Pool for Reranking

**Decision**: First-stage retrieval retrieves up to 50 candidates via hybrid search (dense + BM25 with RRF), and feeds this candidate pool into the cross-encoder to produce the final top-$K$ (e.g., top 5 or 10).

**Why**:
- Algorithmic complexity decoupling: Bi-encoder vector search is $O(1)$ amortized with HNSW indexing; cross-encoder inference is $O(N \cdot L^2)$ where $N$ is candidate count and $L$ is sequence length.
- Diminishing returns beyond 50: In financial RAG, high-recall first-stage retrieval (dense + sparse) almost always places relevant documents within the top 50 (Recall@50 typically $> 95\%$). Expanding to top 100 or 200 doubles inference latency with negligible recall upside.
- Context window economics: Downstream LLMs typically ingest 3–7 chunks. Reranking 50 candidates down to top 5 strikes the optimal Pareto balance between candidate coverage and execution latency.

**Alternatives**:
- Rerank top 10 only: Faster (~10ms CPU inference), but risks missing relevant passages that ranked between 11 and 50 during first-stage hybrid retrieval.
- Rerank top 200: Maximizes recall ceiling, but inflates CPU latency to $> 1$ second, exceeding acceptable latency budgets for interactive queries.

**When I'd change it**: If golden set evaluation reveals that critical evidence frequently falls between ranks 51–100, or if hardware acceleration permits evaluating 100+ candidates in under 30ms.

---

## Decision 16: Golden Evaluation Set Design with Explicit Hard Unanswerable Questions

**Decision**: Construct a 119-question curated golden evaluation set across 6 categories (`financial_metric`, `risk_factor`, `strategy`, `comparison`, `temporal`, `unanswerable`) spanning multiple SEC filings and tickers, with ~17% (20 questions) deliberately designed as unanswerable.

**Why**:
- Guarding against hallucinations: In financial domains, hallucinating a metric or forward-looking guidance is catastrophic. A rigorous evaluation framework must test the system's ability to abstain when evidence is absent, rather than only testing its ability to retrieve when evidence exists.
- Categorical diagnostic power: Grouping queries into categories enables isolating retrieval failures (e.g., does sparse fail on temporal queries? does dense fail on exact metric tables?).
- Reproducibility and honesty: Having an offline golden dataset with ground-truth answer expectations and explicit difficulty tags enables automated, regression-proof benchmarking without fabricating numbers.

**Alternatives**:
- Synthetic automated QA generation (e.g., GPT-4 generating QA pairs from raw text): High volume, but tends to produce surface-level questions with obvious lexical overlap that artificially inflate retrieval scores.
- RAGAS-only evaluation: Powerful for end-to-end generation metrics, but does not provide controlled ground-truth chunk rankings for retrieval-stage component benchmarking.

**When I'd change it**: When expanding the corpus to hundreds of company filings on Day 10, where we augment the human-curated golden set with semi-automated domain-expert verified questions.
