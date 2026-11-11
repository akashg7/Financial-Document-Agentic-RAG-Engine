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
---

## Decision 17: Multi-Provider LLM Architecture (Groq Primary with Gemini Fallback)

**Decision**: Route primary generation through Groq (`llama-3.1-8b-instant`) with automatic failover to Google Gemini (`gemini-1.5-flash`), backed by a deterministic offline mock generator.

**Why**:
- Zero-cost SLA resilience: Both Groq and Gemini provide generous free tiers, but have stringent rate limits (Groq: 30 RPM; Gemini: 15 RPM). Coupling them in an active-standby failover pattern allows the engine to absorb traffic bursts without paying for dedicated inference or dropping user requests.
- Latency optimization: Groq's LPU architecture delivers ~500–700 tokens/second on `llama-3.1-8b-instant`, producing complete grounded financial answers in under 400ms.
- High-availability redundancy: Running across two distinct cloud providers (Groq and Google Cloud) eliminates dependency on any single provider's availability.

**Alternatives**:
- Single provider with retries only: Simpler, but a rate-limit exhaustion (429) or cloud outage halts the entire service until quotas reset.
- OpenAI GPT-4o / GPT-4o-mini: High quality, but violates our zero-cost constraint.
- Local Ollama / vLLM: Zero API cost, but running 8B+ models locally requires 16GB+ unified RAM and competes for CPU/GPU resources with our embedding and reranking models.

**When I'd change it**: In an enterprise production setting with dedicated API budgets, where we would use paid Groq/Gemini tiers or self-hosted vLLM inference on a Kubernetes GPU cluster.

---

## Decision 18: Grounded Citation Binding & Two-Tier Quote Verification

**Decision**: Force the LLM to output structured JSON with explicit citations referencing `chunk_id` and verbatim `quote`, verified via a deterministic two-tier validator:
1. Tier 1: Target `chunk_id` must exist in the retrieved context pool passed into the prompt.
2. Tier 2: The cited `quote` must match text within that chunk (verbatim or high normalized lexical overlap).

**Why**:
- Hallucination prevention: Standard LLMs frequently fabricate citations, attribute real facts to incorrect filings, or make up quotes.
- Auditable provenance: In financial compliance and investment analysis, every metric and forward-looking disclosure must trace back to the authoritative SEC EDGAR filing or earnings call transcript.
- Confidence calibration: Unverified citations directly downgrade the response confidence score and trigger filtering, preventing false evidence from polluting client applications.

**Alternatives**:
- Unstructured inline markdown citations (e.g. `[1]`): Fragile to parse and prone to hallucinations; difficult to verify quotes programmatically.
- Post-hoc LLM judge for grounding: Slow and expensive; doubles token consumption by calling a second LLM to evaluate the first LLM. Deterministic string/lexical validation is $O(1)$, instantaneous, and free.

**When I'd change it**: If citations frequently paraphrase long conceptual arguments rather than quoting short numerical clauses, prompting an NLI (Natural Language Inference) entailment model.

---

## Decision 19: Error Discrimination in Resilience Policies (Fast-Fail vs. Exponential Backoff)

**Decision**: Explicitly discriminate between transient errors (`RateLimitError` 429, 5xx server hiccups) which are retried with exponential backoff (1s, 2s, 4s), and authentication/configuration errors (`AuthenticationError` 401/403, invalid key) which fail fast without retries, triggering immediate failover.

**Why**:
- Latency preservation: Retrying an invalid API key 3 times with exponential backoff wastes ~7 seconds waiting for attempts that are guaranteed to fail.
- Resilience: Transient rate limits on free tiers typically resolve within 1–3 seconds, making backoff with jitter highly effective for 429s.

**Alternatives**:
- Generic retry on all exceptions: Catches everything, but causes unacceptable latency spikes when credentials or models are misconfigured.
- No retries (failover on first error): Faster on permanent errors, but prematurely exhausts provider failover on single-packet network drops or 1-second burst rate limits.

**When I'd change it**: When moving to an asynchronous task queue architecture (Celery/Temporal) where failed jobs can be queued for delayed replay.
---

## Decision 20: Multi-Signal Confidence Fusion vs. Single-Metric Score

**Decision**: Calculate confidence by fusing three orthogonal signals:
1. Grounding Fidelity (45% weight: citation precision and verbatim quote verification)
2. Retrieval Evidence Quality (40% weight: normalized cross-encoder logit and margin between rank #1 and #2)
3. Linguistic Certainty (15% weight: absence of epistemic hedging phrases)

**Why**:
- Brittleness of single-vector similarity: A passage can exhibit high cosine similarity simply by containing repeated financial jargon ("EBITDA", "amortization"), even if it fails to answer the user's specific question.
- Complementary failure modes: The cross-encoder measures semantic relevance, the citation validator enforces factual alignment, and the linguistic parser detects uncertainty emitted by the generator. Fusing all three prevents single-point-of-failure hallucinations.
- Hard insufficiency override: If evidence is missing or the response signals `INSUFFICIENT_EVIDENCE`, the composite score is overridden to 0.0, bypassing weighted arithmetic.

**Alternatives**:
- Retrieval score only: Prone to false positives on keyword-heavy irrelevant passages.
- LLM self-reported confidence ("Rate your confidence 1-10"): Well-documented calibration failure in LLMs; models are notoriously overconfident even when fabricating claims.
- Perplexity / token log probabilities: Sensitive to vocabulary length and prompt phrasing; unavailable on many hosted inference endpoints (e.g. standard Gemini API).

**When I'd change it**: If we fine-tune a dedicated Natural Language Inference (NLI) classifier on financial entailment datasets, replacing heuristic signal weights with learned logistic regression coefficients.

---

## Decision 21: Sigmoid Normalization of Cross-Encoder Logits with Score Margin Bonus

**Decision**: Map unbounded cross-encoder logits ($x \in (-\infty, \infty)$) into $[0, 1]$ using standard logistic sigmoid ($\sigma(x) = \frac{1}{1 + e^{-x}}$) and add a margin bonus ($\min(0.3, \Delta_{\text{top2}} \times 0.1)$) reflecting the score gap between candidate #1 and candidate #2.

**Why**:
- Logit calibration: Cross-encoder outputs (`ms-marco-MiniLM-L-6-v2`) are unconstrained logits where $< -2$ indicates strong irrelevance, $0$ indicates neutrality, and $> 2$ indicates high relevance. Sigmoid maps this smoothly into $(0, 1)$.
- Decisive winner vs. ambiguity: If chunk #1 scores 3.5 and chunk #2 scores 0.5 ($\Delta = 3.0$), the retrieval pool has a single authoritative winner. If chunk #1 scores 1.0 and chunk #2 scores 0.99 ($\Delta = 0.01$), the retrieval stage is ambiguous, warranting a lower retrieval signal to promote query rewriting (Day 7).

**Alternatives**:
- Min-Max scaling over the candidate batch: Highly sensitive to outliers; if all 50 chunks are irrelevant (scores between -8 and -7), min-max artificially stretches the least irrelevant chunk to 1.0.
- Softmax across candidates: Distributes probability across all candidates, penalizing cases where multiple chunks are equally relevant.

**When I'd change it**: If moving to a reranker model that outputs calibrated probabilities directly rather than raw margin logits.

---

## Decision 22: Tri-State Agentic Routing (`generate`, `rewrite`, `abstain`) and Calibrated Thresholds

**Decision**: Calibrate decision boundaries at $\tau_{\text{generate}} = 0.70$ and $\tau_{\text{abstain}} = 0.40$:
- $\text{Confidence} \ge 0.70 \implies \mathbf{"generate"}$
- $0.40 \le \text{Confidence} < 0.70 \implies \mathbf{"rewrite"}$
- $\text{Confidence} < 0.40 \implies \mathbf{"abstain"}$

**Why**:
- Empirical golden set validation: Evaluated across 119 queries in `eval/golden_set.jsonl`:
  - 100% of unanswerable queries correctly triggered `abstain` (zero hallucination leakage).
  - 86.9% of answerable queries generated immediately with verified citations.
  - 10.9% of ambiguous/hard queries were routed to query rewriting (`rewrite`).
- Agentic control flow: Rather than treating RAG as a binary pass/fail, the intermediate `rewrite` band gives the agent (Day 7 LangGraph) the opportunity to refine query keywords, expand synonyms, or adjust filters before giving up or hallucinating.

**Alternatives**:
- Binary generate vs abstain (no rewrite): Forces marginal queries to either hallucinate an answer or prematurely surrender.
- Higher generate threshold (e.g. 0.85): Excessively conservative; abstains on valid complex queries that require slight reasoning.

**When I'd change it**: If user feedback in production indicates that users prefer interactive follow-up clarification over automated query rewriting.

---

## Decision 23: LangGraph `StateGraph` Cyclical State Machine vs. Linear DAG Chains

**Decision**: Implement the agentic RAG loop using LangGraph's `StateGraph` with explicit cyclical edges (`rewrite -> retrieve`) and conditional routing (`route_decision`), rather than linear pipeline abstractions (LangChain LCEL, LlamaIndex query engines).

**Why**:
- Cyclical execution: High-stakes financial RAG requires iterative query reformulation when initial retrieval confidence is ambiguous. Linear DAGs cannot loop back without recursive hacks or external orchestration.
- Explicit state progression: LangGraph exposes the full execution state machine (`retrieve -> generate -> route -> [rewrite | abstain | END]`), making debugging, visualization, and observability straightforward.
- Foundation for checkpointing & human-in-the-loop: StateGraph natively supports checkpointing and persistence for long-running workflows or future agent extensions.

**Alternatives**:
- Linear LCEL (LangChain Expression Language): Clean for single-pass RAG, but fundamentally acyclic; cannot support iterative retrieval retry loops.
- AutoGen / CrewAI multi-agent conversational patterns: Unnecessary conversational overhead and nondeterministic turn-taking for a task that is fundamentally a state-machine control loop.
- Plain Python while-loops: Simple to write initially, but lacks checkpointing, graph visualization, interruptibility, and standardized telemetry integration.

**When I'd change it**: If the workflow grows into a multi-agent debate architecture where multiple autonomous persona agents critique each other asynchronously.

---

## Decision 24: TypedDict State Definition with Partial Node Merging vs. Pydantic Models in StateGraph

**Decision**: Define `AgentState` as a `typing_extensions.TypedDict(total=False)` rather than a strict `pydantic.BaseModel` inside the LangGraph state machine.

**Why**:
- Partial state return: In LangGraph, nodes can return lightweight dictionaries containing only the keys they mutate (e.g., `{"current_query": new_q, "retry_count": count}`). LangGraph automatically merges these partial updates into the global state without requiring full model re-instantiation.
- Execution efficiency: Avoids Pydantic model validation and serialization overhead on every graph edge transition.
- Schema flexibility: Easy to thread optional metadata, tracing tokens, and telemetry fields across nodes without strict schema migration constraints.

**Alternatives**:
- Pydantic BaseModel state: Offers runtime type validation, but requires every node to return a complete model instance or construct complex validation wrappers.
- Global shared mutable dictionary: Mutable shared state can lead to concurrency race conditions and unpredictable edge effects during branching.

**When I'd change it**: If external untrusted plugins or dynamic third-party nodes are injected into the graph, requiring strict runtime validation of state transitions.

---

## Decision 25: Hybrid LLM + Deterministic Financial Query Reformulation with Progressive Deepening

**Decision**: Implement query rewriting with an intelligent deterministic fallback that removes conversational filler, injects ticker symbols, expands domain-specific GAAP terminology ("sales" -> "net sales revenue consolidated operations"), and applies progressive deepening (adding Item 7/8 headers on retry 1, financial statements on retry 2) if LLM rewriting fails or runs offline.

**Why**:
- Zero-cost & offline resilience: If external LLM API rate limits (Groq 429) occur during rewriting, the system must not halt or fail. Deterministic expansion guarantees the agentic retry loop operates reliably even without external API connectivity.
- Domain alignment: SEC EDGAR filings strictly adhere to standardized GAAP headers and item classifications. Expanding colloquial questions into regulatory vocabulary dramatically improves both BM25 term frequency match and dense semantic alignment.
- Preventing query oscillation: Progressive deepening alters the query structure on every retry iteration, ensuring the agent does not repeat identical search queries across cycles.

**Alternatives**:
- Pure LLM query rewriting without deterministic fallback: Dependent on external API availability; fails completely when offline or rate-limited.
- Rule-based synonym replacement without progressive deepening: Can get stuck in semantic loops if the base synonyms still fail to hit relevant chunks.
- Hypothetical Document Embeddings (HyDE): Generates an entire hypothetical 10-K excerpt before retrieval; high latency (~1-2s additional LLM call) and can hallucinate fictitious financial numbers that mislead retrieval.

**When I'd change it**: If a dedicated sequence-to-sequence model fine-tuned specifically on EDGAR query reformulation is deployed locally with sub-20ms inference latency.

---

## Decision 26: Two-Tier Cache Architecture (L1 Exact Hash + L2 Scoped Dense Cosine Similarity)

**Decision**: Implement caching in two hierarchical tiers: L1 exact string lookup via SHA-256 hash ($< 1\text{ ms}$), falling back to L2 semantic similarity via dense vector cosine comparison ($< 15\text{ ms}$).

**Why**:
- Extreme latency optimization: If an identical query is asked repeatedly by clients, L1 resolves it in microseconds without vector computation or model inference.
- Natural language flexibility: Paraphrased queries ("Apple FY2023 revenue" vs "What was Apple's total net sales in fiscal 2023?") hit L2 semantic cache, saving ~1–3 seconds of pipeline execution and costly LLM calls.
- Measured impact: Achieved **$866.6\times$ speedup on L1** and **$248.5\times$ speedup on L2** relative to the cold agentic RAG pipeline.

**Alternatives**:
- Exact matching only: Fails on trivial casing, whitespace, or paraphrasing differences, yielding poor cache hit rates (< 15%).
- Semantic matching only: Unnecessarily runs dense embedding models even when the query string is identical, wasting CPU cycles on common repetitive requests.

**When I'd change it**: If query traffic reaches >10,000 requests/sec, in which case we would front L1 with an in-process LRU cache (e.g. `cachetools`) to avoid Redis network roundtrips altogether.

---

## Decision 27: Strict Calibrated Cosine Similarity Threshold ($\tau = 0.92$) with Scope Partitioning

**Decision**: Enforce a strict semantic similarity cutoff ($\tau \ge 0.92$) and partition the candidate vector search space by scope (`TICKER:DOC_TYPE`).

**Why**:
- Financial precision: In general consumer chatbots, $\tau = 0.80$ is acceptable. In financial intelligence, a lower threshold causes catastrophic false matches between distinct accounting metrics (e.g., matching "Apple revenue" with "Apple net income").
- Scope isolation: Queries about different companies (e.g., "What was total revenue?" for `AAPL` vs `MSFT`) produce identical text embeddings. Scope partitioning (`AAPL:10-K` vs `MSFT:10-K`) prevents cross-ticker cache contamination.

**Alternatives**:
- Global unpartitioned semantic search: Cross-pollinates answers between different companies.
- Lower threshold ($\tau = 0.85$): Frequently confuses related but distinct metrics (operating cash flow vs net income).

**When I'd change it**: If moving to a financial bi-encoder trained specifically on semantic textual similarity of accounting queries with fine-grained metric discriminators.

---

## Decision 28: Bipartite Lineage DAG for Surgical Invalidation vs. Global Cache Flushing

**Decision**: Track forward and reverse dependency edges ($\text{Document} \to \text{Chunk} \to \text{Cached Answers}$) via a dedicated `LineageManager` that invalidates only the specific queries affected by amended or re-ingested filings.

**Why**:
- Zero-stale-read compliance: Under SEC regulations, serving restated or outdated financial figures after an 8-K or 10-K/A amendment is a critical defect. Lineage ensures instant eviction of affected queries.
- Surgical precision: Re-ingesting a new filing for Apple does not evict cached research on Microsoft or unrelated Apple sections, maintaining a high cache hit rate and avoiding unnecessary re-computation.
- Blast radius observability: Enables pre-invalidation impact analysis (`get_affected_queries`), providing operational visibility into which queries will be impacted before an amendment is deployed.

**Alternatives**:
- Passive TTL expiration: Stale answers remain valid until TTL passes (up to 24 hours), causing compliance failures.
- Global cache flush (`FLUSHDB`): Destroys all cached answers on every ingestion job, causing severe cache thrashing and latency spikes.

**When I'd change it**: If cache scale exceeds tens of millions of entries, where maintaining bidirectional set indices in Redis is replaced with an event-driven pub/sub invalidation bus (e.g. Kafka / Debezium).

---

## Decision 29: Server-Sent Events (SSE) over WebSockets for Agent Streaming

**Decision**: Implement streaming over HTTP Server-Sent Events (`text/event-stream`) with typed status events, partial text chunks, and a final structured JSON event, rather than bidirectional WebSockets.

**Why**:
- Unidirectional fit: The agentic loop is purely server-to-client streaming during query execution. The client initiates the request and receives incremental execution progress followed by the response; no client-to-server messaging is needed mid-stream.
- Protocol simplicity: SSE works over standard HTTP/1.1 and HTTP/2, requires no custom framing or ping/pong heartbeats, traverses corporate firewalls and edge proxies without WebSocket upgrade negotiation, and natively reconnects on transient disconnects.
- Rich typed telemetry: Emitting custom SSE event types (`status`, `chunk`, `done`, `error`) enables client UIs to render animated pipeline phase transitions ("Retrieving documents...", "Evaluating confidence...") before text generation even begins.

**Alternatives**:
- WebSockets: Adds bidirectional overhead, stateful connection handling, and complex load-balancer connection pooling for a purely unidirectional flow.
- Polling: Introduces artificial latency and high request overhead to repeatedly inspect job status.
- Long Polling: Clunky connection timeouts and requires holding server connections open across retries without incremental updates.

**When I'd change it**: If adding interactive human-in-the-loop multi-turn conversations where the user interrupts or guides the agent mid-generation.

---

## Decision 30: Application-Level Endpoint Rate Limiting (SlowAPI) over External API Gateways

**Decision**: Implement IP-based rate limiting directly within the FastAPI application using SlowAPI with per-endpoint thresholds (`/ingest`: 5/min, `/query`: 15/min, `/search`: 30/min, `/health`: exempt).

**Why**:
- Zero infrastructure overhead: Runs within the existing Python/FastAPI environment without requiring an external Kong, Traefik, or Envoy gateway deployment.
- Domain-aware tiering: Different endpoints have wildly different computational and monetary costs. External gateways require complex routing rules to enforce distinct limits on vector search vs. LLM generation vs. ingestion, whereas in-code decorators configure this cleanly in one line per endpoint.
- Unified observability: Throttled requests immediately trigger custom Prometheus counters (`rag_rate_limited_total`) and return uniform JSON payloads (`retry_after_seconds: 60`).

**Alternatives**:
- External API Gateway (Kong / Envoy / Cloudflare): More scalable across distributed clusters, but introduces infrastructure sprawl, external dependencies, and local dev friction.
- Global single limit: Throttles lightweight search requests at the same strict threshold needed to protect costly LLM generation.

**When I'd change it**: If scaling to a multi-instance Kubernetes cluster with hundreds of thousands of concurrent users, where rate-limiting state must be enforced at the ingress layer or shared across an enterprise Redis cluster.

---

## Decision 31: Domain-Specific Prometheus Metrics over Generic HTTP Middleware

**Decision**: Instrument custom Prometheus counters, histograms, and gauges specifically designed for agentic RAG (`rag_queries_total`, `rag_query_latency_seconds`, `rag_confidence_score`, `rag_cache_hits_total`, `rag_retrieval_latency_seconds`, `rag_active_queries`) alongside standard application info.

**Why**:
- Actionable RAG telemetry: Generic HTTP middlewares (like `prometheus-fastapi-instrumentator`) only report HTTP 200/500 counts and route latency. They provide zero visibility into RAG quality: What percentage of queries are abstaining? How does confidence correlate with retrieval latency? What is the L1 vs L2 cache hit distribution?
- SRE/Production readiness: Exposing `/metrics` in the standard Prometheus text format allows plug-and-play scraping by any Grafana/Prometheus infrastructure without custom log parsing.
- Zero network latency: Metrics are updated via atomic in-memory counters (< 1 microsecond per observation) with zero external network overhead.

**Alternatives**:
- Generic HTTP instrumentator: Captures route-level status codes but cannot capture domain state (citations, confidence, retry counts, cache tiers).
- Logging-only observability: Writing metrics to stdout requires log aggregation pipelines (ELK / Loki) and expensive query-time histogram aggregations.

**When I'd change it**: If transitioning to OpenTelemetry (OTel) traces and metrics collectors across a polyglot microservice mesh.



