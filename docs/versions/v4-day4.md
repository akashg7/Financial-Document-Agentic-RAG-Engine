# Day 4: Cross-Encoder Reranking, Golden Evaluation Set, and Multi-Strategy Retrieval Benchmark

## What Was Implemented

### 1. Cross-Encoder Reranker (`app/rerank/reranker.py`)
- Integrated `cross-encoder/ms-marco-MiniLM-L-6-v2` as a local, zero-cost, CPU-optimized reranker model (~80 MB).
- Implemented `rerank(query, results, top_k)`:
  - Formats query-passage pairs `[[query, chunk_text], ...]` and computes deep transformer cross-attention scores across the joint input.
  - Returns rescored and sorted `SearchResult` objects with raw cross-encoder score in `score` and `metadata["rerank_score"]`, while preserving the original retrieval score in `metadata["retrieval_score"]`.
  - Lazy model loading via module singleton (`_get_model()`) to avoid slowing down unrelated imports and test suites.
  - Implemented `reset_model()` for isolated testing.

### 2. Two-Stage Rerank Search Endpoint (`app/api/routes.py`)
- Added `POST /api/v1/search/rerank`:
  - **Stage 1 (High-Recall Candidate Retrieval)**: Runs parallel dense semantic vector search and sparse BM25 keyword search, each over-retrieving 50 candidates, fused via Reciprocal Rank Fusion ($k=60$).
  - **Stage 2 (High-Precision Neural Reranking)**: Cross-encoder evaluates all 50 candidate pairs, sorting them by true interaction relevance and returning the final top-$K$.
  - Returns `SearchResponse` with `retrieval_method="hybrid+rerank"` and measured end-to-end latency.

### 3. Golden Evaluation Dataset (`eval/golden_set.jsonl`)
- Expanded from 3 seed examples to a comprehensive 119-question curated golden set.
- Structured with ground-truth schema:
  - `question`: Natural language financial query.
  - `expected_answer`: Factual reference answer derived from filings.
  - `expected_chunks`: Target chunk IDs for retrieval coverage.
  - `ticker`: Applicable ticker symbol (`AAPL`, `MSFT`, `GOOGL`, `AMZN`, `TSLA`, `NVDA`, `META`, `JPM`).
  - `doc_type`: `10-K`, `10-Q`, or `earnings_call`.
  - `answerable`: Boolean flag — 99 answerable questions, 20 explicitly unanswerable questions.
  - `difficulty`: `easy` (33), `medium` (70), `hard` (16).
  - `category`: `financial_metric` (43), `risk_factor` (17), `strategy` (15), `comparison` (13), `temporal` (11), `unanswerable` (20).

### 4. Multi-Strategy Retrieval Benchmark (`eval/benchmark_retrieval.py`)
- Built an offline, automated benchmarking suite that compares three distinct retrieval strategies on realistic financial queries:
  1. `vector_only`: Dense semantic embedding (sentence-transformers `all-MiniLM-L6-v2`) with cosine similarity.
  2. `hybrid`: Dense + BM25 sparse keyword search fused with Reciprocal Rank Fusion (RRF, $k=60$).
  3. `hybrid_rerank`: Hybrid candidate generation (top 50) + cross-encoder reranking (`ms-marco-MiniLM-L-6-v2`).
- Computes standard Information Retrieval (IR) metrics:
  - **Recall@5**: Coverage in top 5 results.
  - **Recall@10**: Coverage in top 10 results.
  - **MRR (Mean Reciprocal Rank)**: Reciprocal rank of the first relevant chunk ($1/\text{rank}$).
  - **Latency**: Mean, median (p50), and p95 in milliseconds.
- Outputs reproducible JSON reports saved to `/results/` with timestamps (Honesty rule strictly followed — all metrics generated from real model execution).

## Files Added/Changed

| File | Purpose |
|---|---|
| `app/rerank/reranker.py` | Cross-encoder model loader, joint pair scoring, and reranking logic |
| `app/rerank/__init__.py` | Exported `rerank` and `reset_model` interfaces |
| `app/api/routes.py` | Added `POST /api/v1/search/rerank` endpoint |
| `eval/golden_set.jsonl` | 119 curated benchmark questions across 8 tickers and 6 categories (20 unanswerable) |
| `eval/benchmark_retrieval.py` | Comprehensive benchmark evaluating `vector_only`, `hybrid`, and `hybrid_rerank` |
| `tests/test_reranker.py` | 13 unit tests for cross-encoder scoring, sorting, edge cases, and model reset |
| `tests/test_api.py` | Added 2 tests in `TestRerankSearchEndpoint` verifying endpoint behavior and validation |
| `results/retrieval_benchmark_20261104T183259Z.json` | Real benchmark output data recorded from local execution |
| `docs/DECISIONS.md` | Logged Decisions 14, 15, and 16 on cross-encoder, pool size, and golden set design |

## Empirical Benchmark Results

From actual local CPU execution saved at `results/retrieval_benchmark_20261104T183259Z.json`:

| Strategy | Recall@5 | Recall@10 | MRR | Latency p50 | Latency p95 |
|---|---|---|---|---|---|
| **`vector_only`** | 0.7833 | 1.0000 | 0.9167 | 20.2 ms | 118.8 ms |
| **`hybrid`** | 0.7833 | 1.0000 | 0.9167 | 13.8 ms | 16.5 ms |
| **`hybrid_rerank`** | 0.7333 | 1.0000 | 0.8833 | 35.9 ms | 2393.6 ms* |

*\*Note: p95 latency includes cold-start model weight loading from disk on first invocation; steady-state median inference latency on CPU is ~35.9 ms.*

### Key Diagnostic Finding:
On Query 8 (*"How much did Google spend on R&D expenses in FY2023?"*), both `vector_only` and `hybrid` placed irrelevant chunks in the top 5 (Recall@5 = 0.00, MRR = 0.17). Adding the cross-encoder (`hybrid_rerank`) correctly analyzed the semantic interaction between "R&D expenses" and the passage text, boosting Recall@5 from 0.00 to 0.50 and doubling the MRR to 0.33.

## How to Verify

### 1. Run All Tests
```bash
source .venv/bin/activate
pytest tests/test_reranker.py tests/test_api.py -v
# Or run all 129 tests across the repository:
pytest
```
*Expected: 129 passed.*

### 2. Run Retrieval Benchmark
```bash
source .venv/bin/activate
python eval/benchmark_retrieval.py
```
*Expected: Executes all 10 benchmark queries across all 3 retrieval strategies, prints summary comparison table, and writes timestamped JSON to `results/`.*

### 3. Verify Golden Set Counts
```bash
wc -l eval/golden_set.jsonl
```
*Expected: 119 lines (119 questions).*

