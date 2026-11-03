# Day 3: Sparse (BM25) Retrieval, Named Vectors in Qdrant, and Hybrid Search (RRF)

## What Was Implemented

### 1. BM25 Sparse Vector Encoder (`app/retrieval/sparse.py`)
- Term frequency and inverse document frequency (TF-IDF / BM25) calculation using Okapi BM25 math.
- Text tokenization with alphanumeric boundary extraction and short-token noise filtering.
- Stateless 32-bit hash-based term-to-index mapping (`_term_to_index`), avoiding centralized vocabulary state.
- Document-level sparse vector encoding using BM25 TF length-normalized weights.
- Query-level sparse vector encoding using corpus-wide IDF term weights.
- Mathematical decomposition where dot product between query and document sparse vectors produces exact BM25 scores.

### 2. Named Vectors in Qdrant Vector Store (`app/retrieval/vector_store.py`)
- Upgraded Qdrant collection architecture from a single unnamed vector to co-located named vectors:
  - `"dense"`: 384-dimensional dense vectors with Cosine distance.
  - `"sparse"`: Variable-length BM25 sparse vectors.
- Single atomic upsert writes both dense embeddings and sparse BM25 vectors simultaneously.
- Implemented `search_sparse()`: queries the named `"sparse"` index with payload filters (`ticker`, `doc_type`).
- Unified payload filtering across both dense and sparse retrieval paths.

### 3. Reciprocal Rank Fusion Hand-Write Module (`app/retrieval/rrf.py`)
- Defined the core algorithm specification, parameter contracts, and edge cases for rank-based result fusion.
- Hand-write piece #1: Function signature and detailed docstrings provided with a `NotImplementedError` and TODO.
- Reference implementation placed in `docs/learn/solutions/rrf_solution.py`.
- Comprehensive test suite in `tests/test_rrf.py` verifying rank aggregation, duplicate handling, metadata preservation, and top-$N$ truncation.

### 4. Hybrid Search Pipeline & Endpoints (`app/api/routes.py`)
- Added `POST /api/v1/search/sparse`: dedicated BM25 keyword search endpoint for debugging and A/B benchmarking.
- Added `POST /api/v1/search/hybrid`: executes dense and sparse searches in parallel with $2\times$ over-retrieval, fusing candidate rankings via Reciprocal Rank Fusion.
- End-to-end integration into the ingestion pipeline (`app/ingestion/pipeline.py`) to fit BM25 and index sparse vectors upon document loading.

### 5. Multi-Strategy Retrieval Comparison Script (`scripts/demo_retrieval_comparison.py`)
- Standalone demonstration script showing concrete failure and success modes:
  - **Scenario A (Dense Wins)**: Semantic paraphrases without lexical keyword overlap.
  - **Scenario B (Sparse Wins)**: Exact regulatory section codes ("Item 1A") and technical keywords.
  - **Scenario C (Hybrid Wins)**: Entity-constrained queries requiring both keyword anchoring and semantic context.

## Files Added/Changed

| File | Purpose |
|---|---|
| `app/retrieval/sparse.py` | BM25 tokenization, IDF fitting, and sparse vector generation |
| `app/retrieval/rrf.py` | Reciprocal Rank Fusion hand-write skeleton with algorithm specs |
| `docs/learn/solutions/rrf_solution.py` | Reference solution for RRF fusion |
| `app/retrieval/vector_store.py` | Qdrant named vectors configuration, sparse search, and atomic upsert |
| `app/retrieval/__init__.py` | Exported sparse search, BM25 encoder, and RRF interface |
| `app/ingestion/pipeline.py` | Wired BM25 fitting and sparse vector generation into ingestion |
| `app/api/routes.py` | Added `/search/sparse` and `/search/hybrid` endpoints |
| `tests/test_sparse.py` | Unit tests for tokenizer, hashing, IDF computation, and sparse vectors |
| `tests/test_rrf.py` | 15 unit tests covering all RRF rank fusion edge cases |
| `tests/test_vector_store.py` | Integration tests for named vector upsert and sparse search |
| `tests/test_api.py` | Endpoint tests for sparse search |
| `scripts/demo_retrieval_comparison.py` | Interactive demo demonstrating queries where each retrieval method wins |
| `docs/DECISIONS.md` | Architecture decisions 11 through 13 |
| `docs/versions/v3-day3.md` | Day 3 release log and changelog |
| `docs/learn/day3-learn.md` | First-principles guide to BM25, sparse representations, and RRF |

## New Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/v1/search/sparse` | Keyword search via Qdrant named BM25 sparse vectors |
| POST | `/api/v1/search/hybrid` | Hybrid search fusing dense + sparse candidate lists via RRF |

## How to Run and Verify

```bash
# 1. Run unit tests for sparse vectors and vector store
pytest tests/test_sparse.py tests/test_vector_store.py tests/test_api.py

# 2. Run the retrieval comparison demo script
python scripts/demo_retrieval_comparison.py

# 3. Test RRF hand-write implementation
pytest tests/test_rrf.py
```

