# Day 5: Multi-Provider LLM Client, Grounded Generation & Citation Verification

## What Was Implemented

### 1. Multi-Provider LLM Client (`app/llm/client.py`)
- Designed and built a resilient multi-provider LLM client:
  - **Primary Provider**: Groq running `llama-3.1-8b-instant` on LPUs for ultra-fast inference (~500+ tok/s).
  - **Fallback Provider**: Google Gemini running `gemini-1.5-flash` for automatic failover when Groq experiences rate limits (429) or transient outages.
  - **Deterministic Mock Generator**: Offline fallback ensuring automated test suites and local development run without requiring API keys or spending quota.
- Implemented error discrimination resilience via `tenacity`:
  - Retries transient rate limits (429) and 5xx network hiccups with exponential backoff (`1s`, `2s`, `4s`).
  - Fast-fails permanent authentication errors (401/403) to trigger immediate provider failover without wasted latency.

### 2. Grounded Financial Prompt Engineering (`app/llm/prompts.py`)
- Authored `GROUNDED_SYSTEM_PROMPT` establishing strict closed-book RAG rules:
  - Explicit prohibition of external knowledge or speculation for factual financial claims.
  - Mandatory citation format `[Chunk: <chunk_id>]` for every claim.
  - Explicit directive to return `INSUFFICIENT_EVIDENCE` when retrieved context chunks do not support the query.
- Implemented `build_rag_context()`: formats retrieved chunks with document ID, ticker, document type (10-K, 10-Q, earnings call), and section headers.
- Implemented `format_grounded_user_prompt()`: bundles user question and numbered context blocks into a structured prompt.

### 3. Two-Tier Citation Validator (`app/llm/citation_validator.py`)
- Implemented deterministic citation auditing:
  - **Tier 1 (Chunk Existence Check)**: Verifies that every cited chunk ID was actually present in the retrieved candidate pool.
  - **Tier 2 (Quote Verification Check)**: Verifies that the cited verbatim excerpt exists in the source chunk text via normalized substring or high-overlap token matching.
- Populates citation provenance metadata (`doc_id`, `ticker`, `section`, `page`).
- Returns detailed diagnostics: `citation_precision`, `quote_precision`, `hallucinated_chunk_ids`, and `unverified_quotes`.
- Filters hallucinated citations from user-facing responses and calibrates confidence score.

### 4. Grounded Generation API Endpoint (`app/api/routes.py`)
- Added `POST /api/v1/generate`:
  - Receives `GenerateRequest` (query, top_k, optional ticker/doc_type filters).
  - Executes full two-stage retrieval (hybrid dense+BM25 with RRF $\to$ cross-encoder rerank).
  - Executes grounded LLM generation with automatic provider failover.
  - Runs citation validation and confidence calibration.
  - Returns `AnswerResponse` with verified citations, confidence score, and decision flag (`generate` vs `abstain`).

### 5. Grounded Generation Demonstration Script (`scripts/demo_grounded_generation.py`)
- Implemented interactive verification script demonstrating:
  - **Scenario 1**: Grounded answer with verified chunk citation and verbatim quote.
  - **Scenario 2**: Abstention on unanswerable query with `INSUFFICIENT_EVIDENCE`.
  - **Scenario 3**: Citation validator catching fabricated chunk IDs and unverified quotes.

## Files Added/Changed

| File | Purpose |
|---|---|
| `app/llm/client.py` | Multi-provider LLM client with Groq, Gemini fallback, retry, and mock mode |
| `app/llm/prompts.py` | System prompt, context builder, and grounded prompt formatting |
| `app/llm/citation_validator.py` | Two-tier citation existence and verbatim quote verification engine |
| `app/llm/__init__.py` | Package exports for LLM client, prompts, and citation validator |
| `app/schemas.py` | Added `GenerateRequest` schema |
| `app/api/routes.py` | Added `POST /api/v1/generate` endpoint |
| `scripts/demo_grounded_generation.py` | Standalone demonstration script for grounded Q&A and validation |
| `tests/test_prompts.py` | Unit tests for context formatting and system prompt rules |
| `tests/test_citation_validator.py` | Unit tests for valid, hallucinated, and fuzzy quote citations |
| `tests/test_llm_client.py` | Unit tests for client failover, retries, and confidence calibration |
| `tests/test_api.py` | Added tests in `TestGenerateEndpoint` for `/generate` API |
| `docs/DECISIONS.md` | Recorded Decisions 17, 18, and 19 on multi-provider, citations, and retries |

## How to Verify

### 1. Run Complete Test Suite
```bash
source .venv/bin/activate
pytest tests/test_prompts.py tests/test_citation_validator.py tests/test_llm_client.py tests/test_api.py -v
# Run all 147 tests across the codebase:
pytest
```
*Expected: 147 passed.*

### 2. Run Grounded Generation Demo
```bash
source .venv/bin/activate
python scripts/demo_grounded_generation.py
```
*Expected: Executes Scenario 1 (grounded generation), Scenario 2 (abstention), and Scenario 3 (auditing hallucinated citations).*

### 3. Test the Endpoint via Curl
```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{"query": "What was Apple revenue in 2023?", "top_k": 3}'
```
