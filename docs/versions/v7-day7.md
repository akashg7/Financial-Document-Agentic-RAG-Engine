# Day 7: Agentic Loop (Hand-Write Piece #3), StateGraph, Query Rewriter & Conditional Router

## What Was Implemented

### 1. LangGraph StateGraph Definition (`app/agent/graph.py`) — Hand-Write Piece #3
- Implemented **Hand-Write Piece #3**: constructed and compiled the cyclical `StateGraph`:
  - Defined explicit typed execution state [`AgentState`](file:///Users/akashg/Desktop/Financial-Document-Agentic-RAG-Engine/app/agent/state.py).
  - Registered four discrete workflow nodes: `retrieve`, `generate`, `rewrite`, and `abstain`.
  - Configured entry edge: `START -> 'retrieve'`.
  - Configured linear pipeline edge: `'retrieve' -> 'generate'`.
  - Configured conditional edge via `route_decision`:
    - $\text{Confidence} \ge 0.70 \implies \mathbf{END}$ (direct high-confidence delivery).
    - $0.40 \le \text{Confidence} < 0.70 \land \text{retries} < \text{max\_retries} \implies \mathbf{"rewrite"}$ (triggers cyclical retry edge).
    - $\text{Confidence} < 0.40 \lor \text{retries} \ge \text{max\_retries} \implies \mathbf{"abstain"}$ (budget exhausted or missing evidence).
  - Configured cyclical loop edge: `'rewrite' -> 'retrieve'`.
  - Configured termination edge: `'abstain' -> END`.
- Reference solution created in `docs/learn/solutions/langgraph_solution.py`.

### 2. Typed State Machine (`app/agent/state.py`)
- Engineered `AgentState` TypedDict maintaining full trace context across cycles:
  - `query`, `current_query`, `top_k`, `ticker_filter`, `doc_type_filter`.
  - `retrieved_chunks`, `answer`, `citations`, `confidence`, `confidence_breakdown`, `decision`.
  - `retry_count`, `max_retries` (default 2), and `rewrite_history`.
  - `latency_ms`, `cached` (ready for Day 8 semantic caching), and `error`.

### 3. Financial Query Rewriter (`app/agent/rewriter.py`)
- Created hybrid LLM + deterministic query reformulation engine:
  - **Conversational Filter Stripping**: Cleans filler phrases (`"can you please tell me"`, `"what was"`, `"how much did"`).
  - **GAAP / Regulatory Terminology Expansion**: Maps informal phrasing to standard SEC terminology (e.g., `"sales"` $\to$ `"net sales total revenue consolidated operations"`, `"debt"` $\to$ `"total debt liabilities commercial paper term notes credit facility"`).
  - **Ticker & Document Type Injection**: Injects missing ticker or SEC form metadata (e.g., `AAPL`, `10-K`).
  - **Progressive Deepening**: Retry 1 appends regulatory section headers (`"Item 7 MD&A Item 8 Financial Statements"`); retry 2+ appends statement types (`"Consolidated Statements of Operations Balance Sheets Cash Flows"`).

### 4. Workflow Nodes & Conditional Router (`app/agent/nodes.py`)
- `retrieve_node`: Orchestrates hybrid search (dense + sparse + RRF) and cross-encoder reranking over `current_query`.
- `generate_node`: Calls LLM with grounded financial prompt, validates citations, and computes multi-signal calibrated confidence.
- `rewrite_node`: Executes `rewrite_query()`, increments `retry_count`, and records query history.
- `abstain_node`: Sets standardized disciplined financial abstention notice (`INSUFFICIENT_EVIDENCE`) and clears citations.
- `route_decision`: Evaluates decision and retry budget to route flow.

### 5. API Endpoint (`POST /api/v1/query`)
- Added `QueryRequest` schema and `POST /query` endpoint in `app/api/routes.py`, exposing the agentic loop over HTTP.

### 6. Interactive Demo Script (`scripts/demo_agentic_loop.py`)
- Exercises the three fundamental state machine paths:
  1. Direct high-confidence generation ($0$ retries, confidence $0.843$).
  2. Query rewriting loop ($1$ retry, confidence jumps from $0.55 \to 0.890$).
  3. Disciplined abstention ($0$ retries, confidence $0.000$).
- Execution metrics saved to `results/agentic_loop_demo_20261109T161001Z.json`.

---

## Files Added/Changed

| File | Purpose |
|---|---|
| `app/agent/graph.py` | Completed Hand-Write Piece #3: LangGraph StateGraph builder and runner |
| `app/agent/state.py` | TypedDict state definition for the agentic loop |
| `app/agent/nodes.py` | Nodes (`retrieve`, `generate`, `rewrite`, `abstain`) and conditional router |
| `app/agent/rewriter.py` | Financial query rewriter with GAAP expansion and progressive deepening |
| `docs/learn/solutions/langgraph_solution.py` | Reference solution for Hand-Write Piece #3 |
| `app/schemas.py` | Added `QueryRequest` schema |
| `app/api/routes.py` | Added `POST /api/v1/query` agent endpoint |
| `scripts/demo_agentic_loop.py` | Demo script exercising direct generate, rewrite cycle, and abstain |
| `results/agentic_loop_demo_20261109T161001Z.json` | Real execution log from agent loop demo |
| `tests/test_agent.py` | 16 unit tests for StateGraph, nodes, rewriter, and end-to-end flow |
| `tests/test_api.py` | Added tests for `POST /api/v1/query` |
| `docs/versions/v7-day7.md` | Version summary for Day 7 |
| `docs/learn/day7-learn.md` | In-depth engineering guide on cyclic agentic RAG and StateGraph |
| `docs/DECISIONS.md` | ADR-007 on LangGraph agent architecture and query rewrite policy |

---

## How to Verify

### 1. Run Agent Test Suite
```bash
source .venv/bin/activate
pytest tests/test_agent.py -v
```
*Expected: 16 passed.*

### 2. Run API Test Suite
```bash
source .venv/bin/activate
pytest tests/test_api.py -v
```
*Expected: 17 passed.*

### 3. Run Entire Repository Test Suite
```bash
source .venv/bin/activate
pytest
```
*Expected: 173 passed across all 14 test suites.*

### 4. Run the Agentic Loop Demo
```bash
source .venv/bin/activate
python scripts/demo_agentic_loop.py
```
*Expected: Executes scenarios 1, 2, and 3, outputs decisions, and writes timestamped JSON to `results/`.*
