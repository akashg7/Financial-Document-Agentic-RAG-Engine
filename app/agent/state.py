"""
State definition for the LangGraph agentic loop.

In LangGraph, state is represented as a TypedDict (or Pydantic model) that flows
between nodes. Each node receives the current state and returns a dictionary of
fields to update, which LangGraph merges into the state.
"""

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

from app.schemas import Citation, SearchResult


class AgentState(TypedDict, total=False):
    """
    State threaded through the Financial Agent's execution graph.

    WHY TypedDict instead of Pydantic:
    - LangGraph natively supports TypedDict for state passage.
    - Nodes can return partial dictionaries containing only mutated keys,
      which LangGraph merges into the main state automatically.
    - Avoids Pydantic model recreation/validation overhead on every edge traversal.
    """

    # --- Query inputs ---
    query: str  # Original natural language question from the user
    current_query: str  # Currently active query (updated on rewrites)
    top_k: int  # Number of reranked chunks to select for generation (default: 5)
    ticker_filter: Optional[str]  # Optional ticker filter, e.g. "AAPL"
    doc_type_filter: Optional[str]  # Optional doc type filter, e.g. "10-K"

    # --- Retrieval & Rerank outputs ---
    retrieved_chunks: List[SearchResult]  # Top candidate chunks after reranking

    # --- Generation & Confidence outputs ---
    answer: str  # Generated answer or abstention message
    citations: List[Citation]  # Validated citations bound to context chunks
    confidence: float  # Calibrated composite confidence score [0.0, 1.0]
    confidence_breakdown: Dict[str, Any]  # Detailed scoring signals (retrieval, grounding, linguistic)
    decision: str  # Calibrated action: 'generate', 'rewrite', or 'abstain'

    # --- Agentic Loop / Retry tracking ---
    retry_count: int  # How many rewrite iterations have occurred so far (0-indexed)
    max_retries: int  # Maximum allowed rewrites before forcing abstention (default: 2)
    rewrite_history: List[str]  # History of reformulated queries: [query_0, query_1, ...]

    # --- Metadata & Telemetry ---
    latency_ms: float  # Total latency in milliseconds
    cached: bool  # Whether the answer was served from semantic cache (Day 8)
    error: Optional[str]  # Error message if any node failed

