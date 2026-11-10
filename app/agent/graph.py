"""
LangGraph StateGraph definition for the Financial Document Intelligence Engine.

HAND-WRITE PIECE #3 (Day 7).

This module defines the agentic loop that connects retrieval, reranking,
grounded generation, multi-signal confidence evaluation, query rewriting,
and disciplined abstention into a unified state machine.
"""

import time
import logging
from typing import Optional, Any
from langgraph.graph import StateGraph, START, END

from app.schemas import AnswerResponse
from app.agent.state import AgentState
from app.agent.nodes import (
    retrieve_node,
    generate_node,
    rewrite_node,
    abstain_node,
    route_decision,
)

logger = logging.getLogger(__name__)


def build_agent_graph() -> Any:
    r"""
    Construct and compile the LangGraph StateGraph for the agentic RAG loop.

    Workflow Topology:
            +-------+
            | START |
            +-------+
                |
                v
          +------------+
          |  retrieve  |<--------------------+
          +------------+                     |
                |                            |
                v                            |
          +------------+                     |
          |  generate  |                     |
          +------------+                     |
                |                            |
         [route_decision]                    |
          /     |      \                     |
   "generate" "rewrite" "abstain"            |
        /       |        \                   |
       v        v         v                  |
    +-----+ +---------+ +---------+          |
    | END | | rewrite | | abstain |          |
    +-----+ +---------+ +---------+          |
                 |            |              |
                 +------------+              |
                              |              |
                              v              |
                           +-----+           |
                           | END |           |
                           +-----+           |

    Edges:
    1. START -> 'retrieve'
    2. 'retrieve' -> 'generate'
    3. 'generate' -> route_decision:
         - 'generate' -> END
         - 'rewrite'  -> 'rewrite'
         - 'abstain'  -> 'abstain'
    4. 'rewrite' -> 'retrieve' (agentic retry loop)
    5. 'abstain' -> END

    Returns:
        CompiledStateGraph: The compiled LangGraph application.
    """
    # 1. Initialize StateGraph with our typed state
    builder = StateGraph(AgentState)

    # 2. Register all workflow nodes
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("rewrite", rewrite_node)
    builder.add_node("abstain", abstain_node)

    # 3. Connect entry point and sequential edges
    builder.add_edge(START, "retrieve")
    builder.add_edge("retrieve", "generate")

    # 4. Attach conditional routing logic from generate
    builder.add_conditional_edges(
        "generate",
        route_decision,
        {
            "generate": END,
            "rewrite": "rewrite",
            "abstain": "abstain",
        },
    )

    # 5. Connect rewrite loop back to retrieve (the agentic cycle)
    builder.add_edge("rewrite", "retrieve")

    # 6. Connect terminal abstain node to END
    builder.add_edge("abstain", END)

    # 7. Compile state graph into runnable application
    return builder.compile()


# Module-level compiled graph cache
_agent_graph: Optional[Any] = None


def get_agent_graph() -> Any:
    """Get or lazily compile the singleton agent graph."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_financial_agent(
    query: str,
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    max_retries: int = 2,
    graph: Optional[Any] = None,
    use_cache: bool = True,
) -> AnswerResponse:
    """
    Execute the financial agentic loop over a query.

    Args:
        query: User's financial question.
        top_k: Number of reranked context chunks to select for answering.
        ticker_filter: Optional company ticker filter (e.g. 'AAPL').
        doc_type_filter: Optional SEC filing type filter (e.g. '10-K').
        max_retries: Maximum rewrite cycles allowed (default: 2).
        graph: Optional pre-compiled LangGraph StateGraph instance.
        use_cache: Whether to consult and populate the semantic cache (default: True).

    Returns:
        AnswerResponse with final answer, validated citations, confidence,
        decision, and count of retries performed.
    """
    start_time = time.time()

    # 1. Check L1/L2 Semantic Cache first (Day 8)
    if use_cache:
        try:
            from app.cache.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cached_entry, hit_type = cache.get(
                query=query,
                ticker_filter=ticker_filter,
                doc_type_filter=doc_type_filter,
            )
            if cached_entry:
                elapsed_ms = (time.time() - start_time) * 1000
                logger.info("Serving query '%s' from %s cache (%.2f ms)", query, hit_type, elapsed_ms)
                return cached_entry.to_answer_response(query=query, latency_ms=elapsed_ms)
        except Exception as cache_err:
            logger.warning("Cache lookup encountered error (%s); falling back to graph execution", cache_err)

    if graph is None:
        graph = get_agent_graph()

    initial_state: AgentState = {
        "query": query,
        "current_query": query,
        "top_k": top_k,
        "ticker_filter": ticker_filter,
        "doc_type_filter": doc_type_filter,
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "confidence_breakdown": {},
        "decision": "abstain",
        "retry_count": 0,
        "max_retries": max_retries,
        "rewrite_history": [query],
        "latency_ms": 0.0,
        "cached": False,
        "error": None,
    }

    final_state = graph.invoke(initial_state)
    elapsed_ms = (time.time() - start_time) * 1000

    response = AnswerResponse(
        query=query,
        answer=final_state.get("answer", ""),
        citations=final_state.get("citations", []),
        confidence=final_state.get("confidence", 0.0),
        decision=final_state.get("decision", "abstain"),
        retries=final_state.get("retry_count", 0),
        latency_ms=round(elapsed_ms, 2),
        cached=final_state.get("cached", False),
    )

    # 3. If successfully answered with high confidence, populate cache with lineage (Day 8)
    if use_cache and response.decision == "generate":
        try:
            from app.cache.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cache.set(
                query=query,
                answer_response=response,
                ticker_filter=ticker_filter,
                doc_type_filter=doc_type_filter,
            )
        except Exception as cache_save_err:
            logger.warning("Could not cache answer (%s)", cache_save_err)

    return response
