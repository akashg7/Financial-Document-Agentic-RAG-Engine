"""
Server-Sent Events (SSE) streaming for the agentic query pipeline.

WHY SSE instead of WebSockets:
- SSE is simpler: unidirectional (server → client), works over standard HTTP,
  and auto-reconnects. Our use case is exactly one-way: stream status and
  answer chunks to the client.
- WebSockets would add complexity (bidirectional protocol, ping/pong,
  connection state management) for no benefit here.
- SSE works through proxies and load balancers without special configuration.

WHY we stream node status events, not just text chunks:
- The client can show a progress UI ("Retrieving documents…",
  "Generating answer…", "Rewriting query…") before any text arrives.
- If the agent abstains, the client knows immediately via a status event
  instead of waiting for the full response.
"""

import json
import time
import logging
from typing import Optional, AsyncIterator

from app.agent.state import AgentState
from app.schemas import AnswerResponse
from app.observability.metrics import (
    QUERIES_TOTAL,
    QUERY_LATENCY,
    CONFIDENCE_SCORE,
    CACHE_HITS,
    CACHE_MISSES,
    ACTIVE_QUERIES,
)

logger = logging.getLogger(__name__)


async def stream_agent_response(
    query: str,
    top_k: int = 5,
    ticker_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    max_retries: int = 2,
) -> AsyncIterator[str]:
    """
    Execute the agentic loop and yield SSE-formatted events.

    Event types:
    - 'status': Pipeline stage transitions (retrieving, generating, rewriting, abstaining)
    - 'chunk': Partial answer text (for streaming display)
    - 'done': Final complete JSON response
    - 'error': Error details if something goes wrong

    Each event is formatted as per SSE spec:
        event: <type>\ndata: <json>\n\n
    """
    start_time = time.time()

    # Helper to format SSE events
    def sse_event(event_type: str, data: dict) -> str:
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    # 1. Check cache first
    yield sse_event("status", {"stage": "cache_check", "message": "Checking semantic cache..."})

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
            resp = cached_entry.to_answer_response(query=query, latency_ms=elapsed_ms)
            tier_label = "L1_exact" if hit_type == "exact" else "L2_semantic"
            CACHE_HITS.labels(tier=tier_label).inc()
            QUERIES_TOTAL.labels(decision="generate").inc()
            QUERY_LATENCY.observe(elapsed_ms / 1000.0)
            CONFIDENCE_SCORE.observe(cached_entry.confidence)
            yield sse_event("status", {"stage": "cache_hit", "message": f"Cache hit ({hit_type})"})
            yield sse_event("chunk", {"text": resp.answer})
            yield sse_event("done", resp.model_dump())
            return
        else:
            CACHE_MISSES.inc()
    except Exception as cache_err:
        logger.warning("SSE: Cache lookup error (%s); continuing with full pipeline", cache_err)

    yield sse_event("status", {"stage": "cache_miss", "message": "Cache miss, executing pipeline..."})

    # 2. Import and run the graph node-by-node for streaming visibility
    from app.agent.graph import get_agent_graph
    from app.agent.nodes import retrieve_node, generate_node, rewrite_node, abstain_node

    graph = get_agent_graph()

    current_state: AgentState = {
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

    try:
        iteration = 0
        while iteration <= max_retries:
            # Retrieve
            yield sse_event("status", {
                "stage": "retrieving",
                "message": f"Retrieving documents (attempt {iteration + 1})...",
                "iteration": iteration + 1,
            })
            retrieve_updates = retrieve_node(current_state)
            current_state.update(retrieve_updates)

            chunk_count = len(current_state.get("retrieved_chunks", []))
            yield sse_event("status", {
                "stage": "retrieved",
                "message": f"Retrieved {chunk_count} chunks",
                "chunk_count": chunk_count,
            })

            # Generate
            yield sse_event("status", {
                "stage": "generating",
                "message": "Generating grounded answer...",
            })
            generate_updates = generate_node(current_state)
            current_state.update(generate_updates)

            decision = current_state.get("decision", "abstain")
            confidence = current_state.get("confidence", 0.0)

            yield sse_event("status", {
                "stage": "evaluated",
                "message": f"Confidence: {confidence:.3f}, Decision: {decision}",
                "confidence": confidence,
                "decision": decision,
            })

            if decision == "generate":
                # Stream the answer text as a chunk
                answer_text = current_state.get("answer", "")
                yield sse_event("chunk", {"text": answer_text})
                break

            elif decision == "rewrite" and current_state.get("retry_count", 0) < max_retries:
                yield sse_event("status", {
                    "stage": "rewriting",
                    "message": "Confidence too low, rewriting query...",
                })
                rewrite_updates = rewrite_node(current_state)
                current_state.update(rewrite_updates)
                yield sse_event("status", {
                    "stage": "rewritten",
                    "message": f"Rewritten to: {current_state.get('current_query', '')}",
                    "rewritten_query": current_state.get("current_query", ""),
                })
                iteration += 1
                continue

            else:
                # Abstain
                yield sse_event("status", {
                    "stage": "abstaining",
                    "message": "Insufficient evidence — abstaining",
                })
                abstain_updates = abstain_node(current_state)
                current_state.update(abstain_updates)
                answer_text = current_state.get("answer", "")
                yield sse_event("chunk", {"text": answer_text})
                break

    except Exception as e:
        logger.error("SSE streaming error: %s", e, exc_info=True)
        yield sse_event("error", {"message": str(e)})
        return

    # 3. Build final response
    elapsed_ms = (time.time() - start_time) * 1000

    response = AnswerResponse(
        query=query,
        answer=current_state.get("answer", ""),
        citations=current_state.get("citations", []),
        confidence=current_state.get("confidence", 0.0),
        decision=current_state.get("decision", "abstain"),
        retries=current_state.get("retry_count", 0),
        latency_ms=round(elapsed_ms, 2),
        cached=False,
    )

    # Record Prometheus metrics
    QUERIES_TOTAL.labels(decision=response.decision).inc()
    QUERY_LATENCY.observe(elapsed_ms / 1000.0)
    CONFIDENCE_SCORE.observe(response.confidence)

    # Cache successful answers
    if response.decision == "generate":
        try:
            from app.cache.semantic_cache import get_semantic_cache
            cache = get_semantic_cache()
            cache.set(
                query=query,
                answer_response=response,
                ticker_filter=ticker_filter,
                doc_type_filter=doc_type_filter,
            )
        except Exception as cache_err:
            logger.warning("SSE: Could not cache answer (%s)", cache_err)

    yield sse_event("done", response.model_dump())

