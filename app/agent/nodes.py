"""
Node implementations for the Financial Agentic RAG StateGraph.

Each node receives the current AgentState and returns a dictionary of updated fields.
The nodes:
1. retrieve_node: Hybrid search (dense + sparse with RRF) -> Cross-encoder rerank.
2. generate_node: Calls LLM with grounded prompt, citation validation & confidence scoring.
3. rewrite_node: Reformulates query using GAAP terminology / section headers, increments retry_count.
4. abstain_node: Produces structured abstention when evidence is insufficient or retries exhausted.
5. route_decision: Conditional edge routing based on confidence score and retry budget.
"""

import time
import logging
from typing import Any, Dict

from app.agent.state import AgentState
from app.agent.rewriter import rewrite_query
from app.retrieval.vector_store import search_dense, search_sparse
from app.retrieval.rrf import reciprocal_rank_fusion
from app.rerank.reranker import rerank
from app.llm.client import get_llm_client
from app.observability.metrics import (
    RETRIEVAL_LATENCY,
    GENERATION_LATENCY,
    CHUNKS_RETRIEVED,
)

logger = logging.getLogger(__name__)


def retrieve_node(state: AgentState) -> Dict[str, Any]:
    """
    Retrieve candidate chunks using hybrid search (dense + sparse + RRF)
    and refine them using the cross-encoder reranker.
    """
    start_time = time.time()
    query = state.get("current_query") or state.get("query", "")
    top_k = state.get("top_k", 5)
    ticker = state.get("ticker_filter")
    doc_type = state.get("doc_type_filter")

    logger.info("Executing retrieve_node for query: '%s' (top_k=%d, ticker=%s)", query, top_k, ticker)

    retrieval_k = min(max(top_k * 4, 20), 50)

    try:
        dense_results = search_dense(
            query=query,
            top_k=retrieval_k,
            ticker_filter=ticker,
            doc_type_filter=doc_type,
        )
    except Exception as e:
        logger.warning("Dense search encountered an error: %s", e)
        dense_results = []

    try:
        sparse_results = search_sparse(
            query=query,
            top_k=retrieval_k,
            ticker_filter=ticker,
            doc_type_filter=doc_type,
        )
    except Exception as e:
        logger.warning("Sparse search encountered an error: %s", e)
        sparse_results = []

    fused = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        k=60,
        top_n=retrieval_k,
    )

    if fused:
        top_chunks = rerank(
            query=query,
            results=fused,
            top_k=top_k,
        )
    else:
        top_chunks = []

    elapsed = time.time() - start_time
    RETRIEVAL_LATENCY.observe(elapsed)
    CHUNKS_RETRIEVED.observe(len(top_chunks))

    logger.info("retrieve_node retrieved and reranked %d chunks in %.3f s", len(top_chunks), elapsed)
    return {"retrieved_chunks": top_chunks}


def generate_node(state: AgentState) -> Dict[str, Any]:
    """
    Generate grounded response using the retrieved chunks, validate citations,
    and compute multi-signal calibrated confidence.
    """
    start_time = time.time()
    query = state.get("current_query") or state.get("query", "")
    chunks = state.get("retrieved_chunks", [])

    logger.info("Executing generate_node with %d context chunks", len(chunks))

    client = get_llm_client()
    answer_resp = client.answer_query(
        query=query,
        context_chunks=chunks,
    )

    elapsed = time.time() - start_time
    GENERATION_LATENCY.observe(elapsed)

    return {
        "answer": answer_resp.answer,
        "citations": answer_resp.citations,
        "confidence": answer_resp.confidence,
        "decision": answer_resp.decision,
    }


def rewrite_node(state: AgentState) -> Dict[str, Any]:
    """
    Rewrite the query to address terminology mismatch or conversational noise,
    incrementing the retry count and tracking the rewrite history.
    """
    current_q = state.get("current_query") or state.get("query", "")
    orig_q = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    history = list(state.get("rewrite_history", []))

    if not history:
        history.append(orig_q)

    client = get_llm_client()
    rewritten = rewrite_query(
        current_query=current_q,
        original_query=orig_q,
        retry_count=retry_count + 1,
        ticker_filter=state.get("ticker_filter"),
        doc_type_filter=state.get("doc_type_filter"),
        llm_client=client,
    )

    history.append(rewritten)
    logger.info("Query rewrite attempt %d: '%s' -> '%s'", retry_count + 1, current_q, rewritten)

    return {
        "current_query": rewritten,
        "retry_count": retry_count + 1,
        "rewrite_history": history,
    }


def abstain_node(state: AgentState) -> Dict[str, Any]:
    """
    Terminal node when evidence is insufficient or retry attempts are exhausted.
    Ensures safe, disciplined financial abstention without hallucination.
    """
    logger.info("Executing abstain_node. Agent declining to answer.")
    abstain_message = (
        "INSUFFICIENT_EVIDENCE: Unable to find sufficiently verifiable financial evidence "
        "in the indexed SEC filings or transcripts to answer this question with high confidence."
    )
    return {
        "answer": abstain_message,
        "citations": [],
        "decision": "abstain",
        "confidence": min(state.get("confidence", 0.0), 0.35),
    }


def route_decision(state: AgentState) -> str:
    """
    Conditional routing function evaluating the confidence-calibrated decision.

    Routing rules:
    - If decision is 'generate' -> 'generate' (navigate to END)
    - If decision is 'rewrite':
        * If retry_count < max_retries -> 'rewrite' (attempt another retrieval pass)
        * If retry_count >= max_retries -> 'abstain' (retry budget exhausted)
    - If decision is 'abstain' -> 'abstain' (navigate to abstain node)
    """
    decision = state.get("decision", "abstain")
    retries = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    logger.info(
        "Routing decision check: decision='%s', retries=%d/%d, confidence=%.3f",
        decision,
        retries,
        max_retries,
        state.get("confidence", 0.0),
    )

    if decision == "generate":
        return "generate"
    elif decision == "rewrite":
        if retries < max_retries:
            return "rewrite"
        else:
            logger.info("Retry budget exhausted (%d >= %d). Routing to abstain.", retries, max_retries)
            return "abstain"
    else:
        return "abstain"

