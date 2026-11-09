"""
Tests for the LangGraph Financial Agent (Day 7).

Tests:
1. AgentState schema and type verification.
2. route_decision conditional router branches (generate, rewrite, abstain, retry budget exhaustion).
3. Deterministic query expansion (filler removal, ticker injection, GAAP synonym mapping).
4. rewrite_node state mutation and history tracking.
5. abstain_node disciplined output format.
6. build_agent_graph StateGraph compilation and topology verification.
7. End-to-end agent workflow execution (direct generate, rewrite loop, forced abstention).
"""

from unittest.mock import MagicMock, patch
import pytest

from app.schemas import Citation, SearchResult
from app.agent.state import AgentState
from app.agent.rewriter import (
    deterministic_financial_rewrite,
    rewrite_query,
)
from app.agent.nodes import (
    retrieve_node,
    generate_node,
    rewrite_node,
    abstain_node,
    route_decision,
)
from app.agent.graph import build_agent_graph, run_financial_agent


# ============================================================================
# 1. State & Router Tests
# ============================================================================

def test_agent_state_keys():
    """Verify AgentState supports all required execution keys."""
    state: AgentState = {
        "query": "What was Apple revenue in 2023?",
        "current_query": "What was Apple revenue in 2023?",
        "top_k": 5,
        "ticker_filter": "AAPL",
        "doc_type_filter": "10-K",
        "retrieved_chunks": [],
        "answer": "",
        "citations": [],
        "confidence": 0.85,
        "decision": "generate",
        "retry_count": 0,
        "max_retries": 2,
        "rewrite_history": ["What was Apple revenue in 2023?"],
        "latency_ms": 12.5,
        "cached": False,
        "error": None,
    }
    assert state["query"] == "What was Apple revenue in 2023?"
    assert state["decision"] == "generate"
    assert state["retry_count"] == 0


def test_route_decision_generate():
    """High confidence decision 'generate' routes directly to 'generate'."""
    state: AgentState = {
        "decision": "generate",
        "retry_count": 0,
        "max_retries": 2,
        "confidence": 0.88,
    }
    assert route_decision(state) == "generate"


def test_route_decision_rewrite_allowed():
    """Ambiguous decision 'rewrite' routes to 'rewrite' if retries are within budget."""
    state: AgentState = {
        "decision": "rewrite",
        "retry_count": 0,
        "max_retries": 2,
        "confidence": 0.55,
    }
    assert route_decision(state) == "rewrite"

    state["retry_count"] = 1
    assert route_decision(state) == "rewrite"


def test_route_decision_rewrite_budget_exhausted():
    """Ambiguous decision 'rewrite' routes to 'abstain' when max_retries is reached."""
    state: AgentState = {
        "decision": "rewrite",
        "retry_count": 2,
        "max_retries": 2,
        "confidence": 0.52,
    }
    assert route_decision(state) == "abstain"


def test_route_decision_abstain():
    """Low confidence decision 'abstain' routes directly to 'abstain'."""
    state: AgentState = {
        "decision": "abstain",
        "retry_count": 0,
        "max_retries": 2,
        "confidence": 0.25,
    }
    assert route_decision(state) == "abstain"


def test_route_decision_fallback_on_empty():
    """Missing or unexpected decision defaults safely to 'abstain'."""
    state: AgentState = {}
    assert route_decision(state) == "abstain"


# ============================================================================
# 2. Query Rewriting Tests
# ============================================================================

def test_deterministic_rewrite_removes_fillers():
    """Filler phrases like 'can you please tell me' are stripped out."""
    raw_query = "Can you please tell me what was Apple's total sales?"
    rewritten = deterministic_financial_rewrite(raw_query)
    assert "can you please tell me" not in rewritten.lower()
    assert "what was" not in rewritten.lower()
    assert "sales" in rewritten.lower() or "net sales" in rewritten.lower()


def test_deterministic_rewrite_injects_ticker():
    """Ticker is injected if missing from the raw query."""
    raw_query = "total long term debt obligations"
    rewritten = deterministic_financial_rewrite(raw_query, ticker="AAPL")
    assert "AAPL" in rewritten


def test_deterministic_rewrite_gaap_synonyms():
    """Informal terms like 'sales' and 'profit' expand to GAAP terminology."""
    sales_q = "Apple sales 2023"
    expanded_sales = deterministic_financial_rewrite(sales_q)
    assert "net sales" in expanded_sales.lower() or "revenue" in expanded_sales.lower()

    debt_q = "Apple debt 2023"
    expanded_debt = deterministic_financial_rewrite(debt_q)
    assert "liabilities" in expanded_debt.lower() or "commercial paper" in expanded_debt.lower()


def test_deterministic_rewrite_progressive_deepening():
    """Retry count 1 adds Item 7/8 headers; retry count 2 adds statement headers."""
    q = "Apple capital expenditure"
    retry_1 = deterministic_financial_rewrite(q, retry_count=1)
    assert "Item 7" in retry_1 or "Financial Statements" in retry_1

    retry_2 = deterministic_financial_rewrite(q, retry_count=2)
    assert "Consolidated Statements" in retry_2 or "Cash Flows" in retry_2


def test_rewrite_node_updates_state():
    """rewrite_node increments retry_count and logs history."""
    state: AgentState = {
        "query": "Apple total debt",
        "current_query": "Apple total debt",
        "retry_count": 0,
        "max_retries": 2,
        "rewrite_history": ["Apple total debt"],
        "ticker_filter": "AAPL",
    }
    with patch("app.agent.nodes.get_llm_client") as mock_client:
        mock_client.return_value = MagicMock(groq_api_key=None)
        update = rewrite_node(state)

    assert update["retry_count"] == 1
    assert len(update["rewrite_history"]) == 2
    assert update["current_query"] != state["query"]


# ============================================================================
# 3. Abstain Node Tests
# ============================================================================

def test_abstain_node_payload():
    """abstain_node produces a disciplined INSUFFICIENT_EVIDENCE message and clears citations."""
    state: AgentState = {
        "query": "What will Apple's stock price be in 2040?",
        "confidence": 0.20,
    }
    result = abstain_node(state)
    assert "INSUFFICIENT_EVIDENCE" in result["answer"]
    assert result["citations"] == []
    assert result["decision"] == "abstain"
    assert result["confidence"] <= 0.35


# ============================================================================
# 4. Graph Construction & Execution Tests
# ============================================================================

def test_build_agent_graph_structure():
    """Verify build_agent_graph returns a compiled LangGraph with expected nodes."""
    try:
        graph = build_agent_graph()
    except NotImplementedError:
        pytest.skip("Hand-write piece #3 not yet implemented in app/agent/graph.py")

    assert graph is not None
    # LangGraph compiled graph inspectability
    nodes = graph.nodes
    for expected_node in ["retrieve", "generate", "rewrite", "abstain"]:
        assert expected_node in nodes


def test_run_financial_agent_direct_generate():
    """Mock test: high confidence produces immediate answer with 0 retries."""
    try:
        graph = build_agent_graph()
    except NotImplementedError:
        pytest.skip("Hand-write piece #3 not yet implemented in app/agent/graph.py")

    fake_chunk = SearchResult(
        chunk_id="AAPL_10-K_2023_chunk_0001",
        text="Apple net sales were $383,285 million in 2023.",
        score=0.92,
        metadata={"ticker": "AAPL", "doc_id": "AAPL_10-K_2023"},
    )

    with patch("app.agent.nodes.search_dense", return_value=[fake_chunk]), \
         patch("app.agent.nodes.search_sparse", return_value=[fake_chunk]), \
         patch("app.agent.nodes.rerank", return_value=[fake_chunk]), \
         patch("app.agent.nodes.get_llm_client") as mock_get_client:

        mock_llm = MagicMock()
        mock_llm.answer_query.return_value = MagicMock(
            answer="Apple net sales were $383.3 billion [Chunk: AAPL_10-K_2023_chunk_0001].",
            citations=[
                Citation(
                    chunk_id="AAPL_10-K_2023_chunk_0001",
                    doc_id="AAPL_10-K_2023",
                    quote="Apple net sales were $383,285 million in 2023.",
                )
            ],
            confidence=0.88,
            decision="generate",
        )
        mock_get_client.return_value = mock_llm

        response = run_financial_agent(
            query="What was Apple net sales in 2023?",
            top_k=1,
            ticker_filter="AAPL",
            graph=graph,
        )

        assert response.decision == "generate"
        assert response.confidence == 0.88
        assert response.retries == 0
        assert len(response.citations) == 1
        assert "383.3" in response.answer


def test_run_financial_agent_rewrite_loop():
    """Mock test: pass 1 returns ambiguous confidence ('rewrite'), pass 2 succeeds ('generate')."""
    try:
        graph = build_agent_graph()
    except NotImplementedError:
        pytest.skip("Hand-write piece #3 not yet implemented in app/agent/graph.py")

    fake_chunk = SearchResult(
        chunk_id="AAPL_10-K_2023_chunk_0001",
        text="Apple net sales were $383,285 million in 2023.",
        score=0.75,
        metadata={"ticker": "AAPL", "doc_id": "AAPL_10-K_2023"},
    )

    call_count = 0

    def mock_answer(query, context_chunks):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First pass: ambiguous
            return MagicMock(
                answer="Sales numbers are partially discussed.",
                citations=[],
                confidence=0.55,
                decision="rewrite",
            )
        else:
            # Second pass after rewrite: high confidence
            return MagicMock(
                answer="Apple net sales were $383.3B [Chunk: AAPL_10-K_2023_chunk_0001].",
                citations=[Citation(chunk_id="AAPL_10-K_2023_chunk_0001", doc_id="AAPL_10-K_2023", quote="Apple net sales")],
                confidence=0.85,
                decision="generate",
            )

    with patch("app.agent.nodes.search_dense", return_value=[fake_chunk]), \
         patch("app.agent.nodes.search_sparse", return_value=[fake_chunk]), \
         patch("app.agent.nodes.rerank", return_value=[fake_chunk]), \
         patch("app.agent.nodes.get_llm_client") as mock_get_client:

        mock_llm = MagicMock()
        mock_llm.groq_api_key = None
        mock_llm.answer_query.side_effect = mock_answer
        mock_get_client.return_value = mock_llm

        response = run_financial_agent(
            query="tell me about Apple sales",
            top_k=1,
            ticker_filter="AAPL",
            max_retries=2,
            graph=graph,
        )

        assert response.decision == "generate"
        assert response.confidence == 0.85
        assert response.retries == 1
        assert len(response.citations) == 1
        assert call_count == 2


def test_run_financial_agent_abstain_path():
    """Mock test: unanswerable query with insufficient evidence triggers abstain."""
    try:
        graph = build_agent_graph()
    except NotImplementedError:
        pytest.skip("Hand-write piece #3 not yet implemented in app/agent/graph.py")

    with patch("app.agent.nodes.search_dense", return_value=[]), \
         patch("app.agent.nodes.search_sparse", return_value=[]), \
         patch("app.agent.nodes.rerank", return_value=[]), \
         patch("app.agent.nodes.get_llm_client") as mock_get_client:

        mock_llm = MagicMock()
        mock_llm.answer_query.return_value = MagicMock(
            answer="INSUFFICIENT_EVIDENCE: No data found.",
            citations=[],
            confidence=0.0,
            decision="abstain",
        )
        mock_get_client.return_value = mock_llm

        response = run_financial_agent(
            query="Apple projected revenue in 2075",
            top_k=2,
            graph=graph,
        )

        assert response.decision == "abstain"
        assert response.confidence <= 0.35
        assert "INSUFFICIENT_EVIDENCE" in response.answer

