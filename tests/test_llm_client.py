"""
Unit tests for LLMClient, provider failover, and grounding pipeline (Day 5).
"""

from unittest.mock import MagicMock
import pytest

from app.schemas import SearchResult
from app.llm.client import (
    LLMClient,
    RateLimitError,
    ProviderUnavailableError,
    get_llm_client,
    set_llm_client,
    _clean_json_text,
)


@pytest.fixture
def sample_chunks():
    return [
        SearchResult(
            chunk_id="AAPL_chunk_001",
            text="Apple reported net sales of $383.3 billion in fiscal year 2023.",
            score=0.92,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL"},
        ),
        SearchResult(
            chunk_id="AAPL_chunk_002",
            text="iPhone sales represented $200.6 billion of total net sales.",
            score=0.85,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL"},
        ),
    ]


def test_clean_json_text():
    """Strips markdown code fences and whitespace."""
    raw = "```json\n{\"answer\": \"test\"}\n```"
    assert _clean_json_text(raw) == "{\"answer\": \"test\"}"

    raw_bare = "```\n{\"answer\": \"test\"}\n```"
    assert _clean_json_text(raw_bare) == "{\"answer\": \"test\"}"

    raw_clean = "{\"answer\": \"test\"}"
    assert _clean_json_text(raw_clean) == "{\"answer\": \"test\"}"


def test_mock_llm_client_generation(sample_chunks):
    """Client in mock mode (no keys configured) produces grounded response."""
    client = LLMClient(groq_api_key="", gemini_api_key="")
    resp = client.answer_query("What was Apple's revenue?", sample_chunks)

    assert resp.decision == "generate"
    assert resp.confidence >= 0.70
    assert len(resp.citations) == 1
    assert resp.citations[0].chunk_id == "AAPL_chunk_001"
    assert "AAPL_chunk_001" in resp.answer


def test_mock_llm_client_insufficient_evidence():
    """Client abstains when no chunks are provided."""
    client = LLMClient(groq_api_key="", gemini_api_key="")
    resp = client.answer_query("What was Apple's revenue in 2040?", [])

    assert resp.decision == "abstain"
    assert resp.confidence == 0.0
    assert "INSUFFICIENT_EVIDENCE" in resp.answer


def test_failover_from_groq_to_gemini(sample_chunks):
    """When Groq raises an error, client automatically fails over to Gemini."""
    client = LLMClient(groq_api_key="mock_groq_key", gemini_api_key="mock_gemini_key")

    # Mock Groq to fail with RateLimitError
    client._call_groq = MagicMock(side_effect=RateLimitError("Rate limit 429"))

    # Mock Gemini to succeed
    client._call_gemini = MagicMock(
        return_value="""{
            "answer": "Apple reported $383.3 billion revenue [Chunk: AAPL_chunk_001].",
            "citations": [
                {
                    "chunk_id": "AAPL_chunk_001",
                    "doc_id": "AAPL_10-K_2023",
                    "quote": "net sales of $383.3 billion in fiscal year 2023"
                }
            ],
            "has_sufficient_evidence": true
        }"""
    )

    parsed, provider, lat = client.generate_grounded_response("What was Apple's revenue?", sample_chunks)

    assert client._call_groq.called
    assert client._call_gemini.called
    assert provider == "gemini"
    assert "383.3 billion" in parsed["answer"]


def test_failover_to_mock_when_both_providers_fail(sample_chunks):
    """When both Groq and Gemini fail, client falls back to safe mock generator."""
    client = LLMClient(groq_api_key="mock_groq", gemini_api_key="mock_gemini")
    client._call_groq = MagicMock(side_effect=ProviderUnavailableError("Groq down"))
    client._call_gemini = MagicMock(side_effect=ProviderUnavailableError("Gemini down"))

    parsed, provider, lat = client.generate_grounded_response("What was Apple's revenue?", sample_chunks)

    assert provider == "mock"
    assert "AAPL_chunk_001" in parsed["answer"]


def test_answer_query_with_hallucinated_citations(sample_chunks):
    """If LLM outputs a hallucinated citation, confidence is degraded and bad citation is filtered."""
    client = LLMClient()
    # Mock LLM output with a fake chunk ID
    fake_json = {
        "answer": "Some answer [Chunk: non_existent_chunk].",
        "citations": [
            {
                "chunk_id": "non_existent_chunk",
                "doc_id": "fake_doc",
                "quote": "fake quote",
            }
        ],
        "has_sufficient_evidence": True,
    }
    client.generate_grounded_response = MagicMock(return_value=(fake_json, "mock", 10.0))

    resp = client.answer_query("Query", sample_chunks)
    assert resp.confidence < 1.0
    # Hallucinated citation should be removed from validated citations list
    assert len(resp.citations) == 0


def test_singleton_client():
    """get_llm_client and set_llm_client manage the singleton instance."""
    original = get_llm_client()
    assert original is not None

    custom = LLMClient(groq_model="custom_model")
    set_llm_client(custom)
    assert get_llm_client().groq_model == "custom_model"

    set_llm_client(original)
