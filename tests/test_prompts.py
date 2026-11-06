"""
Unit tests for grounded prompt formatting (Day 5).
"""

from app.schemas import SearchResult
from app.llm.prompts import build_rag_context, format_grounded_user_prompt, GROUNDED_SYSTEM_PROMPT


def test_grounded_system_prompt_rules():
    """System prompt must contain explicit instructions on grounding and citations."""
    assert "STRICT GROUNDING RULES" in GROUNDED_SYSTEM_PROMPT
    assert "INSUFFICIENT_EVIDENCE" in GROUNDED_SYSTEM_PROMPT
    assert "[Chunk: <chunk_id>]" in GROUNDED_SYSTEM_PROMPT


def test_build_rag_context_empty():
    """Empty chunk list returns informative message."""
    res = build_rag_context([])
    assert "No retrieved document chunks available" in res


def test_build_rag_context_with_chunks():
    """Context includes chunk ID, ticker, and text."""
    chunks = [
        SearchResult(
            chunk_id="AAPL_10-K_2023_chunk_0000",
            text="Apple reported revenue of $383 billion in fiscal year 2023.",
            score=0.9,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL", "doc_type": "10-K", "section": "Item 7"},
        ),
        SearchResult(
            chunk_id="AAPL_10-K_2023_chunk_0001",
            text="iPhone sales accounted for $200.5 billion.",
            score=0.8,
            metadata={"doc_id": "AAPL_10-K_2023", "ticker": "AAPL", "doc_type": "10-K"},
        ),
    ]

    context = build_rag_context(chunks)
    assert "[ID: AAPL_10-K_2023_chunk_0000]" in context
    assert "Section: Item 7" in context
    assert "Apple reported revenue of $383 billion" in context
    assert "[ID: AAPL_10-K_2023_chunk_0001]" in context
    assert "iPhone sales accounted for $200.5 billion" in context


def test_format_grounded_user_prompt():
    """User prompt wraps context and question together."""
    chunks = [
        SearchResult(
            chunk_id="chunk_1",
            text="Sample context text",
            score=0.5,
            metadata={"doc_id": "doc_1"},
        )
    ]
    prompt = format_grounded_user_prompt("What was the revenue?", chunks)
    assert "Context Chunks:" in prompt
    assert "User Question: What was the revenue?" in prompt
    assert "[ID: chunk_1]" in prompt
