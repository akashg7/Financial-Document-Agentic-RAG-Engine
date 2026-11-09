"""
Query reformulation module for the Financial Agentic RAG loop.

When retrieval confidence is ambiguous (0.40 <= confidence < 0.70), the agent
rewrites the query before attempting another retrieval cycle.

WHY query rewriting in financial RAG:
- User queries are frequently conversational ("how much money did Apple make last year?").
- SEC filings (Form 10-K, 10-Q) and earnings calls use strict GAAP and regulatory terminology:
  "Consolidated Statements of Operations", "Net Sales", "Total Net Revenues", "Operating Income".
- A query that fails on colloquial phrasing can succeed when translated to filing vocabulary,
  item headers (Item 7 MD&A, Item 8 Financial Statements), and specific ticker/period tokens.
"""

import re
import json
import logging
from typing import Optional, Any

from app.llm.prompts import FINANCIAL_REWRITE_SYSTEM_PROMPT, format_rewrite_prompt

logger = logging.getLogger(__name__)

# Common conversational filler phrases that dilute BM25 and dense retrieval
CONVERSATIONAL_FILLERS = [
    r"\bcan you (please )?tell me\b",
    r"\bplease (tell me|find|show me|explain)\b",
    r"\bwhat (was|were|is|are)\b",
    r"\bhow much (did|does|was|were)\b",
    r"\bhow (did|does|do)\b",
    r"\btell me (about)?\b",
    r"\bi would like to know\b",
    r"\bgive me the\b",
]

# Domain-specific financial synonym expansions
FINANCIAL_SYNONYMS = {
    r"\b(sales|revenue|revenues|top line)\b": "net sales total revenue consolidated operations",
    r"\b(profit|profits|net income|bottom line)\b": "operating income net income net earnings",
    r"\b(debt|borrowings|loans)\b": "total debt liabilities commercial paper term notes credit facility",
    r"\b(r&d|research and development)\b": "research and development expense",
    r"\b(capex|capital spending|investments)\b": "capital expenditures property plant equipment payments",
    r"\b(risk|risks|hazards)\b": "Item 1A risk factors operational risks uncertainties",
    r"\b(margin|margins)\b": "gross margin operating margin percentage",
}


def deterministic_financial_rewrite(
    query: str,
    retry_count: int = 0,
    ticker: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> str:
    """
    Offline/deterministic query expansion for SEC filings.

    Applies:
    1. Conversational filler stripping.
    2. Ticker symbol and document type injection if missing.
    3. GAAP financial terminology expansion.
    4. Progressive deepening based on retry iteration (Item 7 / Item 8 headers).
    """
    cleaned = query.strip()

    # 1. Remove conversational fillers
    for pattern in CONVERSATIONAL_FILLERS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # Clean punctuation
    cleaned = re.sub(r"[?!,;:\"']", " ", cleaned)
    cleaned = " ".join(cleaned.split())

    # 2. Expand financial synonyms
    expanded_parts = [cleaned]
    for pattern, expansion in FINANCIAL_SYNONYMS.items():
        if re.search(pattern, cleaned, flags=re.IGNORECASE):
            expanded_parts.append(expansion)
            break  # Expand primary financial topic to avoid query bloat

    # 3. Inject ticker if provided and absent from query
    if ticker and ticker.lower() not in cleaned.lower():
        expanded_parts.insert(0, ticker.upper())

    # 4. Inject document type if provided and absent
    if doc_type and doc_type.lower() not in cleaned.lower():
        expanded_parts.append(doc_type.upper())

    # 5. Progressive deepening on subsequent retries
    if retry_count == 1:
        expanded_parts.append("Item 7 Management Discussion Analysis Item 8 Financial Statements")
    elif retry_count >= 2:
        expanded_parts.append("Consolidated Statements of Operations Balance Sheets Cash Flows Notes")

    # Combine and deduplicate terms while preserving order
    combined = " ".join(expanded_parts)
    tokens = combined.split()
    seen = set()
    deduped_tokens = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower not in seen:
            seen.add(token_lower)
            deduped_tokens.append(token)

    rewritten = " ".join(deduped_tokens)
    return rewritten if rewritten else query


def rewrite_query(
    current_query: str,
    original_query: Optional[str] = None,
    retry_count: int = 0,
    ticker_filter: Optional[str] = None,
    doc_type_filter: Optional[str] = None,
    llm_client: Optional[Any] = None,
) -> str:
    """
    Rewrite a query to improve retrieval recall and precision.

    Attempts LLM-based rewrite if a configured client is available;
    falls back cleanly to deterministic SEC-filing keyword expansion.
    """
    base_query = current_query or original_query or ""
    if not base_query.strip():
        return base_query

    # Attempt LLM rewrite if client has active external API credentials
    if llm_client is not None and getattr(llm_client, "groq_api_key", None):
        try:
            prompt = format_rewrite_prompt(
                query=base_query,
                ticker=ticker_filter or "",
                doc_type=doc_type_filter or "",
            )
            raw = llm_client._call_groq(prompt)
            # Parse JSON
            from app.llm.client import _clean_json_text
            parsed = json.loads(_clean_json_text(raw))
            rewritten = parsed.get("rewritten_query", "").strip()
            if rewritten and rewritten.lower() != base_query.lower():
                logger.info("LLM rewritten query (retry %d): '%s' -> '%s'", retry_count, base_query, rewritten)
                return rewritten
        except Exception as err:
            logger.warning("LLM query rewrite failed (%s). Falling back to deterministic rewrite.", err)

    # Deterministic fallback
    rewritten = deterministic_financial_rewrite(
        query=base_query,
        retry_count=retry_count,
        ticker=ticker_filter,
        doc_type=doc_type_filter,
    )
    logger.info("Deterministic rewritten query (retry %d): '%s' -> '%s'", retry_count, base_query, rewritten)
    return rewritten

