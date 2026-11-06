# LLM client subpackage — Groq + Gemini with fallback (Day 5)

from app.llm.client import (
    LLMClient,
    get_llm_client,
    set_llm_client,
    RateLimitError,
    ProviderUnavailableError,
)
from app.llm.prompts import (
    GROUNDED_SYSTEM_PROMPT,
    build_rag_context,
    format_grounded_user_prompt,
)
from app.llm.citation_validator import (
    validate_citations,
    CitationValidationResult,
    ValidatedCitation,
)

__all__ = [
    "LLMClient",
    "get_llm_client",
    "set_llm_client",
    "RateLimitError",
    "ProviderUnavailableError",
    "GROUNDED_SYSTEM_PROMPT",
    "build_rag_context",
    "format_grounded_user_prompt",
    "validate_citations",
    "CitationValidationResult",
    "ValidatedCitation",
]
