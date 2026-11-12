"""
LangSmith tracing setup for the Financial Document Intelligence Engine.

WHY LangSmith:
- Full observability into LangGraph state machine execution: every node transition,
  query rewrite attempt, tool invocation, and LLM prompt/response pair is traced.
- Zero-code runtime overhead when disabled: LangChain/LangGraph check environment
  variables natively without wrapping state machine calls in intrusive middleware.
- Seamless debugging: enables visual inspection of agentic loops (retrieve -> generate -> rewrite)
  in the LangSmith dashboard during complex multi-hop queries.
"""

import os
import logging
from app.config import settings

logger = logging.getLogger(__name__)


def setup_langsmith() -> bool:
    """
    Configure LangSmith tracing environment variables if enabled in settings.

    Returns:
        bool: True if LangSmith tracing was successfully configured, False otherwise.
    """
    if not settings.langsmith_tracing_enabled:
        logger.info("LangSmith tracing is disabled (LANGSMITH_TRACING_ENABLED=False)")
        return False

    if not settings.langsmith_api_key:
        logger.warning(
            "LangSmith tracing enabled but LANGSMITH_API_KEY is empty. Tracing will not be active."
        )
        return False

    # LangChain / LangGraph read these standard environment variables
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"

    logger.info(
        "LangSmith tracing configured successfully for project: %s",
        settings.langsmith_project,
    )
    return True

