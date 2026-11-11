"""
Rate limiter configuration for the Financial Document Intelligence Engine.

WHY SlowAPI with IP-based rate limiting:
- Protects downstream vector store (Qdrant) and LLM APIs (Groq/Gemini) from denial-of-service,
  runaway retry loops, and unexpected API cost spikes.
- Central limiter instance: Decouples limiter configuration from route definitions and app creation,
  avoiding circular import dependencies.
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

from app.config import settings

limiter = Limiter(
    key_func=get_remote_address,
    enabled=settings.rate_limit_enabled,
    default_limits=[settings.rate_limit_default],
)
