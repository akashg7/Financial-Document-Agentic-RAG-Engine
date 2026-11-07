"""
Multi-provider LLM client with Groq primary, Gemini fallback, and exponential backoff.

WHY multi-provider fallback (Day 5):
- Free tiers have strict rate limits: Groq (30 RPM) and Gemini (15 RPM).
- If a burst of financial queries exhausts Groq's RPM quota (HTTP 429), the engine
  automatically switches to Google Gemini without dropping the request.
- Using two completely different cloud infrastructures eliminates single-point-of-failure
  risk for zero cost.

WHY tenacity retry with exponential backoff:
- Rate limits and transient 5xx server hiccups on free tiers are usually temporary.
- Exponential backoff (1s, 2s, 4s) with jitter gives the API time to replenish tokens
  before failing over.
"""

import json
import logging
import re
import time
from typing import Optional, Sequence, Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

from app.config import settings
from app.schemas import AnswerResponse, Citation, SearchResult
from app.llm.prompts import GROUNDED_SYSTEM_PROMPT, format_grounded_user_prompt
from app.llm.citation_validator import validate_citations, CitationValidationResult

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Base exception for LLM provider errors."""
    pass


class RateLimitError(LLMError):
    """Exception raised when an LLM provider returns HTTP 429 rate limit."""
    pass


class AuthenticationError(LLMError):
    """Exception raised when an LLM provider credentials are invalid (401/403)."""
    pass


class ProviderUnavailableError(LLMError):
    """Exception raised when an LLM provider is down or unreachable."""
    pass


def _clean_json_text(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM response text."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


class LLMClient:
    """
    Unified LLM client orchestrating Groq and Google Gemini with automatic failover.
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        groq_model: Optional[str] = None,
        gemini_model: Optional[str] = None,
    ):
        self.groq_api_key = groq_api_key if groq_api_key is not None else settings.groq_api_key
        self.gemini_api_key = gemini_api_key if gemini_api_key is not None else settings.gemini_api_key
        self.groq_model = groq_model if groq_model is not None else settings.groq_model_name
        self.gemini_model = gemini_model if gemini_model is not None else settings.gemini_model_name

        self._groq_client = None
        self._gemini_client = None

    def _get_groq_client(self):
        """Lazily initialize Groq client."""
        if self._groq_client is None:
            if not self.groq_api_key:
                raise ProviderUnavailableError("Groq API key not configured")
            from groq import Groq
            self._groq_client = Groq(api_key=self.groq_api_key)
        return self._groq_client

    def _get_gemini_client(self):
        """Lazily initialize Google Gemini client."""
        if self._gemini_client is None:
            if not self.gemini_api_key:
                raise ProviderUnavailableError("Gemini API key not configured")
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                import google.generativeai as genai
            genai.configure(api_key=self.gemini_api_key)
            self._gemini_client = genai.GenerativeModel(
                model_name=self.gemini_model,
                system_instruction=GROUNDED_SYSTEM_PROMPT,
            )
        return self._gemini_client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((RateLimitError, ProviderUnavailableError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_groq(self, prompt: str) -> str:
        """Call Groq API with retry and backoff."""
        client = self._get_groq_client()
        try:
            logger.info("Calling primary LLM: Groq (%s)", self.groq_model)
            response = client.chat.completions.create(
                model=self.groq_model,
                messages=[
                    {"role": "system", "content": GROUNDED_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,  # Zero temperature for deterministic factual extraction
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err_str = str(e).lower()
            if "invalid api key" in err_str or "invalid_api_key" in err_str or "401" in err_str or "unauthorized" in err_str or "404" in err_str or "model_not_found" in err_str:
                logger.warning("Groq permanent error (invalid key or model not found). Skipping retries.")
                raise AuthenticationError(f"Groq permanent error: {e}") from e
            if "rate limit" in err_str or "429" in err_str:
                logger.warning("Groq rate limit exceeded: %s", e)
                raise RateLimitError(f"Groq 429: {e}") from e
            logger.error("Groq invocation error: %s", e)
            raise ProviderUnavailableError(f"Groq failed: {e}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=4),
        retry=retry_if_exception_type((RateLimitError, ProviderUnavailableError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API with retry and backoff."""
        model = self._get_gemini_client()
        try:
            logger.info("Calling fallback LLM: Gemini (%s)", self.gemini_model)
            response = model.generate_content(
                prompt,
                generation_config={"temperature": 0.0, "response_mime_type": "application/json"},
            )
            return response.text or ""
        except Exception as e:
            err_str = str(e).lower()
            if "api key not valid" in err_str or "api_key_invalid" in err_str or "400" in err_str and "api key" in err_str or "403" in err_str or "404" in err_str or "not found" in err_str:
                logger.warning("Gemini permanent error (invalid key or model not found). Skipping retries.")
                raise AuthenticationError(f"Gemini permanent error: {e}") from e
            if "rate limit" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                logger.warning("Gemini rate limit exceeded: %s", e)
                raise RateLimitError(f"Gemini 429: {e}") from e
            logger.error("Gemini invocation error: %s", e)
            raise ProviderUnavailableError(f"Gemini failed: {e}") from e

    def _call_mock(self, query: str, chunks: Sequence[SearchResult]) -> str:
        """
        Deterministic mock response generator used when no API keys are provided
        or for offline unit testing.
        """
        logger.info("Using mock grounded LLM generator")
        if not chunks:
            return json.dumps({
                "answer": "INSUFFICIENT_EVIDENCE: No document chunks were retrieved for this query.",
                "citations": [],
                "has_sufficient_evidence": False,
            })

        # Check for unanswerable signals in mock mode
        q_lower = query.lower()
        combined_text = " ".join(c.text.lower() for c in chunks)
        # If query asks about distant future or projections not in chunk
        for year in ["2030", "2035", "2040", "2050"]:
            if year in q_lower and year not in combined_text:
                return json.dumps({
                    "answer": f"INSUFFICIENT_EVIDENCE: The provided filings do not contain projections for year {year}.",
                    "citations": [],
                    "has_sufficient_evidence": False,
                })

        # Synthesize a grounded mock answer using the first chunk
        top_chunk = chunks[0]
        quote_sample = top_chunk.text[:80].strip()
        doc_id = top_chunk.metadata.get("doc_id", "doc_unknown")

        return json.dumps({
            "answer": f"Based on {doc_id}, {quote_sample}... [Chunk: {top_chunk.chunk_id}].",
            "citations": [
                {
                    "chunk_id": top_chunk.chunk_id,
                    "doc_id": doc_id,
                    "quote": quote_sample,
                }
            ],
            "has_sufficient_evidence": True,
        })

    def generate_grounded_response(
        self,
        query: str,
        context_chunks: Sequence[SearchResult],
    ) -> tuple[dict[str, Any], str, float]:
        """
        Generate grounded response by querying primary LLM (Groq) with automatic
        failover to fallback (Gemini) or mock.

        Returns:
            Tuple of (parsed_json_dict, provider_name, latency_ms)
        """
        prompt = format_grounded_user_prompt(query, context_chunks)
        start_time = time.time()
        provider_used = "mock"
        raw_text = ""

        # Attempt 1: Primary provider (Groq)
        if self.groq_api_key:
            try:
                raw_text = self._call_groq(prompt)
                provider_used = "groq"
            except Exception as groq_err:
                logger.warning("Primary provider (Groq) failed: %s. Initiating fallback...", groq_err)

        # Attempt 2: Fallback provider (Gemini) if Groq was not used or failed
        if not raw_text and self.gemini_api_key:
            try:
                raw_text = self._call_gemini(prompt)
                provider_used = "gemini"
            except Exception as gemini_err:
                logger.warning("Fallback provider (Gemini) failed: %s.", gemini_err)

        # Attempt 3: Safe offline mock fallback if both failed or no keys configured
        if not raw_text:
            raw_text = self._call_mock(query, context_chunks)
            provider_used = "mock"

        latency_ms = (time.time() - start_time) * 1000

        # Parse JSON
        cleaned = _clean_json_text(raw_text)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as err:
            logger.error("Failed to parse LLM output as JSON: %s\nRaw output: %s", err, raw_text)
            parsed = {
                "answer": raw_text,
                "citations": [],
                "has_sufficient_evidence": False,
            }

        return parsed, provider_used, latency_ms

    def answer_query(
        self,
        query: str,
        context_chunks: Sequence[SearchResult],
    ) -> AnswerResponse:
        """
        Full grounded Q&A pipeline:
        1. Formats grounded prompt with retrieved chunks.
        2. Calls LLM with Groq -> Gemini failover.
        3. Validates citations and quotes against source chunks.
        4. Calculates confidence score based on citation fidelity and evidence sufficiency.
        5. Returns structured AnswerResponse.
        """
        t0 = time.time()
        parsed, provider_used, llm_latency = self.generate_grounded_response(query, context_chunks)

        raw_citations = []
        for c in parsed.get("citations", []):
            try:
                raw_citations.append(
                    Citation(
                        chunk_id=c.get("chunk_id", ""),
                        doc_id=c.get("doc_id", ""),
                        quote=c.get("quote", ""),
                    )
                )
            except Exception as e:
                logger.warning("Skipping malformed citation: %s (%s)", c, e)

        # Validate citations
        val_result: CitationValidationResult = validate_citations(raw_citations, context_chunks)

        answer_text = parsed.get("answer", "")
        has_sufficient_evidence = parsed.get("has_sufficient_evidence", True)

        # Day 6: Multi-signal calibrated confidence scoring
        from app.confidence.scorer import score_confidence

        confidence_breakdown = score_confidence(
            query=query,
            answer=answer_text,
            retrieved_chunks=context_chunks,
            validation_result=val_result,
        )

        decision = confidence_breakdown.decision
        confidence = confidence_breakdown.overall_confidence

        if not has_sufficient_evidence:
            decision = "abstain"
            confidence = 0.0

        # Filter out hallucinated citations from final response
        valid_citations = [
            vc.citation for vc in val_result.validated_citations if vc.is_valid_chunk_id
        ]

        total_latency = (time.time() - t0) * 1000

        return AnswerResponse(
            query=query,
            answer=answer_text,
            citations=valid_citations,
            confidence=confidence,
            decision=decision,
            retries=0,
            latency_ms=round(total_latency, 2),
            cached=False,
        )


# Module-level singleton
_default_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create singleton LLMClient."""
    global _default_llm_client
    if _default_llm_client is None:
        _default_llm_client = LLMClient()
    return _default_llm_client


def set_llm_client(client: Optional[LLMClient]) -> None:
    """Set or reset singleton LLMClient. Useful for testing."""
    global _default_llm_client
    _default_llm_client = client
