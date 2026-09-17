"""Provider selection from configuration."""

from __future__ import annotations

import logging

from app.config import Settings
from app.llm.base import LLMProvider, LLMUnavailable
from app.llm.groq_client import GroqProvider
from app.llm.ollama_client import OllamaProvider
from app.llm.stub import StubProvider

logger = logging.getLogger(__name__)


def build_provider(settings: Settings) -> LLMProvider:
    """Instantiate the configured provider.

    Raises:
        LLMUnavailable: The provider cannot be constructed, for example a missing
            Groq key. Construction failures surface at startup rather than on the
            first query.
    """
    provider = settings.llm_provider

    if provider == "groq":
        p = GroqProvider(
            api_key=settings.groq_api_key or "",
            model=settings.groq_model,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
            strict=settings.groq_strict_schema,
        )
        logger.info("LLM provider: groq model=%s strict=%s", p.model, p.strict_enabled)
        return p

    if provider == "ollama":
        p = OllamaProvider(
            host=settings.ollama_host,
            model=settings.ollama_model,
            timeout=settings.llm_timeout_seconds,
            temperature=settings.llm_temperature,
        )
        logger.info("LLM provider: ollama model=%s host=%s", p.model, settings.ollama_host)
        return p

    if provider == "stub":
        logger.warning("LLM provider: stub (test double, not for evaluation)")
        return StubProvider()

    raise LLMUnavailable(f"Unknown LLM_PROVIDER '{provider}'")
