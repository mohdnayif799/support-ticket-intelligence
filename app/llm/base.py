"""LLM provider abstraction.

Providers differ only in transport and in how a JSON schema is attached to the
request. Groq takes `response_format.json_schema`; Ollama takes `format`. Both
constrain decoding to the schema, so the same prompt and the same QueryPlan
schema work unchanged on either, hosted or fully offline.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


class LLMError(RuntimeError):
    """Base class for provider failures."""


class LLMUnavailable(LLMError):
    """Provider unreachable, unauthenticated, or rate limited."""


class LLMBadOutput(LLMError):
    """Provider responded but the payload was not usable."""


@runtime_checkable
class LLMProvider(Protocol):
    """Minimal interface the pipeline depends on."""

    name: str
    model: str
    #: Which JSON-schema dialect this backend needs: "strict" (no $ref, no
    #: anyOf, closed objects) or "plain" (Pydantic's own output).
    schema_dialect: str

    def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        """Return a JSON string conforming to json_schema."""
        ...

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> str:
        """Return free-form text. Used only for phrasing answers."""
        ...

    def health(self) -> dict:
        """Report reachability without raising."""
        ...
