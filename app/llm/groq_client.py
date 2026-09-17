"""Groq provider.

Uses Structured Outputs. With strict mode the model is constrained at the token
level, so the response is always schema-valid JSON and the usual "model returned
prose instead of JSON" failure disappears.

Strict mode is only available on some models. As of 2026-09 that is
openai/gpt-oss-20b, openai/gpt-oss-120b and qwen/qwen3.8-27b. Note that
llama-3.3-70b-versatile and llama-3.1-8b-instant were deprecated on 2026-06-17.

Structured Outputs cannot be combined with tool calling or streaming, which is
why this sends a plain chat completion with response_format rather than
declaring the plan as a function.

Raw httpx is used rather than the vendor SDK: the call is one POST, and the same
client serves the Ollama provider, keeping the dependency list short.
"""

from __future__ import annotations

import logging

import httpx

from app.llm.base import LLMBadOutput, LLMUnavailable

logger = logging.getLogger(__name__)

API_URL = "https://api.groq.com/openai/v1/chat/completions"
MODELS_URL = "https://api.groq.com/openai/v1/models"

#: Models known to support strict constrained decoding.
STRICT_CAPABLE = frozenset(
    {"openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"}
)

#: Retired models, checked so the failure message is useful.
DEPRECATED = frozenset({"llama-3.3-70b-versatile", "llama-3.1-8b-instant", "qwen/qwen3-32b"})


class GroqProvider:
    name = "groq"
    schema_dialect = "strict"

    def __init__(
        self,
        api_key: str,
        model: str,
        timeout: float = 30.0,
        temperature: float = 0.0,
        strict: bool = True,
    ):
        if not api_key:
            raise LLMUnavailable(
                "GROQ_API_KEY is not set. Create a free key at "
                "https://console.groq.com/keys (no card required), or set "
                "LLM_PROVIDER=ollama to run locally."
            )
        if model in DEPRECATED:
            raise LLMUnavailable(
                f"Model '{model}' was deprecated by Groq and no longer serves "
                f"requests. Use one of: {', '.join(sorted(STRICT_CAPABLE))}."
            )
        self.model = model
        self._key = api_key
        self._timeout = timeout
        self._temperature = temperature
        self._strict = strict and model in STRICT_CAPABLE

        if strict and not self._strict:
            logger.warning(
                "Model %s is not known to support strict mode; falling back to "
                "best-effort schema adherence with validation retries.",
                model,
            )

    @property
    def strict_enabled(self) -> bool:
        return self._strict

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._key}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict) -> dict:
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(API_URL, headers=self._headers(), json=payload)
        except httpx.TimeoutException as exc:
            raise LLMUnavailable(f"Groq request timed out after {self._timeout}s") from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Could not reach Groq: {exc}") from exc

        if resp.status_code == 401:
            raise LLMUnavailable("Groq rejected the API key (401).")
        if resp.status_code == 429:
            retry = resp.headers.get("retry-after", "unknown")
            raise LLMUnavailable(
                f"Groq rate limit reached. Free tier allows 30 requests/minute. "
                f"Retry after {retry}s."
            )
        if resp.status_code == 404:
            raise LLMUnavailable(
                f"Groq does not serve model '{self.model}'. It may have been "
                f"deprecated. Try openai/gpt-oss-20b."
            )
        if resp.status_code >= 400:
            body = resp.text[:400]
            # A json_validate 400 means the schema was accepted and the model's
            # output failed it. That is a bad generation, not a dead provider,
            # so it belongs in the repair loop rather than surfacing as 503.
            if "json_validate" in body:
                raise LLMBadOutput(f"Generated JSON failed the schema: {body}")
            raise LLMUnavailable(f"Groq returned {resp.status_code}: {body}")

        try:
            return resp.json()
        except ValueError as exc:
            raise LLMBadOutput("Groq returned a non-JSON body") from exc

    @staticmethod
    def _extract(data: dict) -> str:
        try:
            choice = data["choices"][0]
        except (KeyError, IndexError) as exc:
            raise LLMBadOutput(f"Unexpected Groq response shape: {str(data)[:300]}") from exc

        message = choice.get("message", {})
        if message.get("refusal"):
            raise LLMBadOutput(f"Model refused: {message['refusal']}")
        content = message.get("content")
        if not content:
            raise LLMBadOutput("Groq returned an empty completion")
        return content

    def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        payload = {
            "model": self.model,
            "temperature": self._temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "query_plan",
                    "strict": self._strict,
                    "schema": json_schema,
                },
            },
        }
        return self._extract(self._post(payload))

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "temperature": self._temperature,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        return self._extract(self._post(payload)).strip()

    def health(self) -> dict:
        info = {
            "provider": self.name,
            "model": self.model,
            "strict_mode": self._strict,
            "reachable": False,
        }
        try:
            with httpx.Client(timeout=min(self._timeout, 10.0)) as client:
                resp = client.get(MODELS_URL, headers=self._headers())
            info["reachable"] = resp.status_code == 200
            if resp.status_code == 200:
                ids = {m.get("id") for m in resp.json().get("data", [])}
                info["model_available"] = self.model in ids
            else:
                info["detail"] = f"HTTP {resp.status_code}"
        except httpx.HTTPError as exc:
            info["detail"] = str(exc)
        return info
