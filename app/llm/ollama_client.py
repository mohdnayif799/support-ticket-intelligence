"""Ollama provider for fully offline operation.

Ollama compiles a JSON schema passed in `format` into a GBNF grammar and
constrains decoding with it, so this path gets the same hard schema guarantee as
Groq strict mode, with no API key and no network.

Default model is qwen3:4b: about 2.6GB, runs on a laptop without a GPU. Set
OLLAMA_MODEL=gpt-oss:20b to use the same weights as the hosted default, which
needs roughly 16GB of RAM.
"""

from __future__ import annotations

import logging

import httpx

from app.llm.base import LLMBadOutput, LLMUnavailable

logger = logging.getLogger(__name__)


class OllamaProvider:
    name = "ollama"
    schema_dialect = "plain"

    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3:4b",
        timeout: float = 60.0,
        temperature: float = 0.0,
    ):
        self.model = model
        self._host = host.rstrip("/")
        # Local generation on CPU is slower than hosted inference.
        self._timeout = max(timeout, 60.0)
        self._temperature = temperature

    def _post(self, payload: dict) -> dict:
        url = f"{self._host}/api/chat"
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(url, json=payload)
        except httpx.ConnectError as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self._host}. Start it with 'ollama serve' "
                f"and pull the model with 'ollama pull {self.model}'."
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMUnavailable(
                f"Ollama timed out after {self._timeout}s. A smaller model such as "
                "qwen3:4b will respond faster."
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"Ollama request failed: {exc}") from exc

        if resp.status_code == 404:
            raise LLMUnavailable(
                f"Ollama has no model '{self.model}'. Run: ollama pull {self.model}"
            )
        if resp.status_code >= 400:
            raise LLMUnavailable(f"Ollama returned {resp.status_code}: {resp.text[:300]}")

        try:
            return resp.json()
        except ValueError as exc:
            raise LLMBadOutput("Ollama returned a non-JSON body") from exc

    @staticmethod
    def _extract(data: dict) -> str:
        content = (data.get("message") or {}).get("content")
        if not content:
            raise LLMBadOutput(f"Unexpected Ollama response: {str(data)[:300]}")
        return content

    def complete_json(self, system: str, user: str, json_schema: dict) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "format": json_schema,
            "options": {"temperature": self._temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        return self._extract(self._post(payload))

    def complete_text(self, system: str, user: str, max_tokens: int = 300) -> str:
        payload = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": self._temperature, "num_predict": max_tokens},
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
            "host": self._host,
            "strict_mode": True,  # grammar-constrained decoding
            "reachable": False,
        }
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self._host}/api/tags")
            info["reachable"] = resp.status_code == 200
            if resp.status_code == 200:
                names = {m.get("name", "") for m in resp.json().get("models", [])}
                info["model_available"] = any(
                    n == self.model or n.split(":")[0] == self.model.split(":")[0]
                    for n in names
                )
        except httpx.HTTPError as exc:
            info["detail"] = str(exc)
        return info
