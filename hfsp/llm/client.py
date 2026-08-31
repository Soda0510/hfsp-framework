"""
LLM client abstraction over local open-source models.

Only a small JSON-in/JSON-out interface is needed, so the default backend is
Ollama's REST API via the standard library (no extra dependency).  Other
backends (vLLM / llama.cpp / Transformers) can be added by implementing
``chat_json``.
"""

import json
import urllib.request
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from .config import LLMConfig


class LLMClient(ABC):
    """Abstract client: returns text or a parsed JSON object for a chat request."""

    @abstractmethod
    def chat_json(self, system: str, user: str,
                  temperature: Optional[float] = None) -> Dict[str, Any]:
        """Send a chat request and return the parsed JSON reply."""

    @abstractmethod
    def chat_text(self, system: str, user: str,
                  temperature: Optional[float] = None) -> str:
        """Send a chat request and return the raw text reply."""


class OllamaClient(LLMClient):
    """Client for a local Ollama server (REST /api/chat)."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()

    def _chat(self, system: str, user: str,
              temperature: Optional[float] = None,
              json_mode: bool = False) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": (
                    temperature if temperature is not None
                    else self.config.temperature
                ),
            },
        }
        if json_mode:
            payload["format"] = "json"   # ask Ollama to constrain output to JSON
        url = f"{self.config.base_url}/api/chat"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            raw = resp.read().decode("utf-8")
        data = json.loads(raw)
        return data["message"]["content"]

    def chat_json(self, system: str, user: str,
                  temperature: Optional[float] = None) -> Dict[str, Any]:
        content = self._chat(system, user, temperature, json_mode=True)
        # Ollama with format=json still returns the JSON as a string.
        parsed = json.loads(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"LLM returned non-object JSON: {parsed!r}")
        return parsed

    def chat_text(self, system: str, user: str,
                  temperature: Optional[float] = None) -> str:
        return self._chat(system, user, temperature, json_mode=False)
