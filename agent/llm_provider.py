"""Model-agnostic LLM interface for the Mode 3 agent.

Add a new provider by subclassing LLMProvider and implementing `complete`.
Set the default in agent.py via LLMProvider.default().
"""

from __future__ import annotations
import json
import os
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        """Return a completion string for the given system + user prompt."""

    @staticmethod
    def default() -> "LLMProvider":
        """Return the best available provider based on env vars."""
        if os.environ.get("ANTHROPIC_API_KEY"):
            return ClaudeProvider()
        return OllamaProvider()


# ── Ollama (local) ────────────────────────────────────────────────────────────

class OllamaProvider(LLMProvider):
    """Talks to a locally running Ollama instance (default: localhost:11434)."""

    def __init__(self, model: str = "qwen2.5:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        import urllib.request
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
        if json_mode:
            payload["format"] = "json"

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
        return result["message"]["content"].strip()


# ── Claude (Anthropic API) ────────────────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    """Uses the Anthropic Claude API. Set ANTHROPIC_API_KEY env var."""

    def __init__(self, model: str = "claude-opus-4-8", api_key: str = None):
        self.model = model
        self.api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def complete(self, system: str, user: str, json_mode: bool = False) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return message.content[0].text.strip()
