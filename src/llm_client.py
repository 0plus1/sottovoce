from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import requests


@dataclass
class LLMConfig:
    endpoint: str
    model: str
    timeout: float = 60.0


class LLMClient:
    # Minimal OpenAI-compatible chat client.

    def __init__(self, config: LLMConfig):
        self.config = config

    def complete(self, prompt: str) -> str:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        response = requests.post(
            self.config.endpoint,
            json=payload,
            timeout=self.config.timeout,
        )
        response.raise_for_status()
        data = response.json()
        # Expected OpenAI-compatible structure
        choices = data.get("choices", [])
        if not choices:
            raise ValueError("LLM response missing choices")
        message = choices[0].get("message", {})
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("LLM response content missing or invalid")
        return content
