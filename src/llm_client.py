from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

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
        self.system_prompt: Optional[str] = None

    def load_system_prompt(self, path: Path) -> None:
        self.system_prompt = path.read_text(encoding="utf-8")

    def complete(self, prompt: str) -> str:
        messages: List[Dict[str, str]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
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
