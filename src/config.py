from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    rtstt_model: str
    rtstt_compute_type: str
    rtstt_language: str
    rtstt_use_microphone: bool
    llm_endpoint: str
    llm_model: str
    llm_timeout: float
    session_logs_dir: str


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_settings() -> Settings:
    # Load settings from environment with sensible defaults.
    rtstt_model = os.getenv("RTSTT_MODEL", "base.en").strip()
    rtstt_compute_type = os.getenv("RTSTT_COMPUTE_TYPE", "default").strip()
    rtstt_language = os.getenv("RTSTT_LANGUAGE", "").strip()
    rtstt_use_microphone = _bool_env("RTSTT_USE_MICROPHONE", True)
    llm_endpoint = os.getenv("LLM_ENDPOINT", "http://localhost:1234/v1/chat/completions").strip()
    llm_model = os.getenv("LLM_MODEL", "local-model").strip()
    llm_timeout = float(os.getenv("LLM_TIMEOUT", "60"))
    session_logs_dir = os.getenv("SESSION_LOGS_DIR", "session_logs").strip()

    return Settings(
        rtstt_model=rtstt_model,
        rtstt_compute_type=rtstt_compute_type,
        rtstt_language=rtstt_language,
        rtstt_use_microphone=rtstt_use_microphone,
        llm_endpoint=llm_endpoint,
        llm_model=llm_model,
        llm_timeout=llm_timeout,
        session_logs_dir=session_logs_dir,
    )
