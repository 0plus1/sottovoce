from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    model: str
    compute_type: str
    language: str
    use_microphone: bool
    log_level: int
    no_log_file: bool


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _log_level_env(name: str, default: str) -> int:
    raw = os.getenv(name, default).strip().upper()
    level = getattr(logging, raw, None)
    if isinstance(level, int):
        return level
    return getattr(logging, default.upper(), logging.WARNING)


def get_settings() -> Settings:
    # Load settings from environment with sensible defaults.
    model = os.getenv("RTSTT_MODEL", "base.en").strip()
    compute_type = os.getenv("RTSTT_COMPUTE_TYPE", "default").strip()
    language = os.getenv("RTSTT_LANGUAGE", "").strip()
    use_microphone = _bool_env("RTSTT_USE_MICROPHONE", True)
    log_level = _log_level_env("RTSTT_LOG_LEVEL", "WARNING")
    no_log_file = os.getenv("RTSTT_LOG_LEVEL", "").strip().upper() == "NONE"
    return Settings(
        model=model,
        compute_type=compute_type,
        language=language,
        use_microphone=use_microphone,
        log_level=log_level,
        no_log_file=no_log_file,
    )
