from __future__ import annotations

from pathlib import Path

from src.config import Settings
from src.llm_client import LLMClient, LLMConfig
from src.memory_manager import MemoryManager
from src.session_logger import SessionLogger
from src.tts_engine import TtsEngine
from src.types import RecorderProtocol


def create_recorder(settings: Settings) -> RecorderProtocol:
    # Import inside to allow warnings filter to be applied beforehand.
    from RealtimeSTT import AudioToTextRecorder  # pyright: ignore[reportMissingTypeStubs]

    recorder: RecorderProtocol = AudioToTextRecorder(
        model=settings.rtstt_model,
        compute_type=settings.rtstt_compute_type,
        language=settings.rtstt_language,
        use_microphone=settings.rtstt_use_microphone,
        no_log_file=True,
    )
    return recorder


def create_llm_client(settings: Settings) -> LLMClient:
    cfg = LLMConfig(
        endpoint=settings.llm_endpoint,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
    return LLMClient(cfg)


def create_session_logger(settings: Settings) -> SessionLogger:
    return SessionLogger(directory=settings.session_logs_dir)


def create_memory_manager(settings: Settings, session_id: str) -> MemoryManager:
    return MemoryManager(settings, session_id=session_id)


def create_tts_engine(settings: Settings) -> TtsEngine:
    return TtsEngine(settings)


def load_system_prompt(settings: Settings, llm_client: LLMClient) -> None:
    prompt_path = Path("PROMPT.md")
    system_prompt_parts = []
    if prompt_path.exists():
        llm_client.load_system_prompt(prompt_path)
        system_prompt_parts.append(llm_client.system_prompt or "")
    system_prompt_parts.append(settings.llm_prompt_conversational)
    combined_prompt = "\n\n".join([p for p in system_prompt_parts if p.strip()])
    if combined_prompt:
        llm_client.system_prompt = combined_prompt
