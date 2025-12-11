from __future__ import annotations

import sys
from typing import Any, Mapping

from src.builders import (
    create_llm_client,
    create_memory_manager,
    create_recorder,
    create_session_logger,
    create_tts_engine,
    load_system_prompt,
)
from src.config import Settings
from src.types import RecorderProtocol
from src.llm_client import LLMClient
from src.memory_manager import MemoryManager
from src.session_logger import SessionLogger
from src.tts_engine import TtsEngine


def _log_usage(usage: Mapping[str, Any], settings: Settings) -> int | None:
    total_tokens = usage.get("total_tokens")
    print(f"[SYSTEM] LLM usage: {total_tokens or 'unknown'} total tokens.")
    return total_tokens if isinstance(total_tokens, int) else None


def _handle_limits(
    total_tokens: int | None,
    settings: Settings,
    memory_manager: MemoryManager,
    llm_client: LLMClient,
) -> None:
    if total_tokens is None:
        return
    warn_threshold = int(settings.context_window_tokens * 0.8)
    if total_tokens > warn_threshold:
        print(
            f"[SYSTEM] Approaching context limit ({total_tokens}/{settings.context_window_tokens}). "
            "Summarising to retain continuity..."
        )
        try:
            summary = memory_manager.summarise_history(llm_client, settings.summarise_prompt)
            print(f"[SYSTEM] Summary stored: {summary}")
        except Exception as exc:
            print(f"[SYSTEM] Summary failed: {exc}", file=sys.stderr)
    if total_tokens > settings.context_window_tokens:
        if memory_manager.window > 2:
            new_window = max(1, memory_manager.window // 2)
            memory_manager.shrink_window(new_window)
            print(
                f"[SYSTEM] Context window exceeded ({total_tokens} > {settings.context_window_tokens}); "
                f"reducing recent window to last {new_window} messages."
            )
        else:
            memory_manager.clear_history()
            print(
                f"[SYSTEM] Context window exceeded ({total_tokens} > {settings.context_window_tokens}); "
                "resetting conversation context."
            )


def transcribe_loop(
    recorder: RecorderProtocol,
    llm_client: LLMClient,
    logger: SessionLogger,
    tts_engine: TtsEngine,
    memory_manager: MemoryManager,
    settings: Settings,
) -> None:
    print("Initialising. Press Ctrl+C to quit.")
    print(f"Session log: {logger.path()}")
    while True:
        user_text = recorder.text()
        if not user_text:
            continue
        prompt = memory_manager.build_context_prompt(user_text)
        print(f"[YOU] {user_text}")
        print("[SYSTEM] Processing response...")
        try:
            llm_response = llm_client.complete(prompt)
        except Exception as exc:
            print(f"[SYSTEM] LLM call failed: {exc}", file=sys.stderr)
            continue
        print(f"[ASSISTANT] {llm_response}")
        logger.append_turn(user_text, llm_response)
        memory_manager.record_turn(user_text, llm_response)
        total_tokens = _log_usage(llm_client.last_usage or {}, settings)
        _handle_limits(total_tokens, settings, memory_manager, llm_client)
        if tts_engine.enabled:
            try:
                print("[SYSTEM] TTS speaking...")
                tts_engine.synthesize(llm_response, logger.path().with_suffix(".wav"))
            except Exception as exc:
                print(f"[SYSTEM] TTS failed: {exc}", file=sys.stderr)


def run_app(settings: Settings) -> None:
    llm_client = create_llm_client(settings)
    load_system_prompt(settings, llm_client)
    logger = create_session_logger(settings)
    tts_engine = create_tts_engine(settings)
    memory_manager = create_memory_manager(settings, session_id=logger.path().stem)
    with create_recorder(settings) as recorder:
        transcribe_loop(recorder, llm_client, logger, tts_engine, memory_manager, settings)
