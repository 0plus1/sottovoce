from __future__ import annotations

import sys
import warnings
from typing import Optional, Protocol, runtime_checkable

from RealtimeSTT import AudioToTextRecorder  # pyright: ignore[reportMissingTypeStubs]

from src.config import Settings, get_settings
from src.llm_client import LLMClient, LLMConfig
from src.session_logger import SessionLogger


@runtime_checkable
class RecorderProtocol(Protocol):
    """Minimal protocol we rely on from RealtimeSTT.AudioToTextRecorder."""

    def text(self, on_transcription_finished: Optional[object] = None) -> Optional[str]: ...

    def __enter__(self) -> "RecorderProtocol": ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None: ...


def build_recorder(settings: Settings) -> RecorderProtocol:
    """Create a recorder configured from Settings."""
    recorder: RecorderProtocol = AudioToTextRecorder(
        model=settings.rtstt_model,
        compute_type=settings.rtstt_compute_type,
        language=settings.rtstt_language,
        use_microphone=settings.rtstt_use_microphone,
        no_log_file=True
    )
    return recorder


def build_llm_client(settings: Settings) -> LLMClient:
    """Instantiate the LLM client configured for LM Studio (OpenAI-compatible)."""
    cfg = LLMConfig(
        endpoint=settings.llm_endpoint,
        model=settings.llm_model,
        timeout=settings.llm_timeout,
    )
    return LLMClient(cfg)


def transcribe_loop(recorder: RecorderProtocol, llm_client: LLMClient, logger: SessionLogger) -> None:
    """Sequential listen -> transcribe -> LLM -> log loop."""
    print("Initialising. Press Ctrl+C to quit.")
    print(f"Session log: {logger.path()}")
    while True:
        user_text = recorder.text()
        if not user_text:
            continue
        print(f"[YOU] {user_text}")
        print("[SYSTEM] Processing response...")
        try:
            llm_response = llm_client.complete(user_text)
        except Exception as exc:
            print(f"[SYSTEM] LLM call failed: {exc}", file=sys.stderr)
            continue
        print(f"[ASSISTANT] {llm_response}")
        logger.append_turn(user_text, llm_response)


def main() -> None:
    settings = get_settings()
    # Hide noisy RuntimeWarnings from faster_whisper feature extraction.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module="faster_whisper.feature_extractor",
    )
    try:
        llm_client = build_llm_client(settings)
        logger = SessionLogger(directory=settings.session_logs_dir)
        with build_recorder(settings) as recorder:
            transcribe_loop(recorder, llm_client, logger)
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as exc:  # pragma: no cover - runtime path
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
