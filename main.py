from __future__ import annotations

import sys
from typing import Optional, Protocol, runtime_checkable
from pathlib import Path

from src.filteredWarnings import suppress_noisy_warnings
# Apply warning filters before importing libraries that emit noisy warnings.
suppress_noisy_warnings()
from RealtimeSTT import AudioToTextRecorder  # pyright: ignore[reportMissingTypeStubs]  # noqa: E402

from src.config import Settings, get_settings  # noqa: E402
from src.llm_client import LLMClient, LLMConfig # noqa: E402
from src.session_logger import SessionLogger # noqa: E402
from src.tts_engine import TtsEngine # noqa: E402

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


def transcribe_loop(recorder: RecorderProtocol, llm_client: LLMClient, logger: SessionLogger, tts_engine: TtsEngine) -> None:
    """Sequential listen -> transcribe -> LLM -> TTS -> log loop."""
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
        if tts_engine.enabled:
            try:
                print("[SYSTEM] TTS speaking...")
                audio_path = logger.path().with_suffix(".wav")
                tts_engine.synthesize(llm_response, audio_path)
            except Exception as exc:
                print(f"[SYSTEM] TTS failed: {exc}", file=sys.stderr)


def main() -> None:
    suppress_noisy_warnings()
    settings = get_settings()
    try:
        llm_client = build_llm_client(settings)
        prompt_path = Path("PROMPT.md")
        if prompt_path.exists():
            llm_client.load_system_prompt(prompt_path)
        logger = SessionLogger(directory=settings.session_logs_dir)
        tts_engine = TtsEngine(settings)
        with build_recorder(settings) as recorder:
            transcribe_loop(recorder, llm_client, logger, tts_engine)
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as exc:  # pragma: no cover - runtime path
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
