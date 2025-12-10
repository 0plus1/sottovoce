from __future__ import annotations

import logging
import sys
import warnings
from typing import Optional, Protocol, runtime_checkable

from RealtimeSTT import AudioToTextRecorder # pyright: ignore[reportMissingTypeStubs]

from src.config import Settings, get_settings


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
        model=settings.model,
        compute_type=settings.compute_type,
        language=settings.language,
        use_microphone=settings.use_microphone,
        level=settings.log_level,
        no_log_file=settings.no_log_file,
    )
    return recorder


def transcribe_loop(recorder: RecorderProtocol) -> None:
    # Sequential listen -> transcribe loop. Blocking and linear by design.
    print("Initialising. Press Ctrl+C to quit.")
    while True:
        text = recorder.text()
        if text:
            print(text)


def main() -> None:
    settings = get_settings()
    # Hide noisy RuntimeWarnings from faster_whisper feature extraction.
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        module="faster_whisper.feature_extractor",
    )
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    try:
        with build_recorder(settings) as recorder:
            transcribe_loop(recorder)
    except KeyboardInterrupt:
        print("\nExiting.")
    except Exception as exc:  # pragma: no cover - runtime path
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
