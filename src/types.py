from __future__ import annotations

from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class RecorderProtocol(Protocol):
    """Minimal protocol expected from the RealtimeSTT recorder."""

    def text(self, on_transcription_finished: Optional[object] = None) -> Optional[str]: ...

    def __enter__(self) -> "RecorderProtocol": ...

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[object],
    ) -> None: ...


class SummariserProtocol(Protocol):
    """Minimal protocol for objects that can summarise text."""

    def complete(self, prompt: str) -> str: ...
