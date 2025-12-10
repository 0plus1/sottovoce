from __future__ import annotations

from pathlib import Path
from typing import Optional
from datetime import datetime, timezone


class SessionLogger:
    # Append conversation turns to a session log file.

    def __init__(self, directory: str = "session_logs", filename: Optional[str] = None):
        self.dir_path = Path(directory)
        self.dir_path.mkdir(parents=True, exist_ok=True)
        if filename is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            filename = f"session_{timestamp}.log"
        self.file_path = self.dir_path / filename

    def append_turn(self, user_text: str, model_response: str) -> None:
        with self.file_path.open("a", encoding="utf-8") as f:
            f.write(f"USER: {user_text}\n")
            f.write(f"ASSISTANT: {model_response}\n")
            f.write("---\n")

    def path(self) -> Path:
        return self.file_path
