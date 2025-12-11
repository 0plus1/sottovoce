from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from langchain_community.chat_message_histories import SQLChatMessageHistory  # type: ignore[import-not-found]
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage  # type: ignore[import-not-found]

from src.config import Settings
from src.types import SummariserProtocol


class MemoryManager:
    """Maintains rolling recent context with persistent SQLite history."""

    def __init__(self, settings: Settings, session_id: Optional[str] = None):
        self.window = settings.context_window_messages
        db_path = Path("memory") / "memory.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or datetime.now(timezone.utc).strftime("session_%Y%m%dT%H%M%SZ")
        self.history = SQLChatMessageHistory(
            session_id=self.session_id,
            connection=f"sqlite:///{db_path}",
        )

    def build_context_prompt(self, user_text: str) -> str:
        msgs: List[BaseMessage] = self.history.messages[-self.window :]
        stitched: List[str] = []
        for m in msgs:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            stitched.append(f"{role}: {m.content}")
        stitched.append(f"User: {user_text}")
        return "\n".join(stitched)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.add_messages(
            [HumanMessage(content=user_text), AIMessage(content=assistant_text)]
        )

    def shrink_window(self, new_window: int) -> None:
        """Reduce the rolling window size (does not delete history, just limits prompt assembly)."""
        self.window = max(1, new_window)

    def clear_history(self) -> None:
        """Clear stored messages if the context must be reset."""
        try:
            self.history.clear()
        except Exception:
            # Fallback: overwrite with empty list if clear is unsupported.
            self.history.messages = []

    def summarise_history(self, summariser: SummariserProtocol, prompt_prefix: str) -> str:
        """
        Summarize the full stored history using the provided LLM client.
        The summariser must expose a .complete(prompt: str) -> str method.
        """
        msgs: List[BaseMessage] = self.history.messages
        transcript = []
        for m in msgs:
            role = "User" if isinstance(m, HumanMessage) else "Assistant"
            transcript.append(f"{role}: {m.content}")
        convo = "\n".join(transcript)
        prompt = f"{prompt_prefix}\n\nConversation:\n{convo}"
        summary = summariser.complete(prompt)
        # Reset and store summary as a single assistant message to keep context light.
        self.clear_history()
        self.history.add_messages([AIMessage(content=f"Summary: {summary}")])
        return summary
