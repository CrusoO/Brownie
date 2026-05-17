"""Persistent per-session chat transcript for conversational context."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str


class ChatHistoryStore:
    """Append-only session transcripts stored beside Chroma data."""

    def __init__(self, memory_dir: str, max_turns: int = 40) -> None:
        os.makedirs(memory_dir, exist_ok=True)
        self.path = os.path.join(memory_dir, "chat_history.json")
        self.max_turns = max(2, max_turns)
        self.lock = threading.Lock()

    def list(self, session_id: str, limit: Optional[int] = None) -> List[ChatTurn]:
        with self.lock:
            sessions = self._read()
        turns = sessions.get(session_id, [])
        if limit is not None:
            return turns[-limit:]
        return turns

    def append_exchange(self, session_id: str, user: str, assistant: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        new_turns = [
            ChatTurn(role="user", content=user.strip(), created_at=timestamp),
            ChatTurn(
                role="assistant",
                content=assistant.strip(),
                created_at=datetime.now(timezone.utc).isoformat(),
            ),
        ]
        if not new_turns[0].content or not new_turns[1].content:
            return

        with self.lock:
            sessions = self._read()
            history = sessions.get(session_id, [])
            history.extend(new_turns)
            sessions[session_id] = history[-self.max_turns :]
            self._write(sessions)

    def clear(self, session_id: str) -> None:
        with self.lock:
            sessions = self._read()
            if session_id not in sessions:
                return
            del sessions[session_id]
            self._write(sessions)

    def has_history(self, session_id: str) -> bool:
        return bool(self.list(session_id, limit=1))

    def to_langchain_messages(self, turns: List[ChatTurn]) -> List[Any]:
        """Convert stored turns to LangChain message objects when available."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage
        except Exception:
            return []

        messages: List[Any] = []
        for turn in turns:
            if turn.role == "user":
                messages.append(HumanMessage(content=turn.content))
            else:
                messages.append(AIMessage(content=turn.content))
        return messages

    def _read(self) -> dict[str, List[ChatTurn]]:
        if not os.path.exists(self.path):
            return {}

        try:
            with open(self.path, "r", encoding="utf-8") as file:
                payload = json.load(file)
        except (json.JSONDecodeError, OSError):
            return {}

        sessions = payload.get("sessions", {}) if isinstance(payload, dict) else {}
        parsed: dict[str, List[ChatTurn]] = {}
        for session_id, turns in sessions.items():
            if not isinstance(turns, list):
                continue
            parsed[session_id] = [
                ChatTurn(**turn) for turn in turns if isinstance(turn, dict)
            ]
        return parsed

    def _write(self, sessions: dict[str, List[ChatTurn]]) -> None:
        temp_path = f"{self.path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "sessions": {
                        session_id: [turn.model_dump() for turn in turns]
                        for session_id, turns in sessions.items()
                    }
                },
                file,
                indent=2,
            )
        os.replace(temp_path, self.path)
