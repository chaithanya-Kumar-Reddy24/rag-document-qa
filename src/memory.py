"""Conversation memory.

Only a short window of recent turns is kept. The history is used for ONE purpose:
rewriting follow-up questions into standalone search queries. It is never used as a
source of facts, so answers remain grounded in retrieved document chunks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.utils import truncate


@dataclass(frozen=True)
class ChatTurn:
    role: str  # "user" or "assistant"
    content: str


class ConversationMemory:
    """Sliding-window chat history."""

    def __init__(self, max_turns: int = 6, max_chars_per_message: int = 600) -> None:
        self._max_messages = max(1, max_turns) * 2
        self._max_chars = max_chars_per_message
        self._messages: list[ChatTurn] = []

    def add_user_message(self, content: str) -> None:
        self._append(ChatTurn("user", content))

    def add_assistant_message(self, content: str) -> None:
        self._append(ChatTurn("assistant", content))

    def _append(self, turn: ChatTurn) -> None:
        self._messages.append(turn)
        self._messages = self._messages[-self._max_messages :]

    def clear(self) -> None:
        self._messages.clear()

    def format_history(self) -> str:
        """History as plain text for the query-rewriting prompt ('' when empty)."""
        return "\n".join(
            f"{'User' if turn.role == 'user' else 'Assistant'}: {truncate(turn.content, self._max_chars)}"
            for turn in self._messages
        )

    def __len__(self) -> int:
        return len(self._messages)
