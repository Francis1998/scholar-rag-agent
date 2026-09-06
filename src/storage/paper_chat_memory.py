"""Paper-scoped multi-turn chat memory for scholarly RAG sessions.

Inspired by LocalGPT / PrivateGPT academic chat memory and PaperQA multi-turn
paper Q&A. Stores conversation turns keyed by ``session_id`` with optional
``document_ids`` / ``chunk_ids`` provenance. Distinct from the agent event log
(run state machine) and from retrieval postprocessors. Local SQLite memory for
GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2 paper-chat pipelines.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChatTurn:
    """One user or assistant turn in a paper chat session."""

    role: str
    content: str
    document_ids: tuple[str, ...] = ()
    chunk_ids: tuple[str, ...] = ()


class PaperChatMemory:
    """SQLite-backed multi-turn memory scoped to paper chat sessions.

    Turns are append-only per ``session_id``. ``format_context`` builds a
    bounded transcript string suitable for LLM prompts. Inputs are not mutated.
    Local memory for GPT-5.5 / Claude Sonnet 4.6 / Gemini 3.x / Kimi K2
    paper-chat pipelines (not a DOI connector).
    """

    def __init__(self, database_path: Path | str) -> None:
        """Create paper chat memory and ensure the schema exists."""
        self._database_path = str(database_path)
        self._initialize()

    def append_turn(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        document_ids: list[str] | tuple[str, ...] | None = None,
        chunk_ids: list[str] | tuple[str, ...] | None = None,
    ) -> int:
        """Append a turn and return its row id.

        Raises:
            ValueError: If ``session_id`` is blank, ``role`` is not
                ``user``/``assistant``, or ``content`` is blank.
        """
        session = session_id.strip()
        normalized_role = role.strip().lower()
        text = content.strip()
        if not session:
            raise ValueError("session_id must be a non-empty string")
        if normalized_role not in {"user", "assistant"}:
            raise ValueError("role must be 'user' or 'assistant'")
        if not text:
            raise ValueError("content must be a non-empty string")
        docs = tuple(item.strip() for item in (document_ids or ()) if item.strip())
        chunks = tuple(item.strip() for item in (chunk_ids or ()) if item.strip())
        with sqlite3.connect(self._database_path) as connection:
            cursor = connection.execute(
                """
                INSERT INTO paper_chat_turns (
                    timestamp, session_id, role, content, document_ids, chunk_ids
                )
                VALUES (
                    strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), ?, ?, ?, ?, ?
                )
                """,
                (
                    session,
                    normalized_role,
                    text,
                    json.dumps(list(docs), sort_keys=True),
                    json.dumps(list(chunks), sort_keys=True),
                ),
            )
            connection.commit()
            row_id = cursor.lastrowid
            if row_id is None:
                raise RuntimeError("SQLite did not return a turn id")
            return row_id

    def get_turns(self, session_id: str, *, limit: int | None = None) -> list[ChatTurn]:
        """Return turns for ``session_id`` in chronological order.

        When ``limit`` is set, returns the most recent ``limit`` turns still in
        chronological order.
        """
        session = session_id.strip()
        if not session:
            return []
        if limit is not None and limit <= 0:
            return []
        with sqlite3.connect(self._database_path) as connection:
            if limit is None:
                rows = connection.execute(
                    """
                    SELECT role, content, document_ids, chunk_ids
                    FROM paper_chat_turns
                    WHERE session_id = ?
                    ORDER BY id ASC
                    """,
                    (session,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT role, content, document_ids, chunk_ids FROM (
                        SELECT id, role, content, document_ids, chunk_ids
                        FROM paper_chat_turns
                        WHERE session_id = ?
                        ORDER BY id DESC
                        LIMIT ?
                    )
                    ORDER BY id ASC
                    """,
                    (session, limit),
                ).fetchall()
        return [
            ChatTurn(
                role=row[0],
                content=row[1],
                document_ids=tuple(json.loads(row[2])),
                chunk_ids=tuple(json.loads(row[3])),
            )
            for row in rows
        ]

    def format_context(self, session_id: str, *, max_chars: int = 4000) -> str:
        """Return a prompt-ready transcript truncated to ``max_chars``.

        Keeps the newest turns that fit. Empty sessions return ``\"\"``.
        """
        if max_chars <= 0:
            raise ValueError("max_chars must be a positive integer")
        turns = self.get_turns(session_id)
        if not turns:
            return ""
        lines = [f"{turn.role}: {turn.content}" for turn in turns]
        # Prefer newest turns when truncating.
        selected: list[str] = []
        used = 0
        for line in reversed(lines):
            extra = len(line) + (1 if selected else 0)
            if used + extra > max_chars:
                break
            selected.append(line)
            used += extra
        selected.reverse()
        return "\n".join(selected)

    def clear_session(self, session_id: str) -> int:
        """Delete all turns for ``session_id`` and return the deleted count."""
        session = session_id.strip()
        if not session:
            return 0
        with sqlite3.connect(self._database_path) as connection:
            cursor = connection.execute(
                "DELETE FROM paper_chat_turns WHERE session_id = ?",
                (session,),
            )
            connection.commit()
            return int(cursor.rowcount)

    def _initialize(self) -> None:
        """Create the paper-chat schema when missing."""
        with sqlite3.connect(self._database_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS paper_chat_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    document_ids TEXT NOT NULL,
                    chunk_ids TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_paper_chat_session
                ON paper_chat_turns(session_id, id)
                """
            )
            connection.commit()
