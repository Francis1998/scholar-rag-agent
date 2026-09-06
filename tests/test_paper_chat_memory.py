"""Tests for PaperChatMemory."""

from pathlib import Path

import pytest

from storage.paper_chat_memory import PaperChatMemory


def test_rejects_blank_session(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    with pytest.raises(ValueError, match="session_id"):
        memory.append_turn("  ", "user", "hello")


def test_rejects_invalid_role(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    with pytest.raises(ValueError, match="role"):
        memory.append_turn("s1", "system", "hello")


def test_rejects_blank_content(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    with pytest.raises(ValueError, match="content"):
        memory.append_turn("s1", "user", "   ")


def test_append_and_get_turns(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn(
        "s1",
        "user",
        "What is the method?",
        document_ids=["paper-a"],
        chunk_ids=["c1"],
    )
    memory.append_turn("s1", "assistant", "It uses hybrid retrieval.", document_ids=["paper-a"])
    turns = memory.get_turns("s1")
    assert len(turns) == 2
    assert turns[0].role == "user"
    assert turns[0].document_ids == ("paper-a",)
    assert turns[0].chunk_ids == ("c1",)
    assert turns[1].role == "assistant"
    assert turns[1].content == "It uses hybrid retrieval."


def test_sessions_are_isolated(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "question one")
    memory.append_turn("s2", "user", "question two")
    assert [turn.content for turn in memory.get_turns("s1")] == ["question one"]
    assert [turn.content for turn in memory.get_turns("s2")] == ["question two"]


def test_get_turns_limit_keeps_newest_in_order(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "one")
    memory.append_turn("s1", "assistant", "two")
    memory.append_turn("s1", "user", "three")
    turns = memory.get_turns("s1", limit=2)
    assert [turn.content for turn in turns] == ["two", "three"]


def test_format_context_truncates_to_newest(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "AAAAAAAAAA")
    memory.append_turn("s1", "assistant", "BBBBBBBBBB")
    memory.append_turn("s1", "user", "CCCCCCCCCC")
    # Only newest two lines fit: "assistant: BBBBBBBBBB\nuser: CCCCCCCCCC" = 41 chars
    text = memory.format_context("s1", max_chars=41)
    assert text == "assistant: BBBBBBBBBB\nuser: CCCCCCCCCC"
    assert "AAAAAAAAAA" not in text


def test_format_context_rejects_non_positive_max_chars(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    with pytest.raises(ValueError, match="max_chars"):
        memory.format_context("s1", max_chars=0)


def test_clear_session(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "keep me elsewhere")
    memory.append_turn("s2", "user", "delete me")
    deleted = memory.clear_session("s2")
    assert deleted == 1
    assert memory.get_turns("s2") == []
    assert len(memory.get_turns("s1")) == 1


def test_empty_session_context(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    assert memory.format_context("missing") == ""


def test_docstring_mentions_frontier_models() -> None:
    doc = PaperChatMemory.__doc__ or ""
    assert "GPT-5.5" in doc
    assert "Claude Sonnet 4.6" in doc
    assert "Gemini 3.x" in doc
    assert "Kimi K2" in doc
