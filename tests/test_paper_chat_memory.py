"""Tests for PaperChatMemory."""

from pathlib import Path
from unittest.mock import patch

import pytest

from storage.paper_chat_memory import ChatTurn, PaperChatMemory


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
    text = memory.format_context("s1", max_chars=38)
    assert text == "assistant: BBBBBBBBBB\nuser: CCCCCCCCCC"
    assert len(text) == 38
    assert "AAAAAAAAAA" not in text


@pytest.mark.parametrize(
    "max_chars",
    [
        pytest.param(0, id="zero"),
        pytest.param(-1, id="negative"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(1.0, id="integral-float"),
        pytest.param(1.5, id="fractional-float"),
        pytest.param(0.0, id="zero-float"),
        pytest.param(-1.0, id="negative-float"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="positive-infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param("1", id="numeric-string"),
        pytest.param("", id="empty-string"),
        pytest.param(b"1", id="bytes"),
        pytest.param(None, id="none"),
        pytest.param([], id="list"),
        pytest.param({}, id="dict"),
        pytest.param(object(), id="object"),
    ],
)
@pytest.mark.parametrize("session_id", ["s1", "missing", "", "  "])
def test_format_context_rejects_invalid_max_chars_before_read(
    tmp_path: Path, max_chars: object, session_id: str
) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "question")
    memory.append_turn("s1", "assistant", "answer")
    with (
        patch.object(memory, "get_turns", wraps=memory.get_turns) as get_turns,
        patch(
            "storage.paper_chat_memory.sqlite3.connect",
            side_effect=AssertionError("unexpected SQLite access"),
        ) as connect,
    ):
        with pytest.raises(ValueError, match=r"^max_chars must be a positive integer$"):
            memory.format_context(session_id, max_chars=max_chars)  # type: ignore[arg-type]
        get_turns.assert_not_called()
        connect.assert_not_called()


@pytest.mark.parametrize(
    ("max_chars", "expected"),
    [
        (1, ""),
        (11, ""),
        (12, "assistant: \U0001f52c"),
        (22, "assistant: \U0001f52c"),
        (23, "user: caf\u00e9\nassistant: \U0001f52c"),
        pytest.param(10**400, "user: caf\u00e9\nassistant: \U0001f52c", id="huge-integer"),
    ],
)
def test_format_context_integer_budgets_count_characters_after_reopen(
    tmp_path: Path, max_chars: int, expected: str
) -> None:
    database = tmp_path / "chat.db"
    memory = PaperChatMemory(database)
    memory.append_turn("s1", "user", "caf\u00e9")
    memory.append_turn("s1", "assistant", "\U0001f52c")
    memory.append_turn("s2", "user", "another session")

    text = PaperChatMemory(database).format_context("s1", max_chars=max_chars)
    assert text == expected
    assert len(text) <= max_chars


@pytest.mark.parametrize("newest_line_length", [4000, 4001])
def test_format_context_default_budget(tmp_path: Path, newest_line_length: int) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "older")
    content = "x" * (newest_line_length - len("assistant: "))
    memory.append_turn("s1", "assistant", content)

    expected = f"assistant: {content}" if newest_line_length == 4000 else ""
    assert memory.format_context("s1") == expected


def test_format_context_does_not_skip_an_oversized_middle_turn(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "old")
    memory.append_turn("s1", "assistant", "x" * 30)
    memory.append_turn("s1", "user", "new")
    assert memory.format_context("s1", max_chars=20) == "user: new"


def test_clear_session(tmp_path: Path) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    memory.append_turn("s1", "user", "keep me elsewhere")
    memory.append_turn("s2", "user", "delete me")
    deleted = memory.clear_session("s2")
    assert deleted == 1
    assert memory.get_turns("s2") == []
    assert len(memory.get_turns("s1")) == 1


@pytest.mark.parametrize("max_chars", [1, 4000, pytest.param(10**400, id="huge-integer")])
def test_empty_session_context(tmp_path: Path, max_chars: int) -> None:
    memory = PaperChatMemory(tmp_path / "chat.db")
    assert memory.format_context("missing", max_chars=max_chars) == ""
    assert memory.format_context("", max_chars=max_chars) == ""
    assert memory.format_context("  ", max_chars=max_chars) == ""


def test_reopened_memory_preserves_normalized_turns_without_mutating_inputs(tmp_path: Path) -> None:
    database = tmp_path / "chat.db"
    memory = PaperChatMemory(database)
    document_ids = [" paper-a ", " "]
    chunk_ids = [" c1 ", ""]
    row_id = memory.append_turn(
        " s1 ", " User ", " question ", document_ids=document_ids, chunk_ids=chunk_ids
    )
    assert isinstance(row_id, int)
    assert row_id > 0
    assert document_ids == [" paper-a ", " "]
    assert chunk_ids == [" c1 ", ""]
    document_ids.append("paper-b")
    chunk_ids.append("c2")

    reopened = PaperChatMemory(str(database))
    assert reopened.get_turns(" s1 ") == [ChatTurn("user", "question", ("paper-a",), ("c1",))]
    assert reopened.format_context("s1") == "user: question"
