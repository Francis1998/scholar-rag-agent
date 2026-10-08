"""SQLite document and chunk store."""

import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from pathlib import Path

from retrieval.models import Chunk, Document


def _document_rows(documents: list[Document]) -> list[tuple[str, ...]]:
    return [
        (
            document.document_id,
            document.title,
            document.text,
            document.source,
            json.dumps(document.metadata, sort_keys=True),
        )
        for document in documents
    ]


def _chunk_rows(chunks: list[Chunk]) -> list[tuple[str, ...]]:
    return [
        (
            chunk.chunk_id,
            chunk.document_id,
            chunk.title,
            chunk.text,
            chunk.source,
            json.dumps(chunk.metadata, sort_keys=True),
        )
        for chunk in chunks
    ]


class SQLiteDocumentStore:
    """Persist normalized documents and chunks in SQLite."""

    def __init__(self, database_path: Path | str) -> None:
        """Create a document store and initialize its schema."""
        self._database_path = str(database_path)
        self._initialize()

    def add_documents(self, documents: list[Document], chunks: list[Chunk]) -> None:
        """Upsert supplied rows without removing other chunks, including when documents is empty."""
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            self._write_rows(connection, _document_rows(documents), _chunk_rows(chunks))
            connection.commit()

    @property
    def database_path(self) -> Path:
        """Return the resolved file used to coordinate a shared ingestion transaction."""
        return Path(self._database_path).resolve()

    @contextmanager
    def replacing_documents(
        self, documents: list[Document], chunks: list[Chunk]
    ) -> Iterator[sqlite3.Connection]:
        """Replace complete documents; commit only after dependent writes also succeed."""
        document_ids = {document.document_id for document in documents}
        if len(document_ids) != len(documents):
            raise ValueError("Replacement documents must have unique document_id values.")
        if any(chunk.document_id not in document_ids for chunk in chunks):
            raise ValueError("Each chunk must belong to a replacement document.")
        if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
            raise ValueError("Replacement chunks must have unique chunk_id values.")
        document_rows, chunk_rows = _document_rows(documents), _chunk_rows(chunks)
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            connection.execute("BEGIN IMMEDIATE")
            for chunk in chunks:
                owner = connection.execute(
                    "SELECT document_id FROM chunks WHERE chunk_id = ?", (chunk.chunk_id,)
                ).fetchone()
                if owner is not None and owner[0] not in document_ids:
                    raise ValueError("Replacement chunk_id belongs to an unrelated document.")
            connection.executemany(
                "DELETE FROM chunks WHERE document_id = ?",
                [(document.document_id,) for document in documents],
            )
            self._write_rows(connection, document_rows, chunk_rows)
            yield connection
            connection.commit()

    @staticmethod
    def _write_rows(
        connection: sqlite3.Connection,
        documents: list[tuple[str, ...]],
        chunks: list[tuple[str, ...]],
    ) -> None:
        connection.executemany(
            """
            INSERT OR REPLACE INTO documents (document_id, title, text, source, metadata)
            VALUES (?, ?, ?, ?, ?)
            """,
            documents,
        )
        connection.executemany(
            """
            INSERT OR REPLACE INTO chunks (chunk_id, document_id, title, text, source, metadata)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            chunks,
        )

    def list_chunks(self) -> list[Chunk]:
        """Return all stored chunks."""
        select_chunks_sql = (
            "SELECT chunk_id, document_id, title, text, source, metadata "
            "FROM chunks ORDER BY chunk_id"
        )
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            rows = connection.execute(select_chunks_sql).fetchall()
        return [
            Chunk(
                chunk_id=row[0],
                document_id=row[1],
                title=row[2],
                text=row[3],
                source=row[4],
                metadata=json.loads(row[5]),
            )
            for row in rows
        ]

    def _initialize(self) -> None:
        """Create document tables when missing."""
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    text TEXT NOT NULL,
                    source TEXT NOT NULL,
                    metadata TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_chunks_document_chunk "
                "ON chunks(document_id, chunk_id)"
            )
            connection.commit()
