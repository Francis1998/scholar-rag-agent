"""SQLite document and chunk store."""

import json
import sqlite3
from pathlib import Path

from retrieval.models import Chunk, Document


class SQLiteDocumentStore:
    """Persist normalized documents and chunks in SQLite."""

    def __init__(self, database_path: Path | str) -> None:
        """Create a document store and initialize its schema."""
        self._database_path = str(database_path)
        self._initialize()

    def add_documents(self, documents: list[Document], chunks: list[Chunk]) -> None:
        """Persist documents and chunks."""
        with sqlite3.connect(self._database_path) as connection:
            connection.executemany(
                """
                INSERT OR REPLACE INTO documents (document_id, title, text, source, metadata)
                VALUES (?, ?, ?, ?, ?)
                """,
                [
                    (
                        document.document_id,
                        document.title,
                        document.text,
                        document.source,
                        json.dumps(document.metadata, sort_keys=True),
                    )
                    for document in documents
                ],
            )
            connection.executemany(
                """
                INSERT OR REPLACE INTO chunks (chunk_id, document_id, title, text, source, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.title,
                        chunk.text,
                        chunk.source,
                        json.dumps(chunk.metadata, sort_keys=True),
                    )
                    for chunk in chunks
                ],
            )
            connection.commit()

    def list_chunks(self) -> list[Chunk]:
        """Return all stored chunks."""
        select_chunks_sql = (
            "SELECT chunk_id, document_id, title, text, source, metadata "
            "FROM chunks ORDER BY chunk_id"
        )
        with sqlite3.connect(self._database_path) as connection:
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
        with sqlite3.connect(self._database_path) as connection:
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
            connection.commit()
