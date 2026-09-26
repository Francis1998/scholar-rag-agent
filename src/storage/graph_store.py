"""SQLite graph storage for GraphRAG entity relationships."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from retrieval.models import Chunk, Entity, EntityEdge
from retrieval.scope import DocumentIdsInput, normalize_document_ids


def _document_filter(document_ids: tuple[str, ...] | None) -> str:
    """SQL structure only: IDs are bound separately, including quotes and Unicode."""
    if document_ids is None:
        return ""
    return f" AND c.document_id IN ({','.join('?' for _ in document_ids)})"


class SQLiteGraphStore:
    """Persist entity mentions and relationships for multi-hop retrieval."""

    def __init__(self, database_path: Path | str) -> None:
        """Create a graph store and initialize its schema."""
        self._database_path = str(database_path)
        self._initialize()

    def add_mentions(self, chunk: Chunk, entities: list[Entity]) -> None:
        """Persist entity mentions for one chunk."""
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO graph_chunks (
                    chunk_id, document_id, title, text, source, metadata
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.title,
                    chunk.text,
                    chunk.source,
                    json.dumps(chunk.metadata, sort_keys=True),
                ),
            )
            connection.executemany(
                """
                INSERT OR IGNORE INTO entity_mentions (entity_name, entity_key, label, chunk_id)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (entity.name, entity.name.lower(), entity.label, chunk.chunk_id)
                    for entity in entities
                ],
            )
            connection.commit()

    def add_edges(self, edges: list[EntityEdge]) -> None:
        """Persist entity co-mention edges."""
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            connection.executemany(
                """
                INSERT INTO entity_edges (
                    source_key, target_key, source_name, target_name, chunk_id, weight
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        edge.source.lower(),
                        edge.target.lower(),
                        edge.source,
                        edge.target,
                        edge.chunk_id,
                        edge.weight,
                    )
                    for edge in edges
                ],
            )
            connection.commit()

    def chunks_for_entities(
        self,
        entities: list[str],
        limit: int = 10,
        *,
        document_ids: DocumentIdsInput | None = None,
    ) -> list[Chunk]:
        """Return chunks mentioning any provided entities."""
        scope = normalize_document_ids(document_ids)
        scope_parameters = () if scope is None else scope
        if not entities:
            return []
        entity_keys = [entity.lower() for entity in entities]
        placeholders = ",".join("?" for _ in entity_keys)
        # Only the number of bound "?" placeholders is interpolated; every value
        # is passed as a query parameter, so this cannot be an injection vector.
        query = f"""
            SELECT DISTINCT c.chunk_id, c.document_id, c.title, c.text, c.source, c.metadata
            FROM graph_chunks c
            JOIN entity_mentions m ON c.chunk_id = m.chunk_id
            WHERE m.entity_key IN ({placeholders}) {_document_filter(scope)}
            LIMIT ?
        """  # nosec B608
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            rows = connection.execute(query, (*entity_keys, *scope_parameters, limit)).fetchall()
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

    def neighbours(
        self,
        entities: list[str],
        limit: int = 20,
        *,
        document_ids: DocumentIdsInput | None = None,
    ) -> list[str]:
        """Return neighbours using only edges owned by selected documents, before LIMIT."""
        scope = normalize_document_ids(document_ids)
        scope_parameters = () if scope is None else scope
        if not entities:
            return []
        entity_keys = [entity.lower() for entity in entities]
        placeholders = ",".join("?" for _ in entity_keys)
        join = "" if scope is None else "JOIN graph_chunks c ON c.chunk_id = e.chunk_id"
        # Only the number of bound "?" placeholders is interpolated; every value
        # is passed as a query parameter, so this cannot be an injection vector.
        query = f"""
            SELECT e.target_name FROM entity_edges e {join}
            WHERE e.source_key IN ({placeholders}) {_document_filter(scope)}
            UNION
            SELECT e.source_name FROM entity_edges e {join}
            WHERE e.target_key IN ({placeholders}) {_document_filter(scope)}
            LIMIT ?
        """  # nosec B608
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            rows = connection.execute(
                query, (*entity_keys, *scope_parameters, *entity_keys, *scope_parameters, limit)
            ).fetchall()
        return [str(row[0]) for row in rows]

    def _initialize(self) -> None:
        """Create graph tables when missing."""
        with closing(sqlite3.connect(self._database_path)) as connection, connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS graph_chunks (
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
                """
                CREATE TABLE IF NOT EXISTS entity_mentions (
                    entity_name TEXT NOT NULL,
                    entity_key TEXT NOT NULL,
                    label TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    UNIQUE(entity_key, chunk_id)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS entity_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_key TEXT NOT NULL,
                    target_key TEXT NOT NULL,
                    source_name TEXT NOT NULL,
                    target_name TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    weight REAL NOT NULL
                )
                """
            )
            connection.commit()
