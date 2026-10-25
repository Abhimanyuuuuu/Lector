"""SQLite persistence for canonical books, chunks, graph snapshots, and vectors.

The database is local application data. Schema changes are versioned with
SQLite's ``user_version`` and all book replacements are committed atomically.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Sequence

from .chunking import BookChunk, ChunkSourcePosition
from .story_graph import (
    GraphEvidence,
    GraphSourcePosition,
    StoryAlias,
    StoryEntity,
    StoryEvent,
    StoryGraph,
    StoryRelationship,
)

SCHEMA_VERSION = 8


class PersistenceIntegrityError(RuntimeError):
    """Persisted evidence is inconsistent with canonical source chunks."""


class ContentVersionConflictError(RuntimeError):
    """A caller reused a content-version ID for different canonical content."""


@dataclass(frozen=True, slots=True)
class PersistedBook:
    book_id: str
    content_version_id: str
    chunks: tuple[BookChunk, ...]
    graph: StoryGraph
    vectors: dict[str, tuple[float, ...]]
    embedding_identity: str | None
    embedding_dimensions: int | None


class SQLiteChunkStore:
    """Transactional SQLite source of truth; ``:memory:`` supports isolated tests."""

    def __init__(self, database_path: str | Path) -> None:
        self.database_path = str(database_path)
        if self.database_path != ":memory:":
            Path(self.database_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._lock = RLock()
        self._cache: dict[str, tuple[str, tuple[BookChunk, ...]]] = {}
        self._connection = sqlite3.connect(
            self.database_path, check_same_thread=False, isolation_level=None
        )
        self._connection.execute("PRAGMA foreign_keys = ON")
        try:
            self._initialize()
        except Exception:
            self._connection.close()
            raise

    def _initialize(self) -> None:
        version = self._connection.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise RuntimeError(
                f"database schema {version} is newer than supported schema {SCHEMA_VERSION}"
            )
        if version == 0:
            with self._transaction() as connection:
                connection.execute(
                    "CREATE TABLE books (book_id TEXT PRIMARY KEY, content_version_id TEXT NOT NULL)"
                )
                connection.execute(
                    """CREATE TABLE chunks (
                        chunk_id TEXT PRIMARY KEY,
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        section_index INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        UNIQUE(book_id, content_version_id, chunk_id)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE graph_snapshots (
                        book_id TEXT PRIMARY KEY REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        payload TEXT NOT NULL
                    )"""
                )
                connection.execute(
                    """CREATE TABLE semantic_vectors (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        provider_identity TEXT NOT NULL,
                        dimensions INTEGER NOT NULL,
                        vector_json TEXT NOT NULL,
                        PRIMARY KEY(book_id, content_version_id, chunk_id),
                        FOREIGN KEY(book_id, content_version_id, chunk_id)
                            REFERENCES chunks(book_id, content_version_id, chunk_id)
                            ON DELETE CASCADE
                    )"""
                )
                connection.execute("PRAGMA user_version = 1")
            version = 1
        if version == 1:
            # Additive migration: existing books, chunks, vectors, and graph
            # snapshots remain untouched. Jobs are version-scoped and cascade
            # only when their canonical book record is deliberately replaced.
            with self._transaction() as connection:
                connection.execute(
                    """CREATE TABLE graph_jobs (
                        job_id TEXT PRIMARY KEY,
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        boundary_section INTEGER NOT NULL,
                        boundary_unit INTEGER NOT NULL,
                        boundary_source_unit_id TEXT NOT NULL,
                        total_chunks INTEGER NOT NULL,
                        processed_chunks INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL,
                        attempts INTEGER NOT NULL DEFAULT 0,
                        current_chunk_ids TEXT NOT NULL DEFAULT '[]',
                        error TEXT,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(book_id, content_version_id, boundary_section, boundary_unit, boundary_source_unit_id)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE graph_processed_chunks (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        chunk_id TEXT NOT NULL,
                        PRIMARY KEY(book_id, content_version_id, chunk_id)
                    )"""
                )
                connection.execute(
                    "UPDATE graph_jobs SET status='queued', current_chunk_ids='[]' WHERE status='processing'"
                )
                # Existing graph snapshots prove that their evidence-bearing
                # chunks were already extracted. Seed those markers so a new
                # job only revisits legacy chunks that produced no stored facts.
                for book_id, version_id, payload in connection.execute(
                    "SELECT book_id,content_version_id,payload FROM graph_snapshots"
                ).fetchall():
                    try:
                        graph_data = json.loads(payload)
                    except (TypeError, json.JSONDecodeError):
                        continue
                    if (
                        not isinstance(graph_data, dict)
                        or graph_data.get("book_id") != book_id
                        or graph_data.get("content_version_id") != version_id
                    ):
                        continue
                    for chunk_id in _source_chunk_ids(graph_data):
                        exists = connection.execute(
                            "SELECT 1 FROM chunks WHERE book_id=? AND content_version_id=? AND chunk_id=?",
                            (book_id, version_id, chunk_id),
                        ).fetchone()
                        if exists:
                            connection.execute(
                                "INSERT OR IGNORE INTO graph_processed_chunks(book_id,content_version_id,chunk_id) VALUES(?,?,?)",
                                (book_id, version_id, chunk_id),
                            )
                connection.execute("PRAGMA user_version = 2")
            version = 2
        if version == 2:
            # Provider identity is captured per durable job so restarts and
            # retries cannot silently switch models midway through a job.
            with self._transaction() as connection:
                columns = {
                    row[1] for row in connection.execute("PRAGMA table_info(graph_jobs)")
                }
                if "provider" not in columns:
                    connection.execute(
                        "ALTER TABLE graph_jobs ADD COLUMN provider TEXT NOT NULL DEFAULT 'ollama'"
                    )
                if "provider_version" not in columns:
                    connection.execute(
                        "ALTER TABLE graph_jobs ADD COLUMN provider_version TEXT NOT NULL DEFAULT ''"
                    )
                connection.execute("PRAGMA user_version = 3")
            version = 3
        if version == 3:
            # Durable upload metadata is independent of indexed-book rows so
            # an uploaded file remains recoverable if indexing has not run yet.
            with self._transaction() as connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS saved_books (
                        book_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        original_filename TEXT NOT NULL,
                        storage_key TEXT NOT NULL UNIQUE,
                        content_version_id TEXT NOT NULL,
                        file_size INTEGER NOT NULL,
                        metadata_json TEXT NOT NULL DEFAULT '{}',
                        indexing_status TEXT NOT NULL DEFAULT 'uploaded',
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )"""
                )
                connection.execute("PRAGMA user_version = 4")
            version = 4
        if version == 4:
            with self._transaction() as connection:
                connection.execute(
                    """CREATE TABLE IF NOT EXISTS book_preparation_jobs (
                        book_id TEXT NOT NULL,
                        content_version_id TEXT NOT NULL,
                        source_units_json TEXT NOT NULL,
                        indexing_status TEXT NOT NULL DEFAULT 'queued',
                        indexing_completed INTEGER NOT NULL DEFAULT 0,
                        indexing_total INTEGER NOT NULL DEFAULT 0,
                        indexing_error TEXT,
                        embedding_status TEXT NOT NULL DEFAULT 'pending',
                        embedding_completed INTEGER NOT NULL DEFAULT 0,
                        embedding_total INTEGER NOT NULL DEFAULT 0,
                        embedding_error TEXT,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(book_id, content_version_id)
                    )"""
                )
                # Seed progress for books indexed before preparation jobs
                # existed. This lets startup reuse compatible vectors and
                # chunks instead of re-extracting or re-indexing an EPUB.
                for book_id, content_version_id in connection.execute(
                    "SELECT book_id,content_version_id FROM books"
                ).fetchall():
                    chunk_rows = connection.execute(
                        "SELECT payload FROM chunks WHERE book_id=? AND content_version_id=?",
                        (book_id, content_version_id),
                    ).fetchall()
                    source_unit_ids: set[str] = set()
                    embedding_total = 0
                    for (chunk_payload,) in chunk_rows:
                        try:
                            chunk_data = json.loads(chunk_payload)
                        except (TypeError, json.JSONDecodeError):
                            continue
                        if not isinstance(chunk_data, dict):
                            continue
                        source_unit_ids.update(
                            item for item in chunk_data.get("source_unit_ids", [])
                            if isinstance(item, str)
                        )
                        if isinstance(chunk_data.get("text"), str) and chunk_data["text"].strip():
                            embedding_total += 1
                    connection.execute(
                        """INSERT OR IGNORE INTO book_preparation_jobs(
                            book_id,content_version_id,source_units_json,indexing_status,
                            indexing_completed,indexing_total,embedding_status,embedding_total
                        ) VALUES(?,?,'[]','complete',?,?,'pending',?)""",
                        (book_id, content_version_id, len(source_unit_ids), len(source_unit_ids), embedding_total),
                    )
                connection.execute("PRAGMA user_version = 5")
            version = 5
        if version == 5:
            # Per-book Graph RAG is opt-in. Preserve earlier graph behavior
            # only for books that already have graph facts or extraction jobs.
            with self._transaction() as connection:
                for name, declaration in (
                    ("graph_rag_enabled", "INTEGER NOT NULL DEFAULT 0"),
                    ("graph_status", "TEXT NOT NULL DEFAULT 'not_enabled'"),
                    ("graph_completed", "INTEGER NOT NULL DEFAULT 0"),
                    ("graph_total", "INTEGER NOT NULL DEFAULT 0"),
                    ("graph_error", "TEXT"),
                ):
                    connection.execute(
                        f"ALTER TABLE book_preparation_jobs ADD COLUMN {name} {declaration}"
                    )
                connection.execute(
                    "ALTER TABLE graph_jobs ADD COLUMN job_scope TEXT NOT NULL DEFAULT 'confirmed_boundary'"
                )
                for book_id, version_id, payload in connection.execute(
                    "SELECT book_id,content_version_id,payload FROM graph_snapshots"
                ).fetchall():
                    try:
                        has_facts = bool(_source_chunk_ids(json.loads(payload)))
                    except (TypeError, json.JSONDecodeError):
                        has_facts = False
                    has_job = connection.execute(
                        "SELECT 1 FROM graph_jobs WHERE book_id=? AND content_version_id=? LIMIT 1",
                        (book_id, version_id),
                    ).fetchone() is not None
                    if has_facts or has_job:
                        connection.execute(
                            "UPDATE book_preparation_jobs SET graph_rag_enabled=1,graph_status='pending' WHERE book_id=? AND content_version_id=?",
                            (book_id, version_id),
                        )
                # A graph job is also evidence of a prior opt-in, including
                # books whose older graph snapshot was empty or absent.
                for book_id, version_id in connection.execute(
                    "SELECT DISTINCT book_id,content_version_id FROM graph_jobs"
                ).fetchall():
                    connection.execute(
                        "UPDATE book_preparation_jobs SET graph_rag_enabled=1,graph_status='pending' WHERE book_id=? AND content_version_id=?",
                        (book_id, version_id),
                    )
                # Do not continue unfinished legacy graph work through Groq.
                # Existing graph snapshots and their evidence remain intact.
                connection.execute(
                    "UPDATE graph_jobs SET status='cancelled',error='Graph extraction is local-only; retry with the configured Ollama model.' WHERE provider='groq' AND status IN ('queued','processing')"
                )
                connection.execute("PRAGMA user_version = 6")
            version = 6
        if version == 6:
            # BookNLP data is isolated from legacy generated StoryGraph
            # snapshots. This additive migration preserves all existing book,
            # chunk, graph, embedding, and QA records.
            with self._transaction() as connection:
                connection.execute(
                    """CREATE TABLE booknlp_graph_jobs (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        status TEXT NOT NULL DEFAULT 'pending',
                        stage TEXT NOT NULL DEFAULT 'queued',
                        completed_chunks INTEGER NOT NULL DEFAULT 0,
                        total_chunks INTEGER NOT NULL DEFAULT 0,
                        error TEXT,
                        bundle_json TEXT,
                        extraction_version TEXT NOT NULL,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(book_id, content_version_id)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE booknlp_profiles (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        character_id TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY(book_id, content_version_id, character_id)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE booknlp_mentions (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        mention_id TEXT NOT NULL,
                        character_id TEXT NOT NULL,
                        source_chunk_id TEXT,
                        section_index INTEGER,
                        chunk_index INTEGER,
                        payload TEXT NOT NULL,
                        PRIMARY KEY(book_id, content_version_id, mention_id),
                        FOREIGN KEY(book_id, content_version_id, source_chunk_id)
                            REFERENCES chunks(book_id, content_version_id, chunk_id)
                            ON DELETE CASCADE
                    )"""
                )
                connection.execute(
                    """CREATE TABLE booknlp_edges (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        edge_id TEXT NOT NULL,
                        source_character_id TEXT NOT NULL,
                        target_character_id TEXT NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY(book_id, content_version_id, edge_id)
                    )"""
                )
                connection.execute(
                    """CREATE TABLE booknlp_edge_evidence (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        evidence_id TEXT NOT NULL,
                        edge_id TEXT NOT NULL,
                        source_chunk_id TEXT NOT NULL,
                        section_index INTEGER NOT NULL,
                        chunk_index INTEGER NOT NULL,
                        payload TEXT NOT NULL,
                        PRIMARY KEY(book_id, content_version_id, evidence_id),
                        FOREIGN KEY(book_id, content_version_id, edge_id)
                            REFERENCES booknlp_edges(book_id, content_version_id, edge_id)
                            ON DELETE CASCADE,
                        FOREIGN KEY(book_id, content_version_id, source_chunk_id)
                            REFERENCES chunks(book_id, content_version_id, chunk_id)
                            ON DELETE CASCADE
                    )"""
                )
                connection.execute(
                    "CREATE INDEX booknlp_mentions_position ON booknlp_mentions(book_id,content_version_id,section_index,chunk_index)"
                )
                connection.execute(
                    "CREATE INDEX booknlp_evidence_position ON booknlp_edge_evidence(book_id,content_version_id,section_index,chunk_index)"
                )
                connection.execute("PRAGMA user_version = 7")
            version = 7
        if version == 7:
            # A rebuild is staged separately so the last committed graph stays
            # readable until its replacement is atomically validated/written.
            with self._transaction() as connection:
                connection.execute(
                    """CREATE TABLE booknlp_graph_rebuild_jobs (
                        book_id TEXT NOT NULL REFERENCES books(book_id) ON DELETE CASCADE,
                        content_version_id TEXT NOT NULL,
                        status TEXT NOT NULL,
                        completed_chunks INTEGER NOT NULL DEFAULT 0,
                        total_chunks INTEGER NOT NULL,
                        extraction_version TEXT NOT NULL,
                        error TEXT,
                        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY(book_id, content_version_id)
                    )"""
                )
                connection.execute("PRAGMA user_version = 8")
        # A process may have exited after claiming a batch. Extraction is
        # idempotent, so return that durable claim to the queue on each open.
        with self._transaction() as connection:
            connection.execute(
                "UPDATE graph_jobs SET status='queued',current_chunk_ids='[]' WHERE status='processing'"
            )
            connection.execute(
                "UPDATE book_preparation_jobs SET indexing_status='queued' WHERE indexing_status='processing'"
            )
            connection.execute(
                "UPDATE book_preparation_jobs SET embedding_status='pending' WHERE embedding_status='processing'"
            )
            connection.execute(
                "UPDATE booknlp_graph_rebuild_jobs SET status='queued' WHERE status='processing'"
            )

    def replace_book(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
        *,
        graph: StoryGraph | None,
        vectors: dict[str, tuple[float, ...]] | None,
        embedding_identity: str | None,
        embedding_dimensions: int | None,
        require_preparation_job: bool = False,
    ) -> None:
        """Replace canonical and derived state in one transaction.

        ``vectors=None`` preserves same-version vectors when ingestion has no
        configured provider; a new content version always clears old vectors.
        """
        with self._lock, self._transaction() as connection:
            if require_preparation_job and connection.execute(
                "SELECT 1 FROM book_preparation_jobs WHERE book_id=? AND content_version_id=?",
                (book_id, content_version_id),
            ).fetchone() is None:
                raise PersistenceIntegrityError("preparation job was superseded before commit")
            previous = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id = ?", (book_id,)
            ).fetchone()
            same_version = previous is not None and previous[0] == content_version_id
            if same_version:
                old_payloads = [
                    row[0]
                    for row in connection.execute(
                        "SELECT payload FROM chunks WHERE book_id = ? ORDER BY section_index, chunk_index, chunk_id",
                        (book_id,),
                    )
                ]
                new_payloads = [
                    _json(asdict(chunk))
                    for chunk in sorted(chunks, key=lambda item: (item.section_index, item.chunk_index, item.id))
                ]
                if old_payloads != new_payloads:
                    raise ContentVersionConflictError(
                        "content_version_id already exists with different content; use a new content version"
                    )
            preserved_vectors = []
            if same_version and vectors is None:
                preserved_vectors = connection.execute(
                    "SELECT chunk_id, provider_identity, dimensions, vector_json FROM semantic_vectors WHERE book_id = ? AND content_version_id = ?",
                    (book_id, content_version_id),
                ).fetchall()
            old_graph_row = None
            if same_version and graph is None:
                old_graph_row = connection.execute(
                    "SELECT payload FROM graph_snapshots WHERE book_id = ? AND content_version_id = ?",
                    (book_id, content_version_id),
                ).fetchone()
            if not same_version:
                connection.execute("DELETE FROM books WHERE book_id = ?", (book_id,))
            connection.execute(
                "INSERT INTO books(book_id, content_version_id) VALUES (?, ?) "
                "ON CONFLICT(book_id) DO UPDATE SET content_version_id=excluded.content_version_id",
                (book_id, content_version_id),
            )
            connection.execute("DELETE FROM chunks WHERE book_id = ?", (book_id,))
            connection.executemany(
                "INSERT INTO chunks(chunk_id, book_id, content_version_id, section_index, chunk_index, payload) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        chunk.id,
                        book_id,
                        content_version_id,
                        chunk.section_index,
                        chunk.chunk_index,
                        _json(asdict(chunk)),
                    )
                    for chunk in chunks
                ],
            )
            if not same_version:
                connection.execute("DELETE FROM semantic_vectors WHERE book_id = ?", (book_id,))
            if vectors is not None:
                connection.execute("DELETE FROM semantic_vectors WHERE book_id = ?", (book_id,))
                if vectors:
                    if not embedding_identity or not embedding_dimensions:
                        raise ValueError("vectors require embedding identity and dimensions")
                    chunk_ids = {chunk.id for chunk in chunks}
                    if not set(vectors).issubset(chunk_ids):
                        raise ValueError("semantic vector references an unknown chunk")
                    connection.executemany(
                        "INSERT INTO semantic_vectors(book_id, content_version_id, chunk_id, provider_identity, dimensions, vector_json) VALUES (?, ?, ?, ?, ?, ?)",
                        [
                            (
                                book_id,
                                content_version_id,
                                chunk_id,
                                embedding_identity,
                                embedding_dimensions,
                                _json(vector),
                            )
                            for chunk_id, vector in sorted(vectors.items())
                        ],
                    )
            elif preserved_vectors:
                current_ids = {chunk.id for chunk in chunks}
                if any(row[0] not in current_ids for row in preserved_vectors):
                    raise ValueError("same-version ingestion changed chunk identities; reindex embeddings")
                connection.executemany(
                    "INSERT INTO semantic_vectors(book_id, content_version_id, chunk_id, provider_identity, dimensions, vector_json) VALUES (?, ?, ?, ?, ?, ?)",
                    [(book_id, content_version_id, *row) for row in preserved_vectors],
                )
            if graph is None and old_graph_row is not None:
                graph = _graph_from_dict(json.loads(old_graph_row[0]))
            if graph is None:
                graph = StoryGraph(book_id, content_version_id)
            if graph.book_id != book_id or graph.content_version_id != content_version_id:
                raise ValueError("graph identity does not match replaced book")
            self._validate_graph(graph, connection)
            if graph.entities or graph.events or graph.relationships:
                connection.execute(
                    "INSERT INTO graph_snapshots(book_id, content_version_id, payload) VALUES (?, ?, ?) "
                    "ON CONFLICT(book_id) DO UPDATE SET content_version_id=excluded.content_version_id, payload=excluded.payload",
                    (book_id, content_version_id, _json(_graph_to_dict(graph))),
                )
            else:
                # An empty in-memory graph is the retrieval default, not
                # Graph RAG data. In particular, an opt-out book should not
                # acquire a graph snapshot just because its chunks were saved.
                connection.execute(
                    "DELETE FROM graph_snapshots WHERE book_id=? AND content_version_id=?",
                    (book_id, content_version_id),
                )
            connection.execute(
                "UPDATE saved_books SET indexing_status=CASE WHEN content_version_id=? THEN 'indexed' ELSE 'stale' END, updated_at=CURRENT_TIMESTAMP WHERE book_id=?",
                (content_version_id, book_id),
            )
        self._cache[book_id] = (content_version_id, tuple(chunks))

    def register_saved_epub(
        self,
        *,
        book_id: str,
        title: str,
        original_filename: str,
        storage_key: str,
        content_version_id: str,
        file_size: int,
        metadata: dict[str, object] | None = None,
    ) -> None:
        """Register an already atomically-written local EPUB file."""
        with self._lock, self._transaction() as connection:
            connection.execute(
                "DELETE FROM book_preparation_jobs WHERE book_id=? AND content_version_id<>?",
                (book_id, content_version_id),
            )
            indexed = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            status = (
                "indexed"
                if indexed is not None and indexed[0] == content_version_id
                else "uploaded"
            )
            connection.execute(
                """INSERT INTO saved_books(
                    book_id,title,original_filename,storage_key,content_version_id,
                    file_size,metadata_json,indexing_status,updated_at
                ) VALUES(?,?,?,?,?,?,?,?,CURRENT_TIMESTAMP)
                ON CONFLICT(book_id) DO UPDATE SET
                    title=excluded.title,original_filename=excluded.original_filename,
                    storage_key=excluded.storage_key,
                    content_version_id=excluded.content_version_id,
                    file_size=excluded.file_size,metadata_json=excluded.metadata_json,
                    indexing_status=excluded.indexing_status,updated_at=CURRENT_TIMESTAMP""",
                (book_id, title, original_filename, storage_key, content_version_id,
                 file_size, _json(metadata or {}), status),
            )

    def saved_book_library(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT s.book_id,s.title,s.original_filename,s.storage_key,s.content_version_id,
                s.file_size,s.metadata_json,s.indexing_status,s.updated_at,
                j.indexing_status,j.indexing_completed,j.indexing_total,j.indexing_error,
                j.embedding_status,j.embedding_completed,j.embedding_total,j.embedding_error,
                j.graph_rag_enabled,j.graph_status,j.graph_completed,j.graph_total,j.graph_error
                FROM saved_books s LEFT JOIN book_preparation_jobs j
                ON j.book_id=s.book_id AND j.content_version_id=s.content_version_id
                ORDER BY s.title COLLATE NOCASE,s.book_id"""
            ).fetchall()
        return tuple(
            {
                "book_id": row[0], "title": row[1], "original_filename": row[2],
                "storage_key": row[3], "content_version_id": row[4],
                "file_size": row[5], "metadata": json.loads(row[6]),
                "indexing_status": row[7], "updated_at": row[8],
                "preparation": ({
                    "indexing_status": row[9], "indexing_completed": row[10],
                    "indexing_total": row[11], "indexing_error": row[12],
                    "embedding_status": row[13], "embedding_completed": row[14],
                    "embedding_total": row[15], "embedding_error": row[16],
                    "graph_rag_enabled": bool(row[17]), "graph_status": row[18],
                    "graph_completed": row[19], "graph_total": row[20],
                    "graph_error": row[21],
                } if row[9] is not None else None),
            }
            for row in rows
        )

    def saved_book(self, book_id: str) -> dict[str, object] | None:
        return next((item for item in self.saved_book_library() if item["book_id"] == book_id), None)

    def delete_book(self, book_id: str) -> str | None:
        """Delete all book-scoped records transactionally; return its file key."""
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT storage_key FROM saved_books WHERE book_id=?", (book_id,)
            ).fetchone()
            connection.execute("DELETE FROM book_preparation_jobs WHERE book_id=?", (book_id,))
            connection.execute("DELETE FROM books WHERE book_id=?", (book_id,))
            connection.execute("DELETE FROM saved_books WHERE book_id=?", (book_id,))
        self._cache.pop(book_id, None)
        return row[0] if row else None

    def schedule_book_preparation(
        self, *, book_id: str, content_version_id: str,
        source_units: Sequence[dict[str, object]], graph_rag_enabled: bool | None = None,
    ) -> dict[str, object]:
        payload = _json(list(source_units))
        with self._lock, self._transaction() as connection:
            connection.execute(
                "DELETE FROM book_preparation_jobs WHERE book_id=? AND content_version_id<>?",
                (book_id, content_version_id),
            )
            connection.execute(
                """INSERT INTO book_preparation_jobs(book_id,content_version_id,source_units_json,indexing_status,embedding_status,graph_rag_enabled,graph_status)
                VALUES(?,?,?,'queued','pending',COALESCE(?,0),CASE WHEN COALESCE(?,0)=1 THEN 'queued' ELSE 'not_enabled' END) ON CONFLICT(book_id,content_version_id) DO UPDATE SET
                source_units_json=CASE WHEN book_preparation_jobs.indexing_status='complete'
                    THEN book_preparation_jobs.source_units_json ELSE excluded.source_units_json END,
                indexing_status=CASE WHEN book_preparation_jobs.indexing_status='complete'
                    THEN 'complete' ELSE 'queued' END,
                indexing_error=NULL,
                embedding_status=CASE WHEN book_preparation_jobs.embedding_status='complete'
                    THEN 'complete' ELSE 'pending' END,
                embedding_error=NULL,
                graph_rag_enabled=COALESCE(?,book_preparation_jobs.graph_rag_enabled),
                graph_status=CASE
                    WHEN COALESCE(?,book_preparation_jobs.graph_rag_enabled)=0 THEN 'not_enabled'
                    WHEN book_preparation_jobs.graph_status IN ('complete','processing','queued') THEN book_preparation_jobs.graph_status
                    ELSE 'queued' END,
                graph_error=CASE WHEN COALESCE(?,book_preparation_jobs.graph_rag_enabled)=0 THEN NULL ELSE book_preparation_jobs.graph_error END,
                updated_at=CURRENT_TIMESTAMP""",
                (book_id, content_version_id, payload, graph_rag_enabled, graph_rag_enabled,
                 graph_rag_enabled, graph_rag_enabled, graph_rag_enabled),
            )
            if graph_rag_enabled is False:
                connection.execute(
                    "UPDATE graph_jobs SET status='cancelled',current_chunk_ids='[]',error='Graph RAG disabled for this book.' WHERE book_id=? AND content_version_id=? AND status IN ('queued','processing')",
                    (book_id, content_version_id),
                )
        status = self.book_preparation_status(book_id, content_version_id)
        assert status is not None
        return status

    def book_preparation_status(self, book_id: str, content_version_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT book_id,content_version_id,indexing_status,indexing_completed,indexing_total,indexing_error,embedding_status,embedding_completed,embedding_total,embedding_error,updated_at,graph_rag_enabled,graph_status,graph_completed,graph_total,graph_error FROM book_preparation_jobs WHERE book_id=? AND content_version_id=?",
                (book_id, content_version_id),
            ).fetchone()
        if row is None:
            return None
        names = ("book_id","content_version_id","indexing_status","indexing_completed","indexing_total","indexing_error","embedding_status","embedding_completed","embedding_total","embedding_error","updated_at","graph_rag_enabled","graph_status","graph_completed","graph_total","graph_error")
        result = dict(zip(names, row, strict=True))
        result["graph_rag_enabled"] = bool(result["graph_rag_enabled"])
        return result

    def pending_book_preparations(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT book_id,content_version_id,source_units_json FROM book_preparation_jobs WHERE indexing_status IN ('queued','processing') OR embedding_status IN ('pending','processing') OR (graph_rag_enabled=1 AND graph_status IN ('queued','processing','pending'))"
            ).fetchall()
        return tuple({"book_id": row[0], "content_version_id": row[1], "source_units": json.loads(row[2])} for row in rows)

    def update_book_preparation(self, book_id: str, version: str, **fields: object) -> None:
        allowed = {"indexing_status","indexing_completed","indexing_total","indexing_error","embedding_status","embedding_completed","embedding_total","embedding_error","graph_rag_enabled","graph_status","graph_completed","graph_total","graph_error"}
        if not fields or set(fields) - allowed:
            raise ValueError("invalid preparation status update")
        assignments = ",".join(f"{name}=?" for name in fields)
        values = [fields[name] for name in fields]
        with self._lock, self._transaction() as connection:
            connection.execute(
                f"UPDATE book_preparation_jobs SET {assignments},updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (*values, book_id, version),
            )
            if fields.get("indexing_status") == "complete":
                connection.execute("UPDATE book_preparation_jobs SET source_units_json='[]' WHERE book_id=? AND content_version_id=?", (book_id, version))

    def graph_rag_enabled(self, book_id: str, version: str) -> bool | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT graph_rag_enabled FROM book_preparation_jobs WHERE book_id=? AND content_version_id=?",
                (book_id, version),
            ).fetchone()
        return bool(row[0]) if row else None

    def graph_preparations_to_resume(self) -> tuple[dict[str, str], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT book_id,content_version_id FROM book_preparation_jobs WHERE graph_rag_enabled=1 AND graph_status IN ('pending','queued','processing','unavailable') ORDER BY updated_at,book_id"
            ).fetchall()
        return tuple(
            {"book_id": row[0], "content_version_id": row[1]} for row in rows
        )

    def schedule_booknlp_job(
        self, *, book_id: str, content_version_id: str,
        total_chunks: int, extraction_version: str, retry: bool = False,
    ) -> dict[str, object]:
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            if current is None or current[0] != content_version_id:
                raise PersistenceIntegrityError("BookNLP job version does not match canonical book")
            row = connection.execute(
                "SELECT status,bundle_json,error FROM booknlp_graph_jobs WHERE book_id=? AND content_version_id=?",
                (book_id, content_version_id),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO booknlp_graph_jobs(book_id,content_version_id,status,stage,total_chunks,extraction_version) VALUES(?,?,'queued','queued',?,?)",
                    (book_id, content_version_id, total_chunks, extraction_version),
                )
            elif row[0] in ("complete", "completed"):
                pass
            elif row[0] == "failed" and retry:
                connection.execute(
                    "UPDATE booknlp_graph_jobs SET status='queued',stage=CASE WHEN bundle_json IS NULL THEN 'queued' ELSE 'persisting' END,error=NULL,total_chunks=?,extraction_version=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                    (total_chunks, extraction_version, book_id, content_version_id),
                )
            elif row[0] in ("unavailable", "failed"):
                connection.execute(
                    "UPDATE booknlp_graph_jobs SET status='queued',stage=CASE WHEN bundle_json IS NULL THEN 'queued' ELSE 'persisting' END,error=NULL,total_chunks=?,extraction_version=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                    (total_chunks, extraction_version, book_id, content_version_id),
                )
            result = self._booknlp_job_locked(connection, book_id, content_version_id)
            connection.execute(
                "UPDATE book_preparation_jobs SET graph_status=?,graph_completed=?,graph_total=?,graph_error=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                (result["status"], result["completed_chunks"], result["total_chunks"], result["error"], book_id, content_version_id),
            )
        return result

    def booknlp_job(self, book_id: str, content_version_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._booknlp_job_locked(self._connection, book_id, content_version_id)
        return row

    @staticmethod
    def _booknlp_job_locked(connection: sqlite3.Connection, book_id: str, version: str) -> dict[str, object] | None:
        row = connection.execute(
            "SELECT book_id,content_version_id,status,stage,completed_chunks,total_chunks,error,extraction_version,updated_at,bundle_json FROM booknlp_graph_jobs WHERE book_id=? AND content_version_id=?",
            (book_id, version),
        ).fetchone()
        if row is None:
            return None
        return {
            "book_id": row[0], "content_version_id": row[1], "status": row[2],
            "stage": row[3], "completed_chunks": row[4], "total_chunks": row[5],
            "error": row[6], "extraction_version": row[7], "updated_at": row[8],
            "bundle_json": row[9],
        }

    def set_booknlp_job_status(
        self, *, book_id: str, content_version_id: str, status: str,
        stage: str, error: str | None = None,
    ) -> None:
        allowed = {"queued", "processing", "persisting", "complete", "failed", "unavailable", "cancelled"}
        if status not in allowed:
            raise ValueError("invalid BookNLP job status")
        with self._lock, self._transaction() as connection:
            connection.execute(
                "UPDATE booknlp_graph_jobs SET status=?,stage=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (status, stage, error[:1000] if error else None, book_id, content_version_id),
            )
            connection.execute(
                "UPDATE book_preparation_jobs SET graph_status=?,graph_error=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                (status, error[:1000] if error else None, book_id, content_version_id),
            )

    def persist_booknlp_bundle(
        self, *, book_id: str, content_version_id: str,
        bundle: dict[str, object], total_chunks: int,
    ) -> None:
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            if current is None or current[0] != content_version_id:
                raise PersistenceIntegrityError("BookNLP output belongs to a stale content version")
            connection.execute(
                "UPDATE booknlp_graph_jobs SET status='persisting',stage='persisting',bundle_json=?,total_chunks=?,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (_json(bundle), total_chunks, book_id, content_version_id),
            )

    def pending_booknlp_jobs(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT book_id,content_version_id FROM booknlp_graph_jobs WHERE status IN ('queued','processing','persisting') ORDER BY updated_at,book_id"
            ).fetchall()
        return tuple({"book_id": row[0], "content_version_id": row[1]} for row in rows)

    def schedule_booknlp_rebuild(
        self, *, book_id: str, content_version_id: str,
        total_chunks: int, extraction_version: str,
    ) -> dict[str, object]:
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            if current is None or current[0] != content_version_id:
                raise PersistenceIntegrityError("graph rebuild version does not match canonical book")
            connection.execute(
                """INSERT INTO booknlp_graph_rebuild_jobs(
                    book_id,content_version_id,status,completed_chunks,total_chunks,
                    extraction_version,error,updated_at
                ) VALUES(?,?,'queued',0,?,?,NULL,CURRENT_TIMESTAMP)
                ON CONFLICT(book_id,content_version_id) DO UPDATE SET
                    status=CASE WHEN booknlp_graph_rebuild_jobs.status IN ('queued','processing')
                                THEN booknlp_graph_rebuild_jobs.status ELSE 'queued' END,
                    completed_chunks=CASE WHEN booknlp_graph_rebuild_jobs.status IN ('queued','processing')
                                           THEN booknlp_graph_rebuild_jobs.completed_chunks ELSE 0 END,
                    total_chunks=excluded.total_chunks,
                    extraction_version=excluded.extraction_version,
                    error=NULL,updated_at=CURRENT_TIMESTAMP""",
                (book_id, content_version_id, total_chunks, extraction_version),
            )
            return self._booknlp_rebuild_job_locked(connection, book_id, content_version_id) or {}

    @staticmethod
    def _booknlp_rebuild_job_locked(
        connection: sqlite3.Connection, book_id: str, version: str
    ) -> dict[str, object] | None:
        row = connection.execute(
            "SELECT book_id,content_version_id,status,completed_chunks,total_chunks,extraction_version,error,updated_at FROM booknlp_graph_rebuild_jobs WHERE book_id=? AND content_version_id=?",
            (book_id, version),
        ).fetchone()
        if row is None:
            return None
        return {"book_id": row[0], "content_version_id": row[1], "status": row[2],
                "completed_chunks": row[3], "total_chunks": row[4],
                "extraction_version": row[5], "error": row[6], "updated_at": row[7]}

    def booknlp_rebuild_job(self, book_id: str, content_version_id: str) -> dict[str, object] | None:
        with self._lock:
            return self._booknlp_rebuild_job_locked(self._connection, book_id, content_version_id)

    def pending_booknlp_rebuild_jobs(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT book_id,content_version_id FROM booknlp_graph_rebuild_jobs WHERE status IN ('queued','processing') ORDER BY updated_at,book_id"
            ).fetchall()
        return tuple({"book_id": row[0], "content_version_id": row[1]} for row in rows)

    def set_booknlp_rebuild_status(
        self, *, book_id: str, content_version_id: str, status: str,
        error: str | None = None,
    ) -> None:
        if status not in {"queued", "processing", "failed", "complete", "cancelled"}:
            raise ValueError("invalid graph rebuild status")
        with self._lock, self._transaction() as connection:
            connection.execute(
                "UPDATE booknlp_graph_rebuild_jobs SET status=?,error=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (status, error[:1000] if error else None, book_id, content_version_id),
            )

    def commit_booknlp_rebuild(
        self, *, book_id: str, content_version_id: str,
        bundle: dict[str, object], total_chunks: int,
    ) -> None:
        """Atomically replace graph rows after validating the staged bundle."""
        if bundle.get("book_id") != book_id or bundle.get("content_version_id") != content_version_id:
            raise PersistenceIntegrityError("staged graph bundle has a mismatched book revision")
        profiles = bundle.get("profiles")
        mentions = bundle.get("mentions")
        edges = bundle.get("edges")
        if not all(isinstance(value, list) for value in (profiles, mentions, edges)):
            raise PersistenceIntegrityError("staged graph bundle collections are malformed")
        profile_ids = {str(item.get("character_id")) for item in profiles if isinstance(item, dict)}
        if len(profile_ids) != len(profiles):
            raise PersistenceIntegrityError("staged graph bundle has invalid or duplicate profiles")
        for mention in mentions:
            if not isinstance(mention, dict) or str(mention.get("character_id")) not in profile_ids:
                raise PersistenceIntegrityError("staged graph mention references an unknown character")
        for edge in edges:
            if not isinstance(edge, dict) or str(edge.get("source_character_id")) not in profile_ids or str(edge.get("target_character_id")) not in profile_ids:
                raise PersistenceIntegrityError("staged graph relationship references an unknown character")
            if not isinstance(edge.get("evidence"), list) or not edge["evidence"]:
                raise PersistenceIntegrityError("staged graph relationships require source evidence")
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            if current is None or current[0] != content_version_id:
                raise PersistenceIntegrityError("staged graph belongs to a stale content version")
            valid_chunks = {row[0] for row in connection.execute(
                "SELECT chunk_id FROM chunks WHERE book_id=? AND content_version_id=?", (book_id, content_version_id)
            )}
            for mention in mentions:
                source = mention.get("source_chunk_id")
                if source is not None and source not in valid_chunks:
                    raise PersistenceIntegrityError("staged mention references an unknown canonical chunk")
            for edge in edges:
                for evidence in edge["evidence"]:
                    if not isinstance(evidence, dict) or evidence.get("source_chunk_id") not in valid_chunks:
                        raise PersistenceIntegrityError("staged relationship evidence references an unknown canonical chunk")
            connection.execute("DELETE FROM booknlp_edges WHERE book_id=? AND content_version_id=?", (book_id, content_version_id))
            connection.execute("DELETE FROM booknlp_mentions WHERE book_id=? AND content_version_id=?", (book_id, content_version_id))
            connection.execute("DELETE FROM booknlp_profiles WHERE book_id=? AND content_version_id=?", (book_id, content_version_id))
            for profile in profiles:
                connection.execute(
                    "INSERT INTO booknlp_profiles(book_id,content_version_id,character_id,payload) VALUES(?,?,?,?)",
                    (book_id, content_version_id, profile["character_id"], _json(profile)),
                )
            for mention in mentions:
                connection.execute(
                    "INSERT INTO booknlp_mentions(book_id,content_version_id,mention_id,character_id,source_chunk_id,section_index,chunk_index,payload) VALUES(?,?,?,?,?,?,?,?)",
                    (book_id, content_version_id, mention["mention_id"], mention["character_id"], mention.get("source_chunk_id"), mention.get("section_index"), mention.get("chunk_index"), _json(mention)),
                )
            for edge in edges:
                edge_payload = {key: value for key, value in edge.items() if key != "evidence"}
                connection.execute(
                    "INSERT INTO booknlp_edges(book_id,content_version_id,edge_id,source_character_id,target_character_id,payload) VALUES(?,?,?,?,?,?)",
                    (book_id, content_version_id, edge["edge_id"], edge["source_character_id"], edge["target_character_id"], _json(edge_payload)),
                )
                for evidence in edge["evidence"]:
                    connection.execute(
                        "INSERT INTO booknlp_edge_evidence(book_id,content_version_id,evidence_id,edge_id,source_chunk_id,section_index,chunk_index,payload) VALUES(?,?,?,?,?,?,?,?)",
                        (book_id, content_version_id, evidence["evidence_id"], edge["edge_id"], evidence["source_chunk_id"], evidence["section_index"], evidence["chunk_index"], _json(evidence)),
                    )
            connection.execute(
                "UPDATE booknlp_graph_jobs SET status='complete',stage='complete',completed_chunks=?,total_chunks=?,bundle_json=?,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (total_chunks, total_chunks, _json(bundle), book_id, content_version_id),
            )
            connection.execute(
                "UPDATE book_preparation_jobs SET graph_status='complete',graph_completed=?,graph_total=?,graph_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (total_chunks, total_chunks, book_id, content_version_id),
            )
            connection.execute(
                "UPDATE booknlp_graph_rebuild_jobs SET status='complete',completed_chunks=?,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                (total_chunks, book_id, content_version_id),
            )

    def commit_booknlp_batch(
        self, *, book_id: str, content_version_id: str, bundle: dict[str, object],
        chunk_ids: Sequence[str], completed_chunks: int, total_chunks: int,
        complete: bool,
    ) -> None:
        """Idempotently commit one chunk-ordered graph assembly batch."""
        allowed = set(chunk_ids)
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            if current is None or current[0] != content_version_id:
                raise PersistenceIntegrityError("BookNLP batch belongs to a stale content version")
            profiles = bundle.get("profiles", [])
            if not isinstance(profiles, list):
                raise PersistenceIntegrityError("BookNLP bundle profiles are malformed")
            if not connection.execute(
                "SELECT 1 FROM booknlp_profiles WHERE book_id=? AND content_version_id=? LIMIT 1",
                (book_id, content_version_id),
            ).fetchone():
                for profile in profiles:
                    character_id = str(profile["character_id"])
                    connection.execute(
                        "INSERT OR IGNORE INTO booknlp_profiles(book_id,content_version_id,character_id,payload) VALUES(?,?,?,?)",
                        (book_id, content_version_id, character_id, _json(profile)),
                    )
            for mention in bundle.get("mentions", []):
                source_chunk_id = mention.get("source_chunk_id")
                if source_chunk_id is not None and source_chunk_id not in allowed:
                    continue
                # Keep unmapped mentions once, in the first committed batch,
                # but never use them as graph or passage evidence.
                if source_chunk_id is None and completed_chunks > len(chunk_ids):
                    continue
                connection.execute(
                    "INSERT OR IGNORE INTO booknlp_mentions(book_id,content_version_id,mention_id,character_id,source_chunk_id,section_index,chunk_index,payload) VALUES(?,?,?,?,?,?,?,?)",
                    (book_id, content_version_id, mention["mention_id"], mention["character_id"], source_chunk_id, mention.get("section_index"), mention.get("chunk_index"), _json(mention)),
                )
            for edge in bundle.get("edges", []):
                edge_evidence = [item for item in edge.get("evidence", []) if item.get("source_chunk_id") in allowed]
                if not edge_evidence:
                    continue
                edge_payload = {key: value for key, value in edge.items() if key != "evidence"}
                connection.execute(
                    "INSERT OR IGNORE INTO booknlp_edges(book_id,content_version_id,edge_id,source_character_id,target_character_id,payload) VALUES(?,?,?,?,?,?)",
                    (book_id, content_version_id, edge["edge_id"], edge["source_character_id"], edge["target_character_id"], _json(edge_payload)),
                )
                for evidence in edge_evidence:
                    connection.execute(
                        "INSERT OR IGNORE INTO booknlp_edge_evidence(book_id,content_version_id,evidence_id,edge_id,source_chunk_id,section_index,chunk_index,payload) VALUES(?,?,?,?,?,?,?,?)",
                        (book_id, content_version_id, evidence["evidence_id"], edge["edge_id"], evidence["source_chunk_id"], evidence["section_index"], evidence["chunk_index"], _json(evidence)),
                    )
            connection.execute(
                "UPDATE booknlp_graph_jobs SET status=?,stage=?,completed_chunks=?,total_chunks=?,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=?",
                ("complete" if complete else "persisting", "complete" if complete else "persisting", completed_chunks, total_chunks, book_id, content_version_id),
            )
            connection.execute(
                "UPDATE book_preparation_jobs SET graph_status=?,graph_completed=?,graph_total=?,graph_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                ("complete" if complete else "processing", completed_chunks, total_chunks, book_id, content_version_id),
            )

    def load_booknlp_graph(self, book_id: str, content_version_id: str) -> dict[str, object] | None:
        with self._lock:
            job = self._booknlp_job_locked(self._connection, book_id, content_version_id)
            if job is None or job["status"] != "complete":
                return None
            profiles = [json.loads(row[0]) for row in self._connection.execute(
                "SELECT payload FROM booknlp_profiles WHERE book_id=? AND content_version_id=? ORDER BY character_id", (book_id, content_version_id)
            )]
            mentions = [json.loads(row[0]) for row in self._connection.execute(
                "SELECT payload FROM booknlp_mentions WHERE book_id=? AND content_version_id=? ORDER BY COALESCE(section_index,2147483647),COALESCE(chunk_index,2147483647),mention_id", (book_id, content_version_id)
            )]
            edges = []
            for edge_id, payload in self._connection.execute(
                "SELECT edge_id,payload FROM booknlp_edges WHERE book_id=? AND content_version_id=? ORDER BY edge_id", (book_id, content_version_id)
            ):
                data = json.loads(payload)
                data["evidence"] = [json.loads(row[0]) for row in self._connection.execute(
                    "SELECT payload FROM booknlp_edge_evidence WHERE book_id=? AND content_version_id=? AND edge_id=? ORDER BY section_index,chunk_index,evidence_id", (book_id, content_version_id, edge_id)
                )]
                edges.append(data)
        return {"book_id": book_id, "content_version_id": content_version_id,
                "extraction_version": job["extraction_version"], "profiles": profiles,
                "mentions": mentions, "edges": edges}

    def replace_graph(self, graph: StoryGraph) -> None:
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id = ?", (graph.book_id,)
            ).fetchone()
            if current is None or current[0] != graph.content_version_id:
                raise PersistenceIntegrityError("graph version does not match canonical book")
            self._validate_graph(graph, connection)
            connection.execute(
                "INSERT INTO graph_snapshots(book_id, content_version_id, payload) VALUES (?, ?, ?) "
                "ON CONFLICT(book_id) DO UPDATE SET content_version_id=excluded.content_version_id, payload=excluded.payload",
                (graph.book_id, graph.content_version_id, _json(_graph_to_dict(graph))),
            )

    def schedule_graph_job(
        self, *, job_id: str, book_id: str, content_version_id: str,
        boundary_section: int, boundary_unit: int, boundary_source_unit_id: str,
        total_chunks: int, already_processed: int = 0, retry: bool = False,
        provider: str = "ollama", provider_version: str = "",
        job_scope: str = "confirmed_boundary",
    ) -> dict[str, object]:
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (book_id,)
            ).fetchone()
            if current is None or current[0] != content_version_id:
                raise PersistenceIntegrityError("graph job version does not match canonical book")
            row = connection.execute(
                "SELECT job_id,status,attempts,error,provider,provider_version FROM graph_jobs WHERE book_id=? AND content_version_id=? AND boundary_section=? AND boundary_unit=? AND boundary_source_unit_id=?",
                (book_id, content_version_id, boundary_section, boundary_unit, boundary_source_unit_id),
            ).fetchone()
            if row is None:
                connection.execute(
                    "INSERT INTO graph_jobs(job_id,book_id,content_version_id,boundary_section,boundary_unit,boundary_source_unit_id,total_chunks,processed_chunks,status,provider,provider_version,job_scope) VALUES(?,?,?,?,?,?,?,?, 'queued',?,?,?)",
                    (job_id, book_id, content_version_id, boundary_section, boundary_unit, boundary_source_unit_id, total_chunks, min(total_chunks, already_processed), provider, provider_version, job_scope),
                )
                selected_id = job_id
            else:
                selected_id = row[0]
                if row[1] in ("queued", "processing") and (
                    row[4] != provider or row[5] != provider_version
                ):
                    raise PersistenceIntegrityError(
                        "another story-extraction provider already owns the active job"
                    )
                if row[1] == "failed" and retry:
                    connection.execute(
                        "UPDATE graph_jobs SET status='queued',attempts=0,error=NULL,total_chunks=?,provider=?,provider_version=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                        (total_chunks, provider, provider_version, selected_id),
                    )
                elif row[1] == "cancelled":
                    connection.execute(
                        "UPDATE graph_jobs SET status='queued',total_chunks=?,provider=?,provider_version=?,job_scope=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                        (total_chunks, provider, provider_version, job_scope, selected_id),
                    )
            result = connection.execute(
                "SELECT job_id,book_id,content_version_id,boundary_section,boundary_unit,boundary_source_unit_id,total_chunks,processed_chunks,status,attempts,error,updated_at,provider,provider_version,job_scope FROM graph_jobs WHERE job_id=?",
                (selected_id,),
            ).fetchone()
            connection.execute(
                "UPDATE book_preparation_jobs SET graph_status=?,graph_completed=?,graph_total=?,graph_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                ("complete" if result[8] == "completed" else "queued", result[7], total_chunks, book_id, content_version_id),
            )
        return _graph_job_dict(result)

    def graph_job(self, book_id: str, content_version_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT job_id,book_id,content_version_id,boundary_section,boundary_unit,boundary_source_unit_id,total_chunks,processed_chunks,status,attempts,error,updated_at,provider,provider_version,job_scope FROM graph_jobs WHERE book_id=? AND content_version_id=? ORDER BY updated_at DESC, rowid DESC LIMIT 1",
                (book_id, content_version_id),
            ).fetchone()
        return _graph_job_dict(row) if row else None

    def graph_job_by_id(self, job_id: str) -> dict[str, object] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT job_id,book_id,content_version_id,boundary_section,boundary_unit,boundary_source_unit_id,total_chunks,processed_chunks,status,attempts,error,updated_at,provider,provider_version,job_scope FROM graph_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
        return _graph_job_dict(row) if row else None

    def runnable_graph_jobs(self) -> tuple[dict[str, object], ...]:
        with self._lock:
            rows = self._connection.execute(
                """SELECT j.job_id,j.book_id,j.content_version_id,j.boundary_section,
                j.boundary_unit,j.boundary_source_unit_id,j.total_chunks,j.processed_chunks,
                j.status,j.attempts,j.error,j.updated_at,j.provider,j.provider_version,j.job_scope
                FROM graph_jobs AS j
                WHERE j.status='queued' AND j.provider='ollama'
                AND COALESCE((SELECT p.graph_rag_enabled FROM book_preparation_jobs AS p
                    WHERE p.book_id=j.book_id AND p.content_version_id=j.content_version_id),1)=1
                ORDER BY j.updated_at,j.rowid"""
            ).fetchall()
        return tuple(_graph_job_dict(row) for row in rows)

    def processed_graph_chunk_ids(self, book_id: str, content_version_id: str) -> set[str]:
        with self._lock:
            return {row[0] for row in self._connection.execute(
                "SELECT chunk_id FROM graph_processed_chunks WHERE book_id=? AND content_version_id=?",
                (book_id, content_version_id),
            )}

    def claim_graph_batch(self, job_id: str, chunk_ids: Sequence[str]) -> bool:
        with self._lock, self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE graph_jobs SET status='processing',attempts=attempts+1,current_chunk_ids=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND status='queued'",
                (_json(list(chunk_ids)), job_id),
            )
            if cursor.rowcount == 1:
                job = connection.execute(
                    "SELECT book_id,content_version_id,processed_chunks,total_chunks FROM graph_jobs WHERE job_id=?",
                    (job_id,),
                ).fetchone()
                if job:
                    connection.execute(
                        "UPDATE book_preparation_jobs SET graph_status='processing',graph_completed=?,graph_total=?,graph_error=NULL WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                        (job[2], job[3], job[0], job[1]),
                    )
            return cursor.rowcount == 1

    def commit_graph_batch(
        self, graph: StoryGraph, *, job_id: str, chunk_ids: Sequence[str],
        complete: bool, max_attempts: int,
    ) -> None:
        with self._lock, self._transaction() as connection:
            current = connection.execute(
                "SELECT content_version_id FROM books WHERE book_id=?", (graph.book_id,)
            ).fetchone()
            if current is None or current[0] != graph.content_version_id:
                connection.execute(
                    "UPDATE graph_jobs SET status='cancelled',error='content version changed',updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                    (job_id,),
                )
                return
            self._validate_graph(graph, connection)
            connection.execute(
                "INSERT INTO graph_snapshots(book_id,content_version_id,payload) VALUES(?,?,?) ON CONFLICT(book_id) DO UPDATE SET content_version_id=excluded.content_version_id,payload=excluded.payload",
                (graph.book_id, graph.content_version_id, _json(_graph_to_dict(graph))),
            )
            connection.executemany(
                "INSERT OR IGNORE INTO graph_processed_chunks(book_id,content_version_id,chunk_id) VALUES(?,?,?)",
                [(graph.book_id, graph.content_version_id, chunk_id) for chunk_id in chunk_ids],
            )
            job = connection.execute(
                "SELECT attempts,total_chunks,processed_chunks FROM graph_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if job is None:
                raise PersistenceIntegrityError("graph job disappeared before batch commit")
            status = "completed" if complete else "queued"
            connection.execute(
                "UPDATE graph_jobs SET status=?,processed_chunks=?,current_chunk_ids='[]',attempts=0,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                (status, job[1] if complete else min(job[1], job[2] + len(chunk_ids)), job_id),
            )
            connection.execute(
                "UPDATE book_preparation_jobs SET graph_status=?,graph_completed=?,graph_total=?,graph_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                ("complete" if complete else "queued", job[1] if complete else min(job[1], job[2] + len(chunk_ids)), job[1], graph.book_id, graph.content_version_id),
            )

    def fail_graph_batch(self, job_id: str, error: str, *, max_attempts: int) -> None:
        with self._lock, self._transaction() as connection:
            row = connection.execute(
                "SELECT attempts FROM graph_jobs WHERE job_id=?", (job_id,)
            ).fetchone()
            if row is None:
                return
            status = "queued" if row[0] < max_attempts else "failed"
            connection.execute(
                "UPDATE graph_jobs SET status=?,current_chunk_ids='[]',error=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                (status, error[:1000], job_id),
            )
            job = connection.execute(
                "SELECT book_id,content_version_id,processed_chunks,total_chunks FROM graph_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            if job:
                connection.execute(
                    "UPDATE book_preparation_jobs SET graph_status=?,graph_completed=?,graph_total=?,graph_error=?,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                    ("failed" if status == "failed" else "queued", job[2], job[3], error[:1000], job[0], job[1]),
                )

    def complete_graph_job(self, job_id: str) -> None:
        with self._lock, self._transaction() as connection:
            job = connection.execute(
                "SELECT book_id,content_version_id,total_chunks FROM graph_jobs WHERE job_id=?",
                (job_id,),
            ).fetchone()
            connection.execute(
                "UPDATE graph_jobs SET status='completed',processed_chunks=total_chunks,current_chunk_ids='[]',attempts=0,error=NULL,updated_at=CURRENT_TIMESTAMP WHERE job_id=? AND status IN ('queued','processing')",
                (job_id,),
            )
            if job:
                connection.execute(
                    "UPDATE book_preparation_jobs SET graph_status='complete',graph_completed=?,graph_total=?,graph_error=NULL,updated_at=CURRENT_TIMESTAMP WHERE book_id=? AND content_version_id=? AND graph_rag_enabled=1",
                    (job[2], job[2], job[0], job[1]),
                )

    def cancel_graph_job(self, job_id: str, reason: str) -> None:
        with self._lock, self._transaction() as connection:
            connection.execute(
                "UPDATE graph_jobs SET status='cancelled',current_chunk_ids='[]',error=?,updated_at=CURRENT_TIMESTAMP WHERE job_id=?",
                (reason[:1000], job_id),
            )

    def load_all(self) -> tuple[PersistedBook, ...]:
        with self._lock:
            books: list[PersistedBook] = []
            for book_id, version in self._connection.execute(
                "SELECT book_id, content_version_id FROM books ORDER BY book_id"
            ):
                chunks = tuple(
                    _chunk_from_dict(json.loads(payload))
                    for (payload,) in self._connection.execute(
                        "SELECT payload FROM chunks WHERE book_id = ? ORDER BY section_index, chunk_index, chunk_id",
                        (book_id,),
                    )
                )
                graph_row = self._connection.execute(
                    "SELECT content_version_id, payload FROM graph_snapshots WHERE book_id = ?",
                    (book_id,),
                ).fetchone()
                graph = (
                    _graph_from_dict(json.loads(graph_row[1]))
                    if graph_row and graph_row[0] == version
                    else StoryGraph(book_id, version)
                )
                self._validate_graph(graph, self._connection)
                vector_rows = self._connection.execute(
                    "SELECT chunk_id, provider_identity, dimensions, vector_json FROM semantic_vectors WHERE book_id = ? AND content_version_id = ? ORDER BY chunk_id",
                    (book_id, version),
                ).fetchall()
                identities = {(row[1], row[2]) for row in vector_rows}
                if len(identities) > 1:
                    raise PersistenceIntegrityError(
                        f"stored semantic vectors for {book_id!r} have mixed model identities or dimensions"
                    )
                vectors = {row[0]: tuple(json.loads(row[3])) for row in vector_rows}
                identity, dimensions = next(iter(identities), (None, None))
                books.append(
                    PersistedBook(book_id, version, chunks, graph, vectors, identity, dimensions)
                )
                self._cache[book_id] = (version, chunks)
            return tuple(books)

    @property
    def book_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                row[0]
                for row in self._connection.execute("SELECT book_id FROM books ORDER BY book_id")
            )

    def get_book(self, book_id: str) -> tuple[str, tuple[BookChunk, ...]] | None:
        with self._lock:
            cached = self._cache.get(book_id)
            if cached is not None:
                return cached
            row = self._connection.execute(
                "SELECT content_version_id FROM books WHERE book_id = ?", (book_id,)
            ).fetchone()
            if row is None:
                return None
            chunks = tuple(
                _chunk_from_dict(json.loads(payload))
                for (payload,) in self._connection.execute(
                    "SELECT payload FROM chunks WHERE book_id = ? ORDER BY section_index, chunk_index, chunk_id",
                    (book_id,),
                )
            )
            stored = (row[0], chunks)
            self._cache[book_id] = stored
            return stored

    def get_semantic_vectors(
        self, book_id: str
    ) -> tuple[str | None, int | None, dict[str, tuple[float, ...]]]:
        """Return the currently committed vectors and their model identity."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT provider_identity, dimensions, chunk_id, vector_json "
                "FROM semantic_vectors WHERE book_id = ? ORDER BY chunk_id",
                (book_id,),
            ).fetchall()
            identities = {(row[0], row[1]) for row in rows}
            if len(identities) > 1:
                raise PersistenceIntegrityError(
                    f"stored semantic vectors for {book_id!r} have mixed model identities or dimensions"
                )
            identity, dimensions = next(iter(identities), (None, None))
            vectors = {row[2]: tuple(json.loads(row[3])) for row in rows}
            return identity, dimensions, vectors

    def _validate_graph(self, graph: StoryGraph, connection: sqlite3.Connection) -> None:
        chunk_rows = connection.execute(
            "SELECT chunk_id, payload FROM chunks WHERE book_id = ? AND content_version_id = ?",
            (graph.book_id, graph.content_version_id),
        ).fetchall()
        chunks = {
            chunk_id: _chunk_from_dict(json.loads(payload))
            for chunk_id, payload in chunk_rows
        }
        entity_ids = {entity.id for entity in graph.entities}
        for fact in (*graph.entities, *graph.events, *graph.relationships):
            if fact.book_id != graph.book_id or fact.content_version_id != graph.content_version_id:
                raise PersistenceIntegrityError("graph fact has stale book/content-version identity")
            for evidence in fact.evidence:
                if evidence.content_version_id != graph.content_version_id:
                    raise PersistenceIntegrityError("graph evidence has stale content version")
                chunk = chunks.get(evidence.source_chunk_id)
                if chunk is None or not set(evidence.source_unit_ids).issubset(
                    set(chunk.source_unit_ids)
                ):
                    raise PersistenceIntegrityError(
                        f"graph evidence references missing canonical source {evidence.source_chunk_id!r}"
                    )
                positions_by_id = {
                    position.source_unit_id: position for position in chunk.source_positions
                }
                if set(evidence.source_unit_ids) != {
                    position.source_unit_id for position in evidence.source_positions
                }:
                    raise PersistenceIntegrityError(
                        "graph evidence source-unit IDs do not match its location records"
                    )
                for source_position in evidence.source_positions:
                    canonical = positions_by_id.get(source_position.source_unit_id)
                    if canonical is None or (
                        source_position.unit_index,
                        source_position.kind,
                        source_position.href,
                        source_position.cfi,
                    ) != (
                        canonical.unit_index,
                        canonical.kind,
                        canonical.href,
                        canonical.cfi,
                    ):
                        raise PersistenceIntegrityError(
                            "graph source location metadata does not match its canonical chunk"
                        )
                if (
                    evidence.book_id != graph.book_id
                    or evidence.section_index != chunk.section_index
                    or evidence.chunk_index != chunk.chunk_index
                    or evidence.content_version_id != chunk.content_version_id
                ):
                    raise PersistenceIntegrityError("graph evidence provenance does not match its canonical chunk")
        if any(not set(event.participant_entity_ids).issubset(entity_ids) for event in graph.events):
            raise PersistenceIntegrityError("graph event references a missing entity")
        if any(
            relation.subject_entity_id not in entity_ids
            or relation.object_entity_id not in entity_ids
            for relation in graph.relationships
        ):
            raise PersistenceIntegrityError("graph relationship references a missing entity")

    def _transaction(self):
        return _Transaction(self._connection)

    def close(self) -> None:
        with self._lock:
            self._connection.close()


class _Transaction:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection

    def __enter__(self) -> sqlite3.Connection:
        self.connection.execute("BEGIN IMMEDIATE")
        return self.connection

    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type:
            self.connection.execute("ROLLBACK")
        else:
            try:
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
        return False


def default_database_path() -> Path:
    """Return the configurable local database path (no secrets are stored)."""
    import os

    configured = os.environ.get("LECTOR_RAG_DB_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parent.parent / "data" / "lector.sqlite3"


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _graph_job_dict(row: tuple[object, ...]) -> dict[str, object]:
    job_scope = row[14] if len(row) > 14 else "confirmed_boundary"
    full_book = job_scope == "full_book"
    return {
        "job_id": row[0], "book_id": row[1], "content_version_id": row[2],
        "boundary_section": None if full_book else row[3],
        "boundary_unit": None if full_book else row[4],
        "boundary_source_unit_id": None if full_book else row[5], "total_chunks": row[6],
        "processed_chunks": row[7], "status": row[8], "attempts": row[9],
        "error": row[10], "updated_at": row[11],
        "provider": row[12] if len(row) > 12 else "ollama",
        "provider_version": row[13] if len(row) > 13 else "",
        "job_scope": job_scope,
    }


def _source_chunk_ids(value: object) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        chunk_id = value.get("source_chunk_id")
        if isinstance(chunk_id, str):
            found.add(chunk_id)
        for child in value.values():
            found.update(_source_chunk_ids(child))
    elif isinstance(value, list):
        for child in value:
            found.update(_source_chunk_ids(child))
    return found


def _chunk_from_dict(value: dict) -> BookChunk:
    value["source_unit_ids"] = tuple(value["source_unit_ids"])
    value["source_positions"] = tuple(ChunkSourcePosition(**item) for item in value["source_positions"])
    return BookChunk(**value)


def _graph_to_dict(graph: StoryGraph) -> dict:
    return asdict(graph)


def _graph_from_dict(value: dict) -> StoryGraph:
    def evidence(item: dict) -> GraphEvidence:
        item["source_unit_ids"] = tuple(item["source_unit_ids"])
        item["source_positions"] = tuple(GraphSourcePosition(**pos) for pos in item["source_positions"])
        return GraphEvidence(**item)

    entities = tuple(
        StoryEntity(
            **{
                **item,
                "aliases": tuple(
                    StoryAlias(name=alias["name"], evidence=tuple(evidence(e) for e in alias["evidence"]))
                    for alias in item["aliases"]
                ),
                "evidence": tuple(evidence(e) for e in item["evidence"]),
            }
        )
        for item in value["entities"]
    )
    events = tuple(
        StoryEvent(
            **{
                **item,
                "participant_entity_ids": tuple(item["participant_entity_ids"]),
                "evidence": tuple(evidence(e) for e in item["evidence"]),
            }
        )
        for item in value["events"]
    )
    relationships = tuple(
        StoryRelationship(
            **{
                **item,
                "temporal_metadata": tuple(tuple(pair) for pair in item["temporal_metadata"]),
                "evidence": tuple(evidence(e) for e in item["evidence"]),
            }
        )
        for item in value["relationships"]
    )
    return StoryGraph(value["book_id"], value["content_version_id"], entities, events, relationships)
