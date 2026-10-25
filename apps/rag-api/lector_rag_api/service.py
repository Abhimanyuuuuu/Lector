"""Transactional ingestion and retrieval orchestration over a SQLite source store."""

from __future__ import annotations
import logging
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from threading import Event, Lock, RLock, Thread
from typing import Callable, Mapping, Sequence

from .chunking import BookChunk, chunk_source_units
from .hybrid import DEFAULT_HYBRID_CONFIG, HybridConfig, fuse_results
from .graph_retrieval import (
    DEFAULT_GRAPH_OPTIONS,
    GraphRetrievalOptions,
    expand_hybrid_results,
)
from .keyword import BM25Index
from .retrieval_types import ReadingBoundary, RetrievalResult, is_chunk_eligible
from .schemas import SourceUnitInput
from .semantic import EmbeddingProvider, SemanticIndex
from .persistence import ContentVersionConflictError, PersistenceIntegrityError, SQLiteChunkStore
from .story_graph import (
    GraphStore,
    StoryEntity,
    StoryEvent,
    StoryExtraction,
    StoryExtractor,
    StoryGraph,
    StoryRelationship,
    GraphEvidence,
    GraphSourcePosition,
    StoryAlias,
)
from .booknlp_graph import (
    BOOKNLP_GRAPH_VERSION,
    BookNLPRunner,
    BookNLPUnavailableError,
    assemble_booknlp_input,
    bundle_to_json,
    categorize_candidate_verbs,
    graph_from_artifacts,
)

logger = logging.getLogger(__name__)
class BookNotIndexedError(LookupError):
    pass


class ContentVersionMismatchError(LookupError):
    pass


class InvalidReadingBoundaryError(ValueError):
    pass


class EmbeddingIndexError(RuntimeError):
    pass


class StoryExtractionUnavailableError(RuntimeError):
    pass


class StoryExtractionProviderConflictError(RuntimeError):
    pass


class StoryGraphUnavailableError(RuntimeError):
    pass


class GraphRagDisabledError(RuntimeError):
    pass


class InMemoryRagService:
    """Owns canonical chunks and rebuilt indexes; name retained for compatibility.

    Use ``database_path`` for a persistent SQLite store. The default ``:memory:``
    keeps unit tests and explicitly ephemeral embeddings isolated.
    """

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider | None = None,
        story_extractor: StoryExtractor | None = None,
        story_extractors: Mapping[str, StoryExtractor] | None = None,
        booknlp_runner: BookNLPRunner | None = None,
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        database_path: str | Path = ":memory:",
    ) -> None:
        self.store = SQLiteChunkStore(database_path)
        self.keyword_index = BM25Index(k1=bm25_k1, b=bm25_b)
        self.semantic_index = SemanticIndex(embedding_provider)
        # Legacy generated graph snapshots are loaded for preservation only.
        # No Ollama/Groq StoryExtractor is retained as an active graph path.
        self.story_extractors: dict[str, StoryExtractor] = {}
        self.story_extractor = None
        self.booknlp_runner = booknlp_runner
        self.story_graph = GraphStore()
        self._lock = RLock()
        self._graph_worker_event = Event()
        self._graph_worker_stop = Event()
        self._graph_worker_guard = Lock()
        self._graph_worker_thread: Thread | None = None
        self._booknlp_worker_thread: Thread | None = None
        self._graph_batch_size = _positive_env_int("LECTOR_GRAPH_BATCH_SIZE", 2)
        self._graph_batch_max_attempts = _positive_env_int("LECTOR_GRAPH_BATCH_MAX_ATTEMPTS", 2)
        self._semantic_storage_errors: dict[str, str] = {}
        try:
            persisted_books = self.store.load_all()
        except Exception:
            self.store.close()
            raise
        for persisted in persisted_books:
            prepared_keyword = self.keyword_index.prepare_book(
                persisted.book_id, persisted.content_version_id, persisted.chunks
            )
            self.keyword_index.commit_book(prepared_keyword)
            self.story_graph.commit(persisted.graph)
            if embedding_provider is None:
                self.semantic_index.commit_book(
                    self.semantic_index.prepare_book(
                        persisted.book_id, persisted.content_version_id, persisted.chunks
                    )
                )
            elif persisted.vectors:
                provider_identity = _embedding_identity(embedding_provider)
                expected_vector_ids = {
                    chunk.id for chunk in persisted.chunks if chunk.text.strip()
                }
                if (
                    persisted.embedding_identity != provider_identity
                    or persisted.embedding_dimensions != embedding_provider.dimensions
                ):
                    self._semantic_storage_errors[persisted.book_id] = (
                        "stored embeddings use a different provider/model or dimension; "
                        "re-ingest the book with the configured embedding provider"
                    )
                elif set(persisted.vectors) != expected_vector_ids:
                    self._semantic_storage_errors[persisted.book_id] = (
                        "stored embeddings are incomplete for the canonical chunks; "
                        "re-ingest the book with the configured embedding provider"
                    )
                else:
                    vectors = [
                        persisted.vectors.get(chunk.id) for chunk in persisted.chunks
                    ]
                    try:
                        prepared_semantic = self.semantic_index.prepare_book_with_embeddings(
                            persisted.book_id,
                            persisted.content_version_id,
                            persisted.chunks,
                            vectors,
                        )
                        self.semantic_index.commit_book(prepared_semantic)
                    except (TypeError, ValueError) as exc:
                        self._semantic_storage_errors[persisted.book_id] = (
                            f"stored embeddings are invalid ({exc}); re-ingest the book"
                        )
            else:
                self._semantic_storage_errors[persisted.book_id] = (
                    "no stored embeddings are available; re-ingest the book with an "
                    "explicitly configured embedding provider"
                )

    def schedule_story_graph(
        self, *, book_id: str, reading_boundary: ReadingBoundary, retry: bool = False,
    ) -> dict[str, object]:
        """Compatibility route: graph construction is now full-book BookNLP."""
        with self._lock:
            stored = self.store.get_book(book_id)
            if stored is None:
                raise BookNotIndexedError(f"book {book_id!r} is not indexed")
            self._validate_book_boundary(book_id, reading_boundary)
            version = stored[0]
        return self.schedule_full_book_story_graph(
            book_id=book_id, content_version_id=version, retry=retry,
        )

    def schedule_full_book_story_graph(
        self, *, book_id: str, content_version_id: str, retry: bool = False,
    ) -> dict[str, object]:
        """Schedule local graph construction over the entire indexed book.

        This processing scope is deliberately separate from the reader's
        confirmed boundary. The complete graph is stored locally; retrieval
        filters every fact against the user's confirmed boundary.
        """
        with self._lock:
            if self.store.graph_rag_enabled(book_id, content_version_id) is not True:
                raise GraphRagDisabledError("Graph RAG is disabled for this book.")
            if self.booknlp_runner is None:
                raise StoryExtractionUnavailableError(
                    "Graph RAG is enabled, but local BookNLP is not configured. Install BookNLP and its model assets locally."
                )
            stored = self.store.get_book(book_id)
            if stored is None or stored[0] != content_version_id:
                raise ContentVersionMismatchError("book content version changed before graph preparation")
            chunks = stored[1]
            if not chunks:
                raise BookNotIndexedError("book has no canonical chunks for graph preparation")
            job = self.store.schedule_booknlp_job(
                book_id=book_id, content_version_id=content_version_id,
                total_chunks=len(chunks), extraction_version=self.booknlp_runner.version,
                retry=retry,
            )
        self.start_booknlp_worker()
        return self.graph_job_status(book_id=book_id, content_version_id=content_version_id) or job

    def start_graph_preparation_if_enabled(self, book_id: str, content_version_id: str) -> dict[str, object] | None:
        status = self.store.book_preparation_status(book_id, content_version_id)
        if not status or not status["graph_rag_enabled"]:
            return None
        try:
            return self.schedule_full_book_story_graph(book_id=book_id, content_version_id=content_version_id)
        except GraphRagDisabledError:
            raise
        except StoryExtractionUnavailableError as exc:
            existing = self.store.booknlp_job(book_id, content_version_id)
            if existing is not None:
                self.store.set_booknlp_job_status(
                    book_id=book_id, content_version_id=content_version_id,
                    status="unavailable", stage="unavailable", error=str(exc),
                )
            else:
                self.store.update_book_preparation(
                    book_id, content_version_id, graph_status="unavailable", graph_error=str(exc),
                )
            logger.warning("Graph preparation unavailable book=%s version=%s reason=%s", book_id, content_version_id, exc)
            return None

    def graph_job_status(
        self, *, book_id: str, content_version_id: str
    ) -> dict[str, object] | None:
        job = self.store.booknlp_job(book_id, content_version_id)
        if job is None:
            return self.store.graph_job(book_id, content_version_id)
        return {
            "job_id": hashlib.sha256(f"{book_id}\0{content_version_id}\0booknlp".encode()).hexdigest()[:32],
            "book_id": book_id, "content_version_id": content_version_id,
            "boundary_section": None, "boundary_unit": None,
            "boundary_source_unit_id": None,
            "total_chunks": job["total_chunks"], "processed_chunks": job["completed_chunks"],
            "status": "completed" if job["status"] == "complete" else job["status"],
            "attempts": 0, "error": job["error"], "updated_at": job["updated_at"],
            "provider": "booknlp", "provider_version": job["extraction_version"],
            "job_scope": "full_book",
        }

    def start_booknlp_worker(self) -> None:
        with self._graph_worker_guard:
            if self._booknlp_worker_thread is None or not self._booknlp_worker_thread.is_alive():
                self._booknlp_worker_thread = Thread(
                    target=self._run_booknlp_worker, name="lector-booknlp-worker", daemon=True
                )
                self._booknlp_worker_thread.start()

    def _run_booknlp_worker(self) -> None:
        """Run one whole-book linguistic analysis, then persist in chunk batches."""
        while True:
            rebuilds = self.store.pending_booknlp_rebuild_jobs()
            if rebuilds:
                self._run_booknlp_rebuild(rebuilds[0])
                continue
            jobs = self.store.pending_booknlp_jobs()
            if not jobs:
                return
            item = jobs[0]
            book_id = str(item["book_id"])
            version = str(item["content_version_id"])
            try:
                if self.booknlp_runner is None:
                    raise BookNLPUnavailableError("BookNLP is not configured in this backend.")
                stored = self.store.get_book(book_id)
                status = self.store.booknlp_job(book_id, version)
                if stored is None or stored[0] != version:
                    self.store.set_booknlp_job_status(
                        book_id=book_id, content_version_id=version,
                        status="cancelled", stage="cancelled", error="content version changed",
                    )
                    continue
                if self.store.graph_rag_enabled(book_id, version) is False:
                    self.store.set_booknlp_job_status(
                        book_id=book_id, content_version_id=version,
                        status="cancelled", stage="cancelled", error="Graph RAG disabled for this book.",
                    )
                    continue
                chunks = tuple(sorted(stored[1], key=lambda c: (c.section_index, c.chunk_index, c.id)))
                if status is None:
                    continue
                bundle = status.get("bundle_json")
                if isinstance(bundle, str) and bundle:
                    bundle_data = json.loads(bundle)
                else:
                    self.store.set_booknlp_job_status(
                        book_id=book_id, content_version_id=version,
                        status="processing", stage="book_analysis",
                    )
                    text, _ = assemble_booknlp_input(chunks)
                    artifacts = self.booknlp_runner.process(text, book_id)
                    parsed = graph_from_artifacts(
                        book_id=book_id, content_version_id=version, chunks=chunks,
                        book_data=artifacts["book"], entities=artifacts["entities"],
                        tokens=artifacts["tokens"], extraction_version=self.booknlp_runner.version,
                    )
                    bundle_data = bundle_to_json(parsed)
                    self.store.persist_booknlp_bundle(
                        book_id=book_id, content_version_id=version,
                        bundle=bundle_data, total_chunks=len(chunks),
                    )
                job = self.store.booknlp_job(book_id, version)
                done = int(job["completed_chunks"]) if job else 0
                for start in range(done, len(chunks), self._graph_batch_size):
                    current = self.store.get_book(book_id)
                    if current is None or current[0] != version:
                        raise ContentVersionMismatchError("book content version changed during BookNLP assembly")
                    batch = chunks[start : start + self._graph_batch_size]
                    next_done = min(start + len(batch), len(chunks))
                    self.store.commit_booknlp_batch(
                        book_id=book_id, content_version_id=version,
                        bundle=bundle_data, chunk_ids=[chunk.id for chunk in batch],
                        completed_chunks=next_done, total_chunks=len(chunks),
                        complete=next_done == len(chunks),
                    )
                if not chunks:
                    self.store.set_booknlp_job_status(
                        book_id=book_id, content_version_id=version,
                        status="complete", stage="complete",
                    )
            except Exception as exc:
                logger.exception("BookNLP graph preparation failed book=%s version=%s", book_id, version)
                self.store.set_booknlp_job_status(
                    book_id=book_id, content_version_id=version,
                    status=("unavailable" if isinstance(exc, BookNLPUnavailableError) else "failed"),
                    stage=("unavailable" if isinstance(exc, BookNLPUnavailableError) else "failed"),
                    error=_safe_graph_error(exc),
                )

    def _run_booknlp_rebuild(self, item: Mapping[str, object]) -> None:
        book_id = str(item["book_id"])
        version = str(item["content_version_id"])
        try:
            if self.booknlp_runner is None:
                raise BookNLPUnavailableError("BookNLP is not configured in this backend.")
            stored = self.store.get_book(book_id)
            job = self.store.booknlp_rebuild_job(book_id, version)
            if stored is None or stored[0] != version or job is None:
                self.store.set_booknlp_rebuild_status(
                    book_id=book_id, content_version_id=version,
                    status="cancelled", error="content version changed",
                )
                return
            if self.store.graph_rag_enabled(book_id, version) is not True:
                self.store.set_booknlp_rebuild_status(
                    book_id=book_id, content_version_id=version,
                    status="cancelled", error="Graph RAG is disabled for this book.",
                )
                return
            self.store.set_booknlp_rebuild_status(
                book_id=book_id, content_version_id=version, status="processing"
            )
            chunks = tuple(sorted(stored[1], key=lambda chunk: (chunk.section_index, chunk.chunk_index, chunk.id)))
            text, _ = assemble_booknlp_input(chunks)
            artifacts = self.booknlp_runner.process(text, book_id)
            bundle = bundle_to_json(graph_from_artifacts(
                book_id=book_id, content_version_id=version, chunks=chunks,
                book_data=artifacts["book"], entities=artifacts["entities"],
                tokens=artifacts["tokens"], extraction_version=self.booknlp_runner.version,
            ))
            if not bundle.get("profiles") or not bundle.get("mentions") or not bundle.get("edges"):
                raise ValueError(
                    "BookNLP produced no usable source-grounded relationship graph; the previously committed graph was preserved."
                )
            # The expensive inference is complete before any live graph row is
            # touched. The store validates and swaps all graph tables atomically.
            self.store.commit_booknlp_rebuild(
                book_id=book_id, content_version_id=version,
                bundle=bundle, total_chunks=len(chunks),
            )
            logger.info(
                "BookNLP graph rebuild committed book=%s version=%s chunks=%d profiles=%d edges=%d",
                book_id, version, len(chunks), len(bundle.get("profiles", [])),
                len(bundle.get("edges", [])),
            )
        except Exception as exc:
            logger.exception("BookNLP graph rebuild failed book=%s version=%s", book_id, version)
            self.store.set_booknlp_rebuild_status(
                book_id=book_id, content_version_id=version,
                status="failed", error=_safe_graph_error(exc),
            )

    def schedule_booknlp_graph_rebuild(
        self, *, book_id: str, content_version_id: str,
    ) -> dict[str, object]:
        if self.booknlp_runner is None:
            raise StoryExtractionUnavailableError("BookNLP is not configured in this backend.")
        if self.store.graph_rag_enabled(book_id, content_version_id) is not True:
            raise GraphRagDisabledError("Graph RAG is disabled for this book.")
        stored = self.store.get_book(book_id)
        if stored is None or stored[0] != content_version_id:
            raise ContentVersionMismatchError("book content version changed before graph rebuild")
        if not stored[1]:
            raise BookNotIndexedError("book has no canonical chunks for graph rebuild")
        self.store.schedule_booknlp_rebuild(
            book_id=book_id, content_version_id=content_version_id,
            total_chunks=len(stored[1]), extraction_version=self.booknlp_runner.version,
        )
        self.start_booknlp_worker()
        return self.booknlp_rebuild_status(book_id=book_id, content_version_id=content_version_id) or {}

    def booknlp_rebuild_status(
        self, *, book_id: str, content_version_id: str,
    ) -> dict[str, object] | None:
        job = self.store.booknlp_rebuild_job(book_id, content_version_id)
        if job is None:
            return None
        return {
            "job_id": hashlib.sha256(f"{book_id}\0{content_version_id}\0booknlp-rebuild".encode()).hexdigest()[:32],
            "book_id": book_id, "content_version_id": content_version_id,
            "boundary_section": None, "boundary_unit": None,
            "boundary_source_unit_id": None,
            "total_chunks": job["total_chunks"],
            "processed_chunks": job["completed_chunks"],
            "status": "completed" if job["status"] == "complete" else job["status"],
            "attempts": 0, "error": job["error"], "updated_at": job["updated_at"],
            "provider": "booknlp", "provider_version": job["extraction_version"],
            "job_scope": "full_book",
        }

    def start_graph_worker(self) -> None:
        """Retired generated-extractor worker; BookNLP owns new graph work."""
        raise StoryExtractionUnavailableError(
            "The generated story-extraction worker was retired; use local BookNLP graph preparation."
        )

    def resume_graph_jobs(self) -> None:
        # Only explicitly created BookNLP jobs resume. Legacy Ollama graph
        # snapshots/jobs remain preserved but are never re-run or re-labelled.
        if self.booknlp_runner is not None and (
            self.store.pending_booknlp_jobs() or self.store.pending_booknlp_rebuild_jobs()
        ):
            self.start_booknlp_worker()

    def _run_graph_worker(self) -> None:
        while not self._graph_worker_stop.is_set():
            jobs = self.store.runnable_graph_jobs()
            if not jobs:
                self._graph_worker_event.clear()
                if not self.store.runnable_graph_jobs():
                    self._graph_worker_event.wait(timeout=30)
                continue
            job = jobs[0]
            job_id = str(job["job_id"])
            book_id = str(job["book_id"])
            version = str(job["content_version_id"])
            provider = str(job.get("provider", "ollama"))
            provider_version = str(job.get("provider_version", ""))
            is_full_book = job.get("job_scope") == "full_book"
            boundary = None if is_full_book else ReadingBoundary(
                book_id=book_id,
                section_index=int(job["boundary_section"]),
                unit_index=int(job["boundary_unit"]),
                source_unit_id=str(job["boundary_source_unit_id"]),
            )
            started = time.monotonic()
            batch_number = int(job["processed_chunks"]) // self._graph_batch_size + 1
            batch_count = 0
            try:
                stored = self.store.get_book(book_id)
                if stored is None or stored[0] != version:
                    self.store.cancel_graph_job(job_id, "content version changed")
                    continue
                if self.store.graph_rag_enabled(book_id, version) is False:
                    self.store.cancel_graph_job(job_id, "Graph RAG disabled for this book.")
                    continue
                if not is_full_book:
                    assert boundary is not None
                    self._validate_book_boundary(book_id, boundary)
                processed = self.store.processed_graph_chunk_ids(book_id, version)
                eligible = tuple(
                    chunk for chunk in stored[1]
                    if (is_full_book or is_chunk_eligible(chunk, boundary)) and chunk.id not in processed
                )
                if not eligible:
                    self.store.complete_graph_job(job_id)
                    continue
                batch = eligible[: self._graph_batch_size]
                batch_count = len(batch)
                if not self.store.claim_graph_batch(job_id, [chunk.id for chunk in batch]):
                    continue
                self._extract_and_commit_graph_batch(
                    job_id, book_id, version, batch, provider, provider_version
                )
                logger.info(
                    "Graph batch committed book=%s version=%s provider=%s model=%s batch=%d chunks=%d input_chars=%d elapsed_seconds=%.2f outcome=success",
                    book_id, version, provider, provider_version, batch_number,
                    batch_count, sum(len(chunk.text) for chunk in batch),
                    time.monotonic() - started,
                )
            except Exception as exc:
                logger.exception(
                    "Graph batch failed book=%s version=%s provider=%s model=%s batch=%d chunks=%d input_chars=%d elapsed_seconds=%.2f outcome=%s",
                    book_id, version, provider, provider_version, batch_number,
                    batch_count, sum(len(chunk.text) for chunk in batch) if batch_count else 0,
                    time.monotonic() - started, type(exc).__name__,
                )
                self.store.fail_graph_batch(
                    job_id,
                    _safe_graph_error(exc),
                    max_attempts=(
                        1
                        if getattr(exc, "retryable", True) is False
                        else self._graph_batch_max_attempts
                    ),
                )

    def _extract_and_commit_graph_batch(
        self, job_id: str, book_id: str, version: str, batch: Sequence[BookChunk],
        provider: str | None = None, provider_version: str = "",
    ) -> None:
        extractor = self.story_extractors.get(provider or "")
        if extractor is None and not provider_version and self.story_extractor is not None:
            # Compatibility for test/custom extractors and pre-provider jobs
            # created before job records captured model identity.
            extractor = self.story_extractor
        if extractor is None and provider is None:
            extractor = self.story_extractor
        if extractor is None:
            raise StoryExtractionUnavailableError(
                f"the {provider or 'selected'} story-extraction provider is not configured"
            )
        if provider_version and extractor.version != provider_version:
            raise StoryExtractionUnavailableError(
                "the extractor version captured by this job is unavailable; explicitly retry with a configured provider"
            )
        # The potentially slow model call is deliberately outside the service
        # lock and outside any SQLite transaction.
        extraction = extractor.extract(batch)
        if not isinstance(extraction, (StoryExtraction, dict)):
            raise TypeError("story extractor must return a validated extraction object")
        with self._lock:
            current = self.store.get_book(book_id)
            if current is None or current[0] != version:
                self.store.cancel_graph_job(job_id, "content version changed during extraction")
                return
            if self.store.graph_rag_enabled(book_id, version) is False:
                self.store.cancel_graph_job(job_id, "Graph RAG disabled for this book.")
                return
            prepared = self.story_graph.prepare_update(
                book_id=book_id, content_version_id=version, chunks=batch,
                extraction=extraction, extractor_name=extractor.name,
                extractor_version=extractor.version,
            )
            processed = self.store.processed_graph_chunk_ids(book_id, version)
            # The current chunk IDs are persisted in the claimed job. Determine
            # completion against the fixed boundary, not the full book.
            status = self.store.graph_job_by_id(job_id)
            assert status is not None
            is_full_book = status.get("job_scope") == "full_book"
            boundary = None if is_full_book else ReadingBoundary(
                book_id=book_id, section_index=int(status["boundary_section"]),
                unit_index=int(status["boundary_unit"]), source_unit_id=str(status["boundary_source_unit_id"]),
            )
            remaining = [
                chunk for chunk in current[1]
                if (is_full_book or is_chunk_eligible(chunk, boundary))
                and chunk.id not in processed
                and chunk.id not in {item.id for item in batch}
            ]
            self.store.commit_graph_batch(
                prepared, job_id=job_id, chunk_ids=[chunk.id for chunk in batch],
                complete=not remaining, max_attempts=self._graph_batch_max_attempts,
            )
            self.story_graph.commit(prepared)

    def ingest_book(
        self,
        book_id: str,
        content_version_id: str,
        source_units: Sequence[SourceUnitInput],
    ) -> tuple[BookChunk, ...]:
        """Prepare all derived state before committing a replacement."""
        chunks = tuple(
            chunk_source_units(book_id, content_version_id, source_units)
        )
        prepared_keyword = self.keyword_index.prepare_book(
            book_id, content_version_id, chunks
        )
        # Provider calls and vector validation happen before any live state changes.
        try:
            provider = self.semantic_index.provider
            previous = self.store.get_book(book_id) if provider is not None else None
            prepared_semantic = None
            if (
                provider is not None
                and previous is not None
                and previous[0] == content_version_id
                and previous[1] == chunks
            ):
                identity, dimensions, stored_vectors = self.store.get_semantic_vectors(book_id)
                expected_ids = {chunk.id for chunk in chunks if chunk.text.strip()}
                if (
                    identity == _embedding_identity(provider)
                    and dimensions == provider.dimensions
                    and set(stored_vectors) == expected_ids
                ):
                    try:
                        prepared_semantic = self.semantic_index.prepare_book_with_embeddings(
                            book_id,
                            content_version_id,
                            chunks,
                            [stored_vectors.get(chunk.id) for chunk in chunks],
                        )
                    except (TypeError, ValueError):
                        # Treat corrupt or incompatible persisted vectors as stale;
                        # rebuild before replacing any canonical state.
                        prepared_semantic = None
            if prepared_semantic is None:
                prepared_semantic = self.semantic_index.prepare_book(
                    book_id, content_version_id, chunks
                )
        except Exception as exc:
            logger.exception("Semantic embedding preparation failed for book %s", book_id)
            raise EmbeddingIndexError(
                "semantic embedding preparation failed; previous book state was retained"
            ) from exc
        with self._lock:
            self.semantic_index.validate_prepared(prepared_semantic)
            current = self.store.get_book(book_id)
            old_graph = self.story_graph.get_book(book_id)
            next_graph = (
                old_graph
                if current is not None
                and current[0] == content_version_id
                and old_graph is not None
                else StoryGraph(book_id, content_version_id)
            )
            vectors = None
            embedding_identity = None
            embedding_dimensions = None
            if self.semantic_index.provider is not None:
                vectors = {
                    document.chunk.id: document.vector
                    for document in prepared_semantic.documents
                }
                embedding_identity = _embedding_identity(self.semantic_index.provider)
                embedding_dimensions = prepared_semantic.dimensions
            self.store.replace_book(
                book_id,
                content_version_id,
                chunks,
                graph=next_graph,
                vectors=vectors,
                embedding_identity=embedding_identity,
                embedding_dimensions=embedding_dimensions,
            )
            self.keyword_index.commit_book(prepared_keyword)
            self.semantic_index.commit_book(prepared_semantic)
            self.story_graph.commit(next_graph)
            self._semantic_storage_errors.pop(book_id, None)
        return chunks

    def ingest_book_keyword_stage(
        self, book_id: str, content_version_id: str,
        source_units: Sequence[SourceUnitInput],
        *, require_scheduled: bool = False,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[BookChunk, ...]:
        """Commit canonical chunks and BM25 independently of embeddings."""
        ordered_sections: dict[int, list[SourceUnitInput]] = {}
        for unit in source_units:
            ordered_sections.setdefault(unit.section_index, []).append(unit)
        built: list[BookChunk] = []
        completed = 0
        for section_index in sorted(ordered_sections):
            section_units = ordered_sections[section_index]
            built.extend(chunk_source_units(book_id, content_version_id, section_units))
            completed += len(section_units)
            if progress is not None:
                progress(completed, len(source_units))
        chunks = tuple(built)
        prepared_keyword = self.keyword_index.prepare_book(book_id, content_version_id, chunks)
        with self._lock:
            current = self.store.get_book(book_id)
            graph = self.story_graph.get_book(book_id)
            if graph is None or graph.content_version_id != content_version_id:
                graph = StoryGraph(book_id, content_version_id)
            self.store.replace_book(
                book_id, content_version_id, chunks, graph=graph, vectors=None,
                embedding_identity=None, embedding_dimensions=None,
                require_preparation_job=require_scheduled,
            )
            self.keyword_index.commit_book(prepared_keyword)
            if current is None or current[0] != content_version_id or current[1] != chunks:
                # A changed canonical version cannot retain an old semantic view.
                self.semantic_index.remove_book(book_id)
            if self.semantic_index.provider is not None and not self.semantic_is_compatible(book_id, content_version_id):
                self._semantic_storage_errors[book_id] = "embeddings are pending or incomplete for this book; keyword retrieval remains available"
        return chunks

    def validate_preparation_content_version(
        self, book_id: str, content_version_id: str,
        source_units: Sequence[SourceUnitInput],
    ) -> None:
        """Preserve the existing fast 409 contract for reused version IDs."""
        current = self.store.get_book(book_id)
        if current is None or current[0] != content_version_id:
            return
        proposed = tuple(chunk_source_units(book_id, content_version_id, source_units))
        if proposed != current[1]:
            raise ContentVersionConflictError(
                "content_version_id already exists with different content; use a new content version"
            )

    def embed_indexed_book(
        self, book_id: str, content_version_id: str,
        progress: Callable[[int, int], None] | None = None,
        *, require_scheduled: bool = False,
    ) -> int:
        """Build vectors in small batches and publish only a complete index."""
        provider = self.semantic_index.provider
        if provider is None:
            raise EmbeddingIndexError("no local embedding provider is configured")
        stored = self.store.get_book(book_id)
        if stored is None or stored[0] != content_version_id:
            raise ContentVersionMismatchError("book content version changed before embedding")
        chunks = stored[1]
        indexable = [chunk for chunk in chunks if chunk.text.strip()]
        vectors: list[Sequence[float] | None] = [None] * len(chunks)
        positions = {chunk.id: index for index, chunk in enumerate(chunks)}
        for start in range(0, len(indexable), 8):
            batch = indexable[start : start + 8]
            result = provider.embed_texts([chunk.text for chunk in batch])
            if len(result) != len(batch):
                raise EmbeddingIndexError("embedding provider returned an incomplete batch")
            for chunk, vector in zip(batch, result, strict=True):
                vectors[positions[chunk.id]] = vector
            if progress is not None:
                progress(min(start + len(batch), len(indexable)), len(indexable))
        prepared = self.semantic_index.prepare_book_with_embeddings(
            book_id, content_version_id, chunks, vectors
        )
        self.semantic_index.validate_prepared(prepared)
        with self._lock:
            current = self.store.get_book(book_id)
            if current is None or current[0] != content_version_id or current[1] != chunks:
                raise ContentVersionMismatchError("book changed while embeddings were generated")
            graph = self.story_graph.get_book(book_id)
            self.store.replace_book(
                book_id, content_version_id, chunks, graph=graph,
                vectors={item.chunk.id: item.vector for item in prepared.documents},
                embedding_identity=_embedding_identity(provider),
                embedding_dimensions=prepared.dimensions,
                require_preparation_job=require_scheduled,
            )
            self.semantic_index.commit_book(prepared)
            self._semantic_storage_errors.pop(book_id, None)
        return len(prepared.documents)

    def semantic_is_compatible(self, book_id: str, content_version_id: str) -> bool:
        provider = self.semantic_index.provider
        stored = self.store.get_book(book_id)
        if provider is None or stored is None or stored[0] != content_version_id:
            return False
        identity, dimensions, vectors = self.store.get_semantic_vectors(book_id)
        expected = {chunk.id for chunk in stored[1] if chunk.text.strip()}
        return identity == _embedding_identity(provider) and dimensions == provider.dimensions and set(vectors) == expected

    def delete_book(self, book_id: str) -> str | None:
        with self._lock:
            storage_key = self.store.delete_book(book_id)
            self.keyword_index.remove_book(book_id)
            self.semantic_index.remove_book(book_id)
            self.story_graph.remove_book(book_id)
            self._semantic_storage_errors.pop(book_id, None)
            return storage_key

    def extract_story_graph(
        self, *, book_id: str, reading_boundary: ReadingBoundary
    ) -> StoryGraph:
        """The legacy direct extractor is intentionally disabled."""
        raise StoryExtractionUnavailableError(
            "Direct generative graph extraction is retired. Schedule full-book local BookNLP preparation instead."
        )

    def eligible_story_facts(
        self, *, book_id: str, reading_boundary: ReadingBoundary
    ) -> tuple[
        tuple[StoryEntity, ...],
        tuple[StoryEvent, ...],
        tuple[StoryRelationship, ...],
    ]:
        """Return graph facts only after validating the boundary against the book."""
        with self._lock:
            stored = self.store.get_book(book_id)
            if stored is None:
                raise BookNotIndexedError(f"book {book_id!r} is not indexed")
            if self.store.graph_rag_enabled(book_id, stored[0]) is False:
                raise GraphRagDisabledError("Graph RAG is disabled for this book.")
            self._validate_book_boundary(book_id, reading_boundary)
            graph = self.story_graph.get_book(book_id)
            assert graph is not None
            return graph.eligible_facts(reading_boundary)

    def search_book(
        self,
        *,
        book_id: str,
        query: str,
        retrieval_mode: str,
        reading_boundary: ReadingBoundary,
        content_version_id: str | None = None,
        limit: int = 10,
        hybrid_config: HybridConfig = DEFAULT_HYBRID_CONFIG,
        graph_options: GraphRetrievalOptions = DEFAULT_GRAPH_OPTIONS,
    ) -> list[RetrievalResult]:
        with self._lock:
            self._ensure_semantic_compatible(
                retrieval_mode, hybrid_config.semantic_weight, (book_id,)
            )
            self._validate_book_boundary(book_id, reading_boundary)
            stored = self.store.get_book(book_id)
            assert stored is not None
            if retrieval_mode in ("graph", "hybrid_graph"):
                self._require_graph_facts({book_id: reading_boundary})
            current_version, _ = stored
            if content_version_id is not None and content_version_id != current_version:
                raise ContentVersionMismatchError(
                    f"book {book_id!r} is indexed as content version {current_version!r}"
                )
            boundaries = {book_id: reading_boundary}
            if retrieval_mode in ("hybrid", "graph", "hybrid_graph"):
                keyword_results = (
                    self.keyword_index.search(
                        query,
                        book_id=book_id,
                        content_version_id=content_version_id,
                        boundaries=boundaries,
                        limit=limit,
                    )
                    if hybrid_config.keyword_weight > 0
                    else []
                )
                semantic_results = (
                    self.semantic_index.search(
                        query,
                        book_id=book_id,
                        content_version_id=content_version_id,
                        boundaries=boundaries,
                        limit=limit,
                    )
                    if hybrid_config.semantic_weight > 0
                    else []
                )
                hybrid_results = [
                    result
                    for result in fuse_results(
                        keyword_results,
                        semantic_results,
                        config=hybrid_config,
                        limit=limit,
                    )
                    if is_chunk_eligible(result.chunk, reading_boundary)
                ]
                graph = self._booknlp_story_graph(
                    book_id, current_version, reading_boundary
                )
                if retrieval_mode in ("graph", "hybrid_graph"):
                    return expand_hybrid_results(
                    query=query,
                    seed_results=hybrid_results,
                    graphs={book_id: graph} if graph is not None else {},
                    chunks_by_book={book_id: stored[1]},
                    boundaries=boundaries,
                    limit=limit,
                    options=graph_options,
                    )
                return hybrid_results
            if retrieval_mode == "keyword":
                results = self.keyword_index.search(
                    query,
                    book_id=book_id,
                    content_version_id=content_version_id,
                    boundaries=boundaries,
                    limit=limit,
                )
            if retrieval_mode == "semantic":
                results = self.semantic_index.search(
                    query,
                    book_id=book_id,
                    content_version_id=content_version_id,
                    boundaries=boundaries,
                    limit=limit,
                )
            if retrieval_mode in ("keyword", "semantic"):
                return results
            raise ValueError(f"unsupported retrieval mode: {retrieval_mode}")

    def search_all_books(
        self,
        *,
        query: str,
        retrieval_mode: str,
        boundaries: Mapping[str, ReadingBoundary],
        limit: int = 10,
        hybrid_config: HybridConfig = DEFAULT_HYBRID_CONFIG,
        graph_options: GraphRetrievalOptions = DEFAULT_GRAPH_OPTIONS,
    ) -> list[RetrievalResult]:
        with self._lock:
            self._ensure_semantic_compatible(
                retrieval_mode, hybrid_config.semantic_weight, self.store.book_ids
            )
            indexed_book_ids = set(self.store.book_ids)
            if set(boundaries) != indexed_book_ids:
                raise InvalidReadingBoundaryError(
                    "all-books search requires one confirmed boundary for every indexed book"
                )
            for book_id, boundary in boundaries.items():
                self._validate_book_boundary(book_id, boundary)
            graphs = {
                book_id: graph
                for book_id in indexed_book_ids
                if self.store.graph_rag_enabled(
                    book_id, self.store.get_book(book_id)[0]
                ) is not False
                and (graph := self._booknlp_story_graph(
                    book_id,
                    self.store.get_book(book_id)[0],
                    boundaries[book_id],
                )) is not None
            }
            if retrieval_mode in ("graph", "hybrid_graph"):
                self._require_graph_facts(boundaries)
            if retrieval_mode in ("hybrid", "graph", "hybrid_graph"):
                keyword_results = (
                    self.keyword_index.search(
                        query,
                        all_books=True,
                        boundaries=boundaries,
                        limit=limit,
                    )
                    if hybrid_config.keyword_weight > 0
                    else []
                )
                semantic_results = (
                    self.semantic_index.search(
                        query,
                        all_books=True,
                        boundaries=boundaries,
                        limit=limit,
                    )
                    if hybrid_config.semantic_weight > 0
                    else []
                )
                hybrid_results = [
                    result
                    for result in fuse_results(
                        keyword_results,
                        semantic_results,
                        config=hybrid_config,
                        limit=limit,
                    )
                    if is_chunk_eligible(
                        result.chunk, boundaries[result.chunk.book_id]
                    )
                ]
                if retrieval_mode == "hybrid":
                    return hybrid_results
                if retrieval_mode in ("graph", "hybrid_graph") and not graphs:
                    raise StoryGraphUnavailableError(
                        "BookNLP story graph is not ready for every requested book."
                    )
                chunks_by_book = {
                    book_id: stored[1]
                    for book_id in indexed_book_ids
                    if (stored := self.store.get_book(book_id)) is not None
                }
                return expand_hybrid_results(
                    query=query,
                    seed_results=hybrid_results,
                    graphs=graphs,
                    chunks_by_book=chunks_by_book,
                    boundaries=boundaries,
                    limit=limit,
                    options=graph_options,
                )
            if retrieval_mode == "keyword":
                results = self.keyword_index.search(
                    query,
                    all_books=True,
                    boundaries=boundaries,
                    limit=limit,
                )
            if retrieval_mode == "semantic":
                results = self.semantic_index.search(
                    query,
                    all_books=True,
                    boundaries=boundaries,
                    limit=limit,
                )
            if retrieval_mode in ("keyword", "semantic"):
                return results
            raise ValueError(f"unsupported retrieval mode: {retrieval_mode}")

    def _validate_book_boundary(
        self, book_id: str, boundary: ReadingBoundary
    ) -> None:
        if boundary.book_id != book_id:
            raise InvalidReadingBoundaryError(
                "reading boundary book_id does not match query scope"
            )
        stored = self.store.get_book(book_id)
        if stored is None:
            raise BookNotIndexedError(f"book {book_id!r} is not indexed")
        _, chunks = stored
        if not any(
            chunk.section_index == boundary.section_index
            and any(
                position.unit_index == boundary.unit_index
                and position.source_unit_id == boundary.source_unit_id
                for position in chunk.source_positions
            )
            for chunk in chunks
        ):
            raise InvalidReadingBoundaryError(
                "reading boundary must identify a source unit in the indexed book"
            )

    def _booknlp_story_graph(
        self, book_id: str, version: str, boundary: ReadingBoundary
    ) -> StoryGraph | None:
        """Adapt only boundary-eligible BookNLP evidence to graph retrieval."""
        data = self.store.load_booknlp_graph(book_id, version)
        if data is None:
            return None
        stored = self.store.get_book(book_id)
        if stored is None or stored[0] != version:
            return None
        chunks = {chunk.id: chunk for chunk in stored[1]}
        profiles = data.get("profiles", [])
        mentions = data.get("mentions", [])
        edges = data.get("edges", [])
        mention_by_character: dict[str, list[dict[str, object]]] = {}
        for mention in mentions:
            chunk = chunks.get(str(mention.get("source_chunk_id", "")))
            if chunk is not None and is_chunk_eligible(chunk, boundary):
                mention_by_character.setdefault(
                    str(mention["character_id"]), []
                ).append(mention)

        def evidence_for(source_chunk_id: object, assertedness: str = "inferred") -> GraphEvidence | None:
            if not isinstance(source_chunk_id, str):
                return None
            chunk = chunks.get(source_chunk_id)
            if chunk is None:
                return None
            return GraphEvidence(
                book_id=book_id, content_version_id=version, source_chunk_id=chunk.id,
                section_index=chunk.section_index, chunk_index=chunk.chunk_index,
                source_unit_ids=tuple(chunk.source_unit_ids),
                source_positions=tuple(GraphSourcePosition(
                    source_unit_id=position.source_unit_id,
                    unit_index=position.unit_index, kind=position.kind,
                    href=position.href, cfi=position.cfi,
                ) for position in chunk.source_positions),
                assertedness=assertedness, extractor_name="booknlp-candidate-co-mention",
                extractor_version=str(data.get("extraction_version", BOOKNLP_GRAPH_VERSION)),
            )

        entities: list[StoryEntity] = []
        for profile in profiles:
            character_id = str(profile["character_id"])
            profile_mentions = mention_by_character.get(character_id, [])
            mention_evidence = tuple(
                evidence for item in profile_mentions
                if (evidence := evidence_for(item.get("source_chunk_id"))) is not None
            )
            if not mention_evidence:
                continue
            aliases: dict[str, list[GraphEvidence]] = {}
            for mention in profile_mentions:
                kind = str(mention.get("mention_kind", "")).upper()
                name = str(mention.get("mention_text", "")).strip()
                evidence = evidence_for(mention.get("source_chunk_id"))
                if name and evidence and kind != "PRON":
                    aliases.setdefault(name.casefold(), []).append(evidence)
            entities.append(StoryEntity(
                id=character_id, book_id=book_id, content_version_id=version,
                # BookNLP's preferred cluster name may first appear in a
                # future chapter. Use the first eligible textual mention so
                # graph reasons cannot reveal that later identity.
                name=_boundary_safe_character_name(profile_mentions),
                kind="character",
                identity_status="resolved",
                aliases=tuple(StoryAlias(name=next(
                    str(m["mention_text"]) for m in profile_mentions
                    if str(m.get("mention_text", "")).casefold() == key
                ), evidence=tuple(sorted(set(ev), key=lambda e: (e.section_index, e.chunk_index, e.source_chunk_id))))
                    for key, ev in sorted(aliases.items())),
                description=None, evidence=tuple(sorted(set(mention_evidence), key=lambda e: (e.section_index, e.chunk_index, e.source_chunk_id))),
            ))
        entity_ids = {entity.id for entity in entities}
        relationships: list[StoryRelationship] = []
        for edge in edges:
            source, target = str(edge["source_character_id"]), str(edge["target_character_id"])
            if source not in entity_ids or target not in entity_ids:
                continue
            evidence = tuple(
                item for row in edge.get("evidence", [])
                if (item := evidence_for(row.get("source_chunk_id"))) is not None
                and item.is_eligible(boundary)
            )
            if not evidence:
                continue
            relationships.append(StoryRelationship(
                id=str(edge["edge_id"]), book_id=book_id,
                content_version_id=version, subject_entity_id=source,
                object_entity_id=target,
                relationship_type=str(edge.get("candidate_category", "CO_MENTION")),
                description="Candidate co-mention relationship; verbs are heuristic evidence only.",
                temporal_metadata=(("candidate_verbs", ", ".join(edge.get("candidate_verbs", []))),),
                evidence=evidence,
            ))
        return StoryGraph(
            book_id=book_id, content_version_id=version,
            entities=tuple(entities), relationships=tuple(relationships),
        )

    def eligible_character_graph(
        self, *, book_id: str, reading_boundary: ReadingBoundary | None,
        view_mode: str = "till_now",
    ) -> dict[str, object]:
        """Build full-book UI data or a fail-closed source-unit-filtered view."""
        with self._lock:
            stored = self.store.get_book(book_id)
            if stored is None:
                raise BookNotIndexedError(f"book {book_id!r} is not indexed")
            if self.store.graph_rag_enabled(book_id, stored[0]) is False:
                raise GraphRagDisabledError("Graph RAG is disabled for this book.")
            if view_mode not in ("full_book", "till_now"):
                raise ValueError("unsupported graph view mode")
            if view_mode == "till_now":
                if reading_boundary is None:
                    raise InvalidReadingBoundaryError("Till now requires an explicitly confirmed boundary")
                self._validate_book_boundary(book_id, reading_boundary)
            elif reading_boundary is not None:
                raise ValueError("full book view does not accept a reading boundary")
            data = self.store.load_booknlp_graph(book_id, stored[0])
            if data is None:
                raise StoryGraphUnavailableError("BookNLP graph preparation is not complete yet.")
            chunks = {chunk.id: chunk for chunk in stored[1]}
            source_positions = {
                (chunk.section_index, position.unit_index, position.source_unit_id)
                for chunk in stored[1]
                for position in chunk.source_positions
            }
            def eligible(record: Mapping[str, object]) -> bool:
                if view_mode == "full_book":
                    return True
                assert reading_boundary is not None
                section = record.get("section_index")
                unit = record.get("unit_index")
                source_id = record.get("source_unit_id")
                if isinstance(section, int) and isinstance(unit, int) and isinstance(source_id, str):
                    if (section, unit, source_id) not in source_positions:
                        return False
                    return (section, unit, source_id) <= (
                        reading_boundary.section_index,
                        reading_boundary.unit_index,
                        reading_boundary.source_unit_id,
                    )
                # Read legacy records using their persisted chunk/source positions.
                chunk = chunks.get(str(record.get("source_chunk_id", "")))
                return chunk is not None and is_chunk_eligible(chunk, reading_boundary)

            eligible_mentions: dict[str, list[dict[str, object]]] = {}
            for mention in data["mentions"]:
                if eligible(mention):
                    eligible_mentions.setdefault(str(mention["character_id"]), []).append(mention)
            profiles_by_id = {str(profile["character_id"]): profile for profile in data["profiles"]}
            edges: list[dict[str, object]] = []
            visible_ids: set[str] = set()
            for edge in data["edges"]:
                evidence = []
                for row in edge.get("evidence", []):
                    chunk = chunks.get(str(row.get("source_chunk_id", "")))
                    if view_mode == "full_book":
                        if chunk is None:
                            continue
                    elif not eligible(row) or chunk is None:
                        continue
                    unit_id = row.get("source_unit_id")
                    unit_position = next(
                        (position for position in chunk.source_positions if position.source_unit_id == unit_id),
                        None,
                    )
                    evidence.append({
                        **row,
                        "source_unit_ids": [str(unit_id)] if unit_position else list(chunk.source_unit_ids),
                        "href": row.get("href") or (unit_position.href if unit_position else chunk.href),
                        "cfi_status": (
                            "complete" if unit_position and unit_position.cfi
                            else "unavailable" if unit_position
                            else chunk.cfi_status
                        ),
                        "source_positions": ([{
                            "source_unit_id": unit_position.source_unit_id,
                            "unit_index": unit_position.unit_index,
                            "href": unit_position.href,
                            "cfi": unit_position.cfi,
                        }] if unit_position else [
                            {"source_unit_id": position.source_unit_id,
                             "unit_index": position.unit_index,
                             "href": position.href, "cfi": position.cfi}
                            for position in chunk.source_positions
                        ]),
                    })
                source, target = str(edge["source_character_id"]), str(edge["target_character_id"])
                if (
                    not evidence
                    or source not in profiles_by_id
                    or target not in profiles_by_id
                    or (view_mode == "till_now" and (
                        source not in eligible_mentions or target not in eligible_mentions
                    ))
                ):
                    continue
                verbs = sorted(Counter(
                    str(verb) for row in evidence for verb in row.get("verbs", [])
                ).items(), key=lambda item: (-item[1], item[0]))[:5]
                verb_names = [name for name, _ in verbs]
                edges.append({
                    "id": edge["edge_id"], "source": source, "target": target,
                    "candidate_category": categorize_candidate_verbs(verb_names),
                    "candidate_verbs": verb_names, "weight": len(evidence),
                    "evidence_count": len(evidence),
                    # A full-book truncation bit can reveal that later evidence
                    # exists. In Till now, expose only truncation among the
                    # eligible evidence actually returned by this response.
                    "evidence_truncated": (
                        bool(edge.get("evidence_truncated"))
                        if view_mode == "full_book"
                        else len(evidence) > 5
                    ),
                    "evidence": evidence[:5],
                })
                visible_ids.update((source, target))
            nodes = []
            node_ids = (
                {str(profile["character_id"]) for profile in data["profiles"]}
                if view_mode == "full_book"
                else visible_ids | set(eligible_mentions)
            )
            for character_id in sorted(node_ids):
                mentions = eligible_mentions.get(character_id, [])
                display_name = (
                    str(profiles_by_id[character_id].get("canonical_name", "Unknown"))
                    if view_mode == "full_book"
                    else _boundary_safe_character_name(mentions)
                )
                alias_names = sorted({
                    str(item["mention_text"]).strip() for item in mentions
                    if str(item.get("mention_kind", "")).upper() != "PRON"
                    and str(item.get("mention_text", "")).strip().casefold() != display_name.casefold()
                }, key=str.casefold)
                pronouns = sorted({
                    str(item["mention_text"]).strip() for item in mentions
                    if str(item.get("mention_kind", "")).upper() == "PRON"
                    and str(item.get("mention_text", "")).strip()
                }, key=str.casefold)
                node_payload: dict[str, object] = {
                    "id": character_id, "name": display_name,
                    "mention_count": int(profiles_by_id.get(character_id, {}).get("mention_count", len(mentions))) if view_mode == "full_book" else len(mentions), "aliases": (sorted({str(item.get("term", "")) for item in profiles_by_id.get(character_id, {}).get("aliases", []) if isinstance(item, Mapping) and item.get("term")}, key=str.casefold)[:20] if view_mode == "full_book" else alias_names[:20]),
                    "pronouns": pronouns[:12],
                }
                if view_mode == "full_book":
                    summary = profiles_by_id.get(character_id, {}).get("summary")
                    if isinstance(summary, Mapping):
                        node_payload["profile_summary"] = dict(summary)
                nodes.append(node_payload)
            return {
                "book_id": book_id, "content_version_id": stored[0],
                "view_mode": view_mode,
                "boundary": ({
                    "section_index": reading_boundary.section_index,
                    "unit_index": reading_boundary.unit_index,
                    "source_unit_id": reading_boundary.source_unit_id,
                } if reading_boundary is not None else None),
                "nodes": nodes, "edges": sorted(edges, key=lambda e: str(e["id"])),
            }

    def close(self) -> None:
        self._graph_worker_stop.set()
        self._graph_worker_event.set()
        if self._graph_worker_thread is not None:
            self._graph_worker_thread.join(timeout=5)
            if self._graph_worker_thread.is_alive():
                logger.warning("Graph extraction worker is still stopping; deferring store close")
                return
        if self._booknlp_worker_thread is not None:
            self._booknlp_worker_thread.join(timeout=5)
            if self._booknlp_worker_thread.is_alive():
                logger.warning("BookNLP graph worker is still stopping; deferring store close")
                return
        self.store.close()

    def _require_graph_facts(
        self, boundaries: Mapping[str, ReadingBoundary]
    ) -> None:
        if not boundaries:
            raise StoryGraphUnavailableError("no indexed books were provided for graph retrieval")
        for book_id, boundary in boundaries.items():
            stored = self.store.get_book(book_id)
            if stored is None:
                raise BookNotIndexedError(f"book {book_id!r} is not indexed")
            if self.store.graph_rag_enabled(book_id, stored[0]) is False:
                raise GraphRagDisabledError(
                    f"GraphRAG is not enabled for book {book_id!r}."
                )
            if self._booknlp_story_graph(book_id, stored[0], boundary) is None:
                raise StoryGraphUnavailableError(
                    f"BookNLP story graph for book {book_id!r} is not complete yet"
                )

    def _ensure_semantic_compatible(
        self, retrieval_mode: str, semantic_weight: float, book_ids: Sequence[str]
    ) -> None:
        if retrieval_mode == "semantic" or (
            retrieval_mode in ("hybrid", "graph", "hybrid_graph") and semantic_weight > 0
        ):
            failures = {
                book_id: self._semantic_storage_errors[book_id]
                for book_id in book_ids
                if book_id in self._semantic_storage_errors
            }
            if failures:
                details = "; ".join(
                    f"{book_id}: {message}" for book_id, message in sorted(failures.items())
                )
                raise EmbeddingIndexError(details)


def _embedding_identity(provider: EmbeddingProvider) -> str:
    explicit = (
        getattr(provider, "identity", None)
        or getattr(provider, "provider_id", None)
        or getattr(provider, "model_id", None)
    )
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    provider_type = type(provider)
    return f"{provider_type.__module__}.{provider_type.__qualname__}:{provider.dimensions}"


def _positive_env_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid %s; using default %d", name, default)
        return default
    if value < 1 or value > 100:
        logger.warning("Out-of-range %s; using default %d", name, default)
        return default
    return value


def _safe_graph_error(error: Exception) -> str:
    # Provider exception strings are useful diagnostics but should not include
    # request payloads or configuration secrets. Current provider errors only
    # report status/category; cap their length before exposing in job status.
    return f"{type(error).__name__}: {str(error)[:900]}"


def _boundary_safe_character_name(mentions: Sequence[Mapping[str, object]]) -> str:
    """Choose a visible label only from mentions at the confirmed boundary."""
    ordered = sorted(
        mentions,
        key=lambda mention: (
            int(mention.get("section_index") or 0),
            int(mention.get("chunk_index") or 0),
            int(mention.get("token_start") or 0),
            str(mention.get("mention_id") or ""),
        ),
    )
    proper = [
        item for item in ordered
        if str(item.get("mention_kind", "")).upper() == "PROP"
        and str(item.get("mention_text", "")).strip()
    ]
    usable = proper or [
        item for item in ordered
        if str(item.get("mention_kind", "")).upper() != "PRON"
        and str(item.get("mention_text", "")).strip()
    ] or ordered
    return str(usable[0].get("mention_text", "")).strip() or "Character"
