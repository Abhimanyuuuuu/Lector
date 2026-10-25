"""Durable incremental graph extraction job tests."""

from __future__ import annotations

import tempfile
import sqlite3
import time
import unittest
from pathlib import Path
from threading import Event
from typing import Sequence

from fastapi.testclient import TestClient

from lector_rag_api.chunking import BookChunk
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService
from lector_rag_api.story_graph import StoryExtraction
from lector_rag_api.main import create_app


class RecordingExtractor:
    name = "recording"
    version = "1"

    def __init__(self, *, gate: Event | None = None, fail_unit: str | None = None) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.gate = gate
        self.fail_unit = fail_unit
        self.failed = False
        self.entered = Event()

    def extract(self, chunks: Sequence[BookChunk]) -> StoryExtraction:
        ids = tuple(chunk.id for chunk in chunks)
        self.calls.append(ids)
        self.entered.set()
        if self.gate is not None:
            self.gate.wait(timeout=3)
        if self.fail_unit and any(self.fail_unit in chunk.source_unit_ids for chunk in chunks):
            raise TimeoutError("injected model timeout")
        return StoryExtraction()


class FactExtractor(RecordingExtractor):
    def extract(self, chunks: Sequence[BookChunk]) -> StoryExtraction:
        chunk = chunks[0]
        return StoryExtraction.model_validate({
            "entities": [{
                "ref": "entity-local", "name": "Ada", "kind": "character",
                "identity_status": "resolved", "assertedness": "explicit",
                "evidence": [{"source_chunk_id": chunk.id, "source_unit_ids": list(chunk.source_unit_ids)}],
            }],
            "events": [], "relationships": [],
        })


def source_units(count: int) -> list[SourceUnitInput]:
    # Each section is isolated by the production chunker, making boundary and
    # batch membership directly auditable in these tests.
    return [
        SourceUnitInput(
            source_unit_id=f"u{index}", section_index=index, unit_index=0,
            kind="paragraph", text=f"Readable passage number {index}.",
        )
        for index in range(count)
    ]


def wait_for(service: InMemoryRagService, expected: str, timeout: float = 5) -> dict[str, object]:
    deadline = time.monotonic() + timeout
    result = None
    while time.monotonic() < deadline:
        result = service.graph_job_status(book_id="book", content_version_id="v1")
        if result and result["status"] == expected:
            return result
        time.sleep(0.02)
    raise AssertionError(f"graph job did not reach {expected!r}; last status={result!r}")


class GraphJobTests(unittest.TestCase):
    def test_local_only_extractor_extends_graph_at_a_later_boundary(self) -> None:
        class NamedFactExtractor(RecordingExtractor):
            name = "ollama-story-extractor"
            version = "ollama:fixture"

            def extract(self, chunks: Sequence[BookChunk]) -> StoryExtraction:
                chunk = chunks[0]
                self.calls.append((chunk.id,))
                return StoryExtraction.model_validate({
                    "entities": [{
                        "ref": "local-ref", "name": "Mara", "kind": "character",
                        "identity_status": "resolved", "assertedness": "explicit",
                        "evidence": [{"source_chunk_id": chunk.id, "source_unit_ids": list(chunk.source_unit_ids)}],
                    }],
                    "events": [], "relationships": [],
                })

        extractor = NamedFactExtractor()
        service = InMemoryRagService(story_extractor=extractor, story_extractors={"ollama": extractor})
        try:
            service.ingest_book("book", "v1", source_units(2))
            service.schedule_story_graph(book_id="book", reading_boundary=ReadingBoundary("book", 0, 0, "u0"))
            self.assertEqual(wait_for(service, "completed")["provider"], "ollama")
            service.schedule_story_graph(book_id="book", reading_boundary=ReadingBoundary("book", 1, 0, "u1"))
            self.assertEqual(wait_for(service, "completed")["provider_version"], extractor.version)
            graph = service.story_graph.get_book("book")
            self.assertEqual(len(graph.entities), 1)
            self.assertEqual(len(graph.entities[0].evidence), 2)
            self.assertEqual(len(extractor.calls), 2)
        finally:
            service.close()

    def test_graph_api_rejects_provider_selection(self) -> None:
        extractor = RecordingExtractor()
        service = InMemoryRagService(story_extractor=extractor, story_extractors={"ollama": extractor})
        client = TestClient(create_app(service))
        try:
            client.post("/api/v1/books/book/source-units", json={
                "content_version_id": "v1",
                "source_units": [{
                    "source_unit_id": "u0", "section_index": 0, "unit_index": 0,
                    "kind": "paragraph", "text": "A safe passage.",
                }],
            })
            rejected = client.post("/api/v1/books/book/graph/extract", json={
                "confirmed": True, "section_index": 0, "unit_index": 0,
                "source_unit_id": "u0", "provider": "groq",
            })
            self.assertEqual(rejected.status_code, 422)
            self.assertEqual(rejected.json()["error"]["code"], "validation_error")
        finally:
            client.close()
            service.close()
    def test_schema_migration_preserves_graph_and_seeds_existing_evidence_progress(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = Path(directory) / "legacy.sqlite3"
            service = InMemoryRagService(story_extractor=FactExtractor(), database_path=db)
            chunks = service.ingest_book("book", "v1", source_units(1))
            graph = service.extract_story_graph(
                book_id="book", reading_boundary=ReadingBoundary("book", 0, 0, "u0")
            )
            self.assertEqual(len(graph.entities), 1)
            service.close()

            connection = sqlite3.connect(db)
            try:
                connection.execute("DROP TABLE graph_jobs")
                connection.execute("DROP TABLE graph_processed_chunks")
                connection.execute("PRAGMA user_version=1")
                connection.commit()
            finally:
                connection.close()

            migrated = InMemoryRagService(database_path=db)
            restored = migrated.story_graph.get_book("book")
            self.assertEqual(restored, graph)
            self.assertEqual(
                migrated.store.processed_graph_chunk_ids("book", "v1"),
                {chunks[0].id},
            )
            migrated.close()

    def test_restart_recovers_an_interrupted_processing_claim(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = Path(directory) / "interrupted.sqlite3"
            initial = InMemoryRagService(database_path=db)
            chunks = initial.ingest_book("book", "v1", source_units(1))
            initial.store.schedule_graph_job(
                job_id="interrupted-job", book_id="book", content_version_id="v1",
                boundary_section=0, boundary_unit=0, boundary_source_unit_id="u0",
                total_chunks=1,
            )
            initial.store.claim_graph_batch("interrupted-job", [chunks[0].id])
            initial.close()

            extractor = RecordingExtractor()
            recovered = InMemoryRagService(story_extractor=extractor, database_path=db)
            self.assertEqual(
                recovered.graph_job_status(book_id="book", content_version_id="v1")["status"],
                "queued",
            )
            recovered.resume_graph_jobs()
            self.assertEqual(wait_for(recovered, "completed")["processed_chunks"], 1)
            self.assertEqual(len(extractor.calls), 1)
            recovered.close()

    def test_graph_api_returns_durable_job_promptly_and_exposes_status(self) -> None:
        gate = Event()
        extractor = RecordingExtractor(gate=gate)
        service = InMemoryRagService(story_extractor=extractor)
        client = TestClient(create_app(service))
        try:
            indexed = client.post("/api/v1/books/book/source-units", json={
                "content_version_id": "v1",
                "source_units": [{
                    "source_unit_id": "u0", "section_index": 0, "unit_index": 0,
                    "kind": "paragraph", "text": "A safe passage.",
                }],
            })
            self.assertEqual(indexed.status_code, 202)
            started = time.monotonic()
            scheduled = client.post("/api/v1/books/book/graph/extract", json={
                "confirmed": True, "section_index": 0, "unit_index": 0,
                "source_unit_id": "u0",
            })
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertEqual(scheduled.status_code, 202)
            payload = scheduled.json()
            self.assertEqual(payload["total_chunks"], 1)
            self.assertIn(payload["status"], ("queued", "processing"))
            status = client.get(
                "/api/v1/books/book/graph/status?content_version_id=v1"
            )
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["job_id"], payload["job_id"])
            gate.set()
        finally:
            gate.set()
            client.close()
            service.close()

    def test_graph_api_has_no_provider_selection(self) -> None:
        extractor = RecordingExtractor()
        service = InMemoryRagService(
            story_extractor=extractor,
            story_extractors={"ollama": extractor},
        )
        client = TestClient(create_app(service))
        try:
            client.post("/api/v1/books/book/source-units", json={
                "content_version_id": "v1",
                "source_units": [{
                    "source_unit_id": "u0", "section_index": 0, "unit_index": 0,
                    "kind": "paragraph", "text": "A safe passage.",
                }],
            })
            rejected = client.post("/api/v1/books/book/graph/extract", json={
                "confirmed": True, "section_index": 0, "unit_index": 0,
                "source_unit_id": "u0", "provider": "groq",
            })
            self.assertEqual(rejected.status_code, 422, rejected.text)
            self.assertEqual(rejected.json()["error"]["code"], "validation_error")
            self.assertEqual(extractor.calls, [])
        finally:
            client.close()
            service.close()

    def test_unconfigured_local_extractor_fails_clearly(self) -> None:
        service = InMemoryRagService()
        client = TestClient(create_app(service))
        try:
            client.post("/api/v1/books/book/source-units", json={
                "content_version_id": "v1",
                "source_units": [{
                    "source_unit_id": "u0", "section_index": 0, "unit_index": 0,
                    "kind": "paragraph", "text": "A safe passage.",
                }],
            })
            body = {"confirmed": True, "section_index": 0, "unit_index": 0,
                    "source_unit_id": "u0"}
            missing = client.post("/api/v1/books/book/graph/extract", json=body)
            self.assertEqual(missing.status_code, 503)
            self.assertEqual(missing.json()["error"]["code"], "story_extraction_unavailable")
        finally:
            client.close()
            service.close()

    def test_batches_are_bounded_boundary_filtered_and_persisted_before_completion(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = Path(directory) / "lector.sqlite3"
            extractor = RecordingExtractor()
            service = InMemoryRagService(story_extractor=extractor, database_path=db)
            service._graph_batch_size = 2
            chunks = service.ingest_book("book", "v1", source_units(7))
            job = service.schedule_story_graph(
                book_id="book", reading_boundary=ReadingBoundary("book", 3, 0, "u3")
            )
            self.assertEqual(job["status"], "queued")
            completed = wait_for(service, "completed")
            self.assertEqual(completed["total_chunks"], 4)
            self.assertEqual(completed["processed_chunks"], 4)
            self.assertEqual([len(batch) for batch in extractor.calls], [2, 2])
            eligible_ids = {chunk.id for chunk in chunks if chunk.section_index <= 3}
            called_ids = {chunk_id for batch in extractor.calls for chunk_id in batch}
            self.assertEqual(called_ids, eligible_ids)
            self.assertTrue(all(len(batch) <= 2 for batch in extractor.calls))
            self.assertEqual(service.store.processed_graph_chunk_ids("book", "v1"), eligible_ids)
            service.close()

            resumed_extractor = RecordingExtractor()
            reopened = InMemoryRagService(story_extractor=resumed_extractor, database_path=db)
            reopened.resume_graph_jobs()
            time.sleep(0.1)
            self.assertEqual(resumed_extractor.calls, [])
            self.assertEqual(reopened.graph_job_status(book_id="book", content_version_id="v1")["status"], "completed")
            reopened.close()

    def test_schedule_returns_while_extractor_is_busy_and_same_job_deduplicates(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            gate = Event()
            extractor = RecordingExtractor(gate=gate)
            service = InMemoryRagService(
                story_extractor=extractor, database_path=Path(directory) / "db.sqlite3"
            )
            service.ingest_book("book", "v1", source_units(3))
            boundary = ReadingBoundary("book", 2, 0, "u2")
            started = time.monotonic()
            first = service.schedule_story_graph(book_id="book", reading_boundary=boundary)
            self.assertLess(time.monotonic() - started, 0.5)
            self.assertTrue(extractor.entered.wait(timeout=1))
            second = service.schedule_story_graph(book_id="book", reading_boundary=boundary)
            self.assertEqual(first["job_id"], second["job_id"])
            gate.set()
            self.assertEqual(wait_for(service, "completed")["processed_chunks"], 3)
            service.close()

    def test_advancing_confirmation_schedules_only_newly_eligible_chunks(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            extractor = RecordingExtractor()
            service = InMemoryRagService(
                story_extractor=extractor, database_path=Path(directory) / "db.sqlite3"
            )
            service._graph_batch_size = 2
            chunks = service.ingest_book("book", "v1", source_units(5))
            service.schedule_story_graph(
                book_id="book", reading_boundary=ReadingBoundary("book", 1, 0, "u1")
            )
            self.assertEqual(wait_for(service, "completed")["processed_chunks"], 2)
            first_run_ids = {chunk_id for call in extractor.calls for chunk_id in call}

            later = service.schedule_story_graph(
                book_id="book", reading_boundary=ReadingBoundary("book", 3, 0, "u3")
            )
            self.assertEqual(later["total_chunks"], 4)
            self.assertEqual(later["processed_chunks"], 2)
            self.assertEqual(wait_for(service, "completed")["processed_chunks"], 4)
            all_called = {chunk_id for call in extractor.calls for chunk_id in call}
            self.assertEqual(all_called, {chunk.id for chunk in chunks[:4]})
            self.assertEqual(len(all_called - first_run_ids), 2)
            service.close()

    def test_failed_batch_preserves_prior_batches_and_retry_resumes_remaining_work(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            db = Path(directory) / "db.sqlite3"
            extractor = RecordingExtractor(fail_unit="u1")
            service = InMemoryRagService(story_extractor=extractor, database_path=db)
            service._graph_batch_size = 1
            service.ingest_book("book", "v1", source_units(3))
            boundary = ReadingBoundary("book", 2, 0, "u2")
            service.schedule_story_graph(book_id="book", reading_boundary=boundary)
            failed = wait_for(service, "failed")
            self.assertEqual(failed["processed_chunks"], 1)
            self.assertEqual(len(service.store.processed_graph_chunk_ids("book", "v1")), 1)

            extractor.fail_unit = None
            retried = service.schedule_story_graph(
                book_id="book", reading_boundary=boundary, retry=True
            )
            self.assertEqual(retried["status"], "queued")
            completed = wait_for(service, "completed")
            self.assertEqual(completed["processed_chunks"], 3)
            self.assertEqual(len(service.store.processed_graph_chunk_ids("book", "v1")), 3)
            service.close()

    def test_new_content_version_cascades_old_jobs_and_processed_markers(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            service = InMemoryRagService(database_path=Path(directory) / "db.sqlite3")
            service.ingest_book("book", "v1", source_units(1))
            # Seed job directly: this also verifies migration-backed persistence
            # without invoking a model provider.
            job = service.store.schedule_graph_job(
                job_id="job", book_id="book", content_version_id="v1",
                boundary_section=0, boundary_unit=0, boundary_source_unit_id="u0",
                total_chunks=1,
            )
            self.assertEqual(job["status"], "queued")
            service.ingest_book("book", "v2", source_units(1))
            self.assertIsNone(service.graph_job_status(book_id="book", content_version_id="v1"))
            self.assertEqual(service.store.processed_graph_chunk_ids("book", "v1"), set())
            service.close()


if __name__ == "__main__":
    unittest.main()
