"""Per-book Graph RAG opt-in and automatic full-book preparation tests."""

from __future__ import annotations

import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from typing import Sequence

from fastapi.testclient import TestClient

from lector_rag_api.chunking import BookChunk
from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService
from lector_rag_api.story_graph import StoryExtraction


def units(count: int = 3) -> list[dict[str, object]]:
    return [
        {
            "source_unit_id": f"unit-{index}",
            "section_index": index,
            "unit_index": 0,
            "kind": "paragraph",
            "text": f"Story passage {index}.",
        }
        for index in range(count)
    ]


class RecordingLocalExtractor:
    name = "ollama-story-extractor"
    version = "ollama:qwen3:0.6b"

    def __init__(self) -> None:
        self.batches: list[tuple[str, ...]] = []

    def extract(self, chunks: Sequence[BookChunk]) -> StoryExtraction:
        self.batches.append(tuple(chunk.id for chunk in chunks))
        return StoryExtraction()


class GraphOptInTests(unittest.TestCase):
    def test_no_choice_does_not_create_a_graph_job_or_call_the_extractor(self) -> None:
        extractor = RecordingLocalExtractor()
        service = InMemoryRagService(
            story_extractor=extractor,
            story_extractors={"ollama": extractor},
        )
        try:
            service.ingest_book_keyword_stage(
                "book", "v1", [SourceUnitInput.model_validate(item) for item in units()]
            )
            service.store.schedule_book_preparation(
                book_id="book", content_version_id="v1", source_units=units(),
                graph_rag_enabled=False,
            )
            self.assertIsNone(service.start_graph_preparation_if_enabled("book", "v1"))
            self.assertIsNone(service.graph_job_status(book_id="book", content_version_id="v1"))
            self.assertEqual(extractor.batches, [])
            self.assertIsNone(
                service.store._connection.execute(
                    "SELECT 1 FROM graph_snapshots WHERE book_id=?", ("book",)
                ).fetchone()
            )
            status = service.store.book_preparation_status("book", "v1")
            self.assertFalse(status["graph_rag_enabled"])
            self.assertEqual(status["graph_status"], "not_enabled")
        finally:
            service.close()

    def test_api_automatically_builds_opted_in_full_book_in_bounded_batches(self) -> None:
        extractor = RecordingLocalExtractor()
        with tempfile.TemporaryDirectory() as directory:
            service = InMemoryRagService(
                story_extractor=extractor,
                story_extractors={"ollama": extractor},
                database_path=Path(directory) / "lector.sqlite3",
            )
            service._graph_batch_size = 1
            client = TestClient(create_app(service))
            try:
                with client:
                    response = client.post(
                        "/api/v1/books/book/prepare",
                        json={"content_version_id": "v1", "source_units": units(), "graph_rag_enabled": True},
                    )
                    self.assertEqual(response.status_code, 202, response.text)
                    deadline = time.monotonic() + 5
                    status = None
                    while time.monotonic() < deadline:
                        status = client.get(
                            "/api/v1/books/book/preparation",
                            params={"content_version_id": "v1"},
                        ).json()
                        if status["graph_status"] == "complete":
                            break
                        time.sleep(0.02)
                    self.assertIsNotNone(status)
                    self.assertTrue(status["graph_rag_enabled"])
                    self.assertEqual(status["graph_status"], "complete")
                    self.assertEqual(status["graph_completed"], 3)
                    self.assertEqual(status["graph_total"], 3)
                    self.assertEqual([len(batch) for batch in extractor.batches], [1, 1, 1])
                    all_book_chunks = service.store.get_book("book")[1]
                    self.assertEqual(
                        {chunk_id for batch in extractor.batches for chunk_id in batch},
                        {chunk.id for chunk in all_book_chunks},
                    )
                    graph_status = client.get(
                        "/api/v1/books/book/graph/status",
                        params={"content_version_id": "v1"},
                    )
                    self.assertEqual(graph_status.status_code, 200)
                    self.assertEqual(graph_status.json()["job_scope"], "full_book")
                    self.assertIsNone(graph_status.json()["boundary_section"])
                    self.assertIsNone(graph_status.json()["boundary_unit"])
            finally:
                client.close()
                service.close()

    def test_preference_survives_reopen_and_legacy_graph_jobs_opt_in_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lector.sqlite3"
            service = InMemoryRagService(database_path=path)
            service.ingest_book_keyword_stage(
                "existing-graph", "v1", [SourceUnitInput.model_validate(item) for item in units(1)]
            )
            service.ingest_book_keyword_stage(
                "no-graph", "v1", [SourceUnitInput.model_validate(item) for item in units(1)]
            )
            service.store.schedule_book_preparation(
                book_id="existing-graph", content_version_id="v1", source_units=units(1),
                graph_rag_enabled=True,
            )
            service.store.schedule_book_preparation(
                book_id="no-graph", content_version_id="v1", source_units=units(1),
                graph_rag_enabled=False,
            )
            service.store.schedule_graph_job(
                job_id="legacy", book_id="existing-graph", content_version_id="v1",
                boundary_section=0, boundary_unit=0, boundary_source_unit_id="unit-0",
                total_chunks=1,
            )
            service.close()

            connection = sqlite3.connect(path)
            try:
                for column in ("graph_error", "graph_total", "graph_completed", "graph_status", "graph_rag_enabled"):
                    connection.execute(f"ALTER TABLE book_preparation_jobs DROP COLUMN {column}")
                connection.execute("ALTER TABLE graph_jobs DROP COLUMN job_scope")
                connection.execute("PRAGMA user_version=5")
                connection.commit()
            finally:
                connection.close()

            reopened = InMemoryRagService(database_path=path)
            try:
                self.assertTrue(reopened.store.graph_rag_enabled("existing-graph", "v1"))
                self.assertFalse(reopened.store.graph_rag_enabled("no-graph", "v1"))
                self.assertEqual(reopened.store.book_preparation_status("existing-graph", "v1")["graph_status"], "pending")
                self.assertIsNotNone(reopened.store.graph_job_by_id("legacy"))
            finally:
                reopened.close()

    def test_full_book_job_does_not_collide_with_a_final_boundary_job(self) -> None:
        extractor = RecordingLocalExtractor()
        service = InMemoryRagService(
            story_extractor=extractor,
            story_extractors={"ollama": extractor},
        )
        try:
            service.ingest_book_keyword_stage(
                "book", "v1", [SourceUnitInput.model_validate(item) for item in units()]
            )
            service.store.schedule_book_preparation(
                book_id="book", content_version_id="v1", source_units=units(),
                graph_rag_enabled=True,
            )
            final_chunk = service.store.get_book("book")[1][-1]
            final_position = final_chunk.source_positions[-1]
            boundary_job = service.store.schedule_graph_job(
                job_id="legacy-final-boundary", book_id="book", content_version_id="v1",
                boundary_section=final_chunk.section_index,
                boundary_unit=final_position.unit_index,
                boundary_source_unit_id=final_position.source_unit_id,
                total_chunks=3,
            )

            full_book_job = service.schedule_full_book_story_graph(
                book_id="book", content_version_id="v1"
            )

            self.assertNotEqual(boundary_job["job_id"], full_book_job["job_id"])
            self.assertEqual(full_book_job["job_scope"], "full_book")
            self.assertIsNone(full_book_job["boundary_section"])
            self.assertIsNone(full_book_job["boundary_unit"])
            self.assertIsNone(full_book_job["boundary_source_unit_id"])
        finally:
            service.close()

    def test_recovery_ignores_jobs_for_explicitly_disabled_books(self) -> None:
        service = InMemoryRagService()
        try:
            service.ingest_book_keyword_stage(
                "book", "v1", [SourceUnitInput.model_validate(item) for item in units(1)]
            )
            service.store.schedule_book_preparation(
                book_id="book", content_version_id="v1", source_units=units(1),
                graph_rag_enabled=False,
            )
            service.store.schedule_graph_job(
                job_id="disabled-legacy", book_id="book", content_version_id="v1",
                boundary_section=0, boundary_unit=0,
                boundary_source_unit_id="unit-0", total_chunks=1,
            )

            self.assertEqual(service.store.runnable_graph_jobs(), ())
            service.resume_graph_jobs()
            self.assertEqual(service.store.graph_job_by_id("disabled-legacy")["status"], "queued")
        finally:
            service.close()

    def test_keyword_retrieval_remains_available_when_graph_is_disabled(self) -> None:
        service = InMemoryRagService()
        try:
            service.ingest_book_keyword_stage(
                "book", "v1", [SourceUnitInput.model_validate(item) for item in units(1)]
            )
            service.store.schedule_book_preparation(
                book_id="book", content_version_id="v1", source_units=units(1),
                graph_rag_enabled=False,
            )

            results = service.search_book(
                book_id="book", query="story passage", retrieval_mode="keyword",
                reading_boundary=ReadingBoundary("book", 0, 0, "unit-0"),
            )

            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].chunk.id, service.store.get_book("book")[1][0].id)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
