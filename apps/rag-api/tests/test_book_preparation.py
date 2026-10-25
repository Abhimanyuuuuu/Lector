"""Durable, independent keyword and embedding preparation stages."""

from __future__ import annotations

import threading
import time
import unittest
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import EmbeddingIndexError, InMemoryRagService


class FakeEmbeddingProvider:
    dimensions = 2
    identity = "fake:v1:2"

    def __init__(self, *, block: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        if not block:
            self.release.set()
        self.calls = 0

    def embed_texts(self, texts):
        self.calls += len(texts)
        self.entered.set()
        if not self.release.wait(3):
            raise RuntimeError("test provider timed out")
        return [[1.0, 0.0] for _ in texts]


def payload(version="v1"):
    return {
        "content_version_id": version,
        "source_units": [
            {"source_unit_id":"u0","section_index":0,"unit_index":0,"kind":"paragraph","text":"Tess watches over the narrator."},
            {"source_unit_id":"u1","section_index":0,"unit_index":1,"kind":"paragraph","text":"She is two years younger."},
        ],
    }


class BookPreparationTests(unittest.TestCase):
    def test_async_prepare_preserves_content_version_conflict_response(self):
        service = InMemoryRagService()
        service.ingest_book_keyword_stage(
            "book", "v1", [SourceUnitInput.model_validate(item) for item in payload()["source_units"]]
        )
        changed = payload()
        changed["source_units"][0]["text"] = "Changed content with the old version id."
        with TestClient(create_app(service)) as client:
            response = client.post("/api/v1/books/book/prepare", json=changed)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "content_version_conflict")
        service.close()

    def test_explicit_delete_removes_book_scoped_database_and_epub_data(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"LECTOR_EPUB_STORAGE_DIR": directory}):
            service = InMemoryRagService()
            for book_id in ("book", "other"):
                service.ingest_book_keyword_stage(book_id, "v1", [SourceUnitInput.model_validate(item) for item in payload()["source_units"]])
                key = f"{book_id}.epub"
                Path(directory, key).write_bytes(b"local epub test")
                service.store.register_saved_epub(book_id=book_id, title=book_id, original_filename=f"{book_id}.epub", storage_key=key, content_version_id="v1", file_size=15)
                service.store.schedule_book_preparation(book_id=book_id, content_version_id="v1", source_units=payload()["source_units"])
            client = TestClient(create_app(service))
            with client:
                response = client.delete("/api/v1/books/book")
                self.assertEqual(response.status_code, 200)
            self.assertIsNone(service.store.get_book("book"))
            self.assertIsNone(service.store.saved_book("book"))
            self.assertIsNone(service.store.book_preparation_status("book", "v1"))
            self.assertFalse(Path(directory, "book.epub").exists())
            self.assertIsNotNone(service.store.get_book("other"))
            self.assertTrue(Path(directory, "other.epub").exists())
            service.close()

    def test_api_persists_independent_stages_and_keyword_is_ready_first(self):
        provider = FakeEmbeddingProvider(block=True)
        service = InMemoryRagService(embedding_provider=provider)
        client = TestClient(create_app(service))
        with client:
            response = client.post("/api/v1/books/book/prepare", json=payload())
            self.assertEqual(response.status_code, 202)
            self.assertTrue(provider.entered.wait(2))
            status = client.get("/api/v1/books/book/preparation", params={"content_version_id":"v1"}).json()
            self.assertEqual(status["indexing_status"], "complete")
            self.assertEqual(status["embedding_status"], "processing")
            boundary = {"confirmed":True,"section_index":0,"unit_index":1,"source_unit_id":"u1"}
            keyword = client.post("/api/v1/books/book/query", json={"query":"Tess","content_version_id":"v1","reading_boundary":boundary})
            self.assertEqual(keyword.status_code, 200)
            self.assertEqual(len(keyword.json()["results"]), 1)
            semantic = client.post("/api/v1/books/book/query", json={"query":"Tess","content_version_id":"v1","retrieval_mode":"semantic","reading_boundary":boundary})
            self.assertEqual(semantic.status_code, 503)
            self.assertEqual(semantic.json()["error"]["code"], "embedding_index_failed")
            provider.release.set()
            deadline = time.monotonic() + 3
            while time.monotonic() < deadline:
                status = client.get("/api/v1/books/book/preparation", params={"content_version_id":"v1"}).json()
                if status["embedding_status"] == "complete":
                    break
                time.sleep(0.02)
            self.assertEqual(status["embedding_status"], "complete")
            self.assertEqual(status["embedding_completed"], 1)
        service.close()

    def test_embedding_failure_keeps_keyword_chunks_and_retry_can_reuse_index(self):
        class FailingProvider(FakeEmbeddingProvider):
            def embed_texts(self, texts):
                raise RuntimeError("offline")

        failed = InMemoryRagService(embedding_provider=FailingProvider())
        chunks = failed.ingest_book_keyword_stage("book", "v1", [
            SourceUnitInput.model_validate(item) for item in payload()["source_units"]
        ])
        with self.assertRaises(RuntimeError):
            failed.embed_indexed_book("book", "v1")
        self.assertEqual(failed.store.get_book("book")[1], chunks)
        boundary = ReadingBoundary("book", 0, 1, "u1")
        with self.assertRaises(EmbeddingIndexError):
            failed.search_book(book_id="book", query="Tess", retrieval_mode="semantic", reading_boundary=boundary)
        self.assertTrue(failed.search_book(book_id="book", query="Tess", retrieval_mode="keyword", reading_boundary=boundary))
        failed.close()

    def test_preparation_state_survives_store_reopen(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lector.sqlite3"
            service = InMemoryRagService(database_path=path)
            service.store.schedule_book_preparation(book_id="book", content_version_id="v1", source_units=payload()["source_units"])
            service.store.update_book_preparation("book", "v1", indexing_status="complete", indexing_completed=2, indexing_total=2, embedding_status="failed", embedding_error="offline")
            service.close()
            reopened = InMemoryRagService(database_path=path)
            status = reopened.store.book_preparation_status("book", "v1")
            self.assertEqual(status["indexing_status"], "complete")
            self.assertEqual(status["embedding_status"], "failed")
            reopened.close()


if __name__ == "__main__":
    unittest.main()
