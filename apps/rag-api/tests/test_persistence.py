"""Restart, migration, transaction, and safety tests for SQLite storage."""

import json
import sqlite3
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from fastapi.testclient import TestClient

from lector_rag_api.chunking import BookChunk
from lector_rag_api.answering import generate_grounded_answer
from lector_rag_api.context import ContextOptions, assemble_context
from lector_rag_api.main import create_app
from lector_rag_api.persistence import ContentVersionConflictError
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import EmbeddingIndexError, InMemoryRagService
from lector_rag_api.story_graph import StoryExtraction


class FakeEmbeddings:
    provider_id = "fixture-model-v1"
    dimensions = 2

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [[1.0, 0.0] for _ in texts]


class FakeExtractor:
    name = "fixture"
    version = "1"

    def extract(self, chunks: Sequence[BookChunk]) -> StoryExtraction:
        chunk = chunks[0]
        return StoryExtraction.model_validate(
            {
                "entities": [
                    {
                        "ref": "hero",
                        "name": "Mara",
                        "kind": "character",
                        "aliases": ["M."],
                        "assertedness": "explicit",
                        "evidence": [
                            {
                                "source_chunk_id": chunk.id,
                                "source_unit_ids": list(chunk.source_unit_ids),
                            }
                        ],
                    },
                    {
                        "ref": "garden",
                        "name": "Old Garden",
                        "kind": "location",
                        "assertedness": "explicit",
                        "evidence": [
                            {
                                "source_chunk_id": chunk.id,
                                "source_unit_ids": list(chunk.source_unit_ids),
                            }
                        ],
                    },
                ],
                "events": [
                    {
                        "ref": "arrival",
                        "name": "Mara arrives at the garden",
                        "participant_refs": ["hero"],
                        "assertedness": "explicit",
                        "evidence": [
                            {
                                "source_chunk_id": chunk.id,
                                "source_unit_ids": list(chunk.source_unit_ids),
                            }
                        ],
                    }
                ],
                "relationships": [
                    {
                        "subject_ref": "hero",
                        "object_ref": "garden",
                        "relationship_type": "visited",
                        "assertedness": "inferred",
                        "evidence": [
                            {
                                "source_chunk_id": chunk.id,
                                "source_unit_ids": list(chunk.source_unit_ids),
                            }
                        ],
                    }
                ],
            }
        )


class CitingAnswerProvider:
    def generate(self, *, question: str, context_json: str) -> str:
        evidence = json.loads(context_json)
        return json.dumps(
            {
                "answer": "Mara entered the garden.",
                "citations": [evidence["passages"][0]["passage_id"]],
                "insufficient_evidence": False,
                "uncertainty": None,
            }
        )


def units(text: str = "Mara entered the old garden.") -> list[SourceUnitInput]:
    return [
        SourceUnitInput(
            source_unit_id="u1",
            section_index=0,
            unit_index=0,
            kind="paragraph",
            text=text,
            href="chapter.xhtml",
            cfi="epubcfi(/6/2!/4/2/1:0,/1:10)",
        )
    ]


def boundary() -> ReadingBoundary:
    return ReadingBoundary("book", 0, 0, "u1")


class SQLitePersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.path = Path(self.temp.name) / "lector.sqlite3"
        self.services: list[InMemoryRagService] = []

    def tearDown(self) -> None:
        for service in self.services:
            try:
                service.close()
            except sqlite3.ProgrammingError:
                pass
        self.temp.cleanup()

    def make_service(self, **kwargs) -> InMemoryRagService:
        service = InMemoryRagService(database_path=self.path, **kwargs)
        self.services.append(service)
        return service

    def test_missing_database_initializes_and_reopen_restores_chunks_graph_and_vectors(self) -> None:
        provider = FakeEmbeddings()
        service = self.make_service(
            embedding_provider=provider,
            story_extractor=FakeExtractor(),
        )
        chunks = service.ingest_book("book", "v1", units())
        graph = service.extract_story_graph(book_id="book", reading_boundary=boundary())
        self.assertTrue(self.path.exists())
        service.close()

        restarted_provider = FakeEmbeddings()
        reopened = self.make_service(embedding_provider=restarted_provider)
        restored = reopened.store.get_book("book")
        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertEqual(restored[0], "v1")
        self.assertEqual(restored[1], chunks)
        self.assertEqual(reopened.story_graph.get_book("book"), graph)
        self.assertEqual(restarted_provider.calls, [])
        keyword = reopened.search_book(
            book_id="book",
            query="garden",
            retrieval_mode="keyword",
            reading_boundary=boundary(),
        )
        semantic = reopened.search_book(
            book_id="book",
            query="where",
            retrieval_mode="semantic",
            reading_boundary=boundary(),
        )
        self.assertEqual(keyword[0].chunk.id, chunks[0].id)
        self.assertEqual(semantic[0].chunk.id, chunks[0].id)
        self.assertEqual(restarted_provider.calls, [("where",)])
        entities, events, relationships = reopened.eligible_story_facts(
            book_id="book", reading_boundary=boundary()
        )
        self.assertEqual(len(entities), 2)
        self.assertEqual(next(item for item in entities if item.name == "Mara").alias_names, ("M.",))
        self.assertEqual(len(events), 1)
        self.assertEqual(len(relationships), 1)
        self.assertEqual(relationships[0].assertedness, "inferred")

    def test_same_version_is_idempotent_and_new_version_removes_stale_graph_and_vectors(self) -> None:
        provider = FakeEmbeddings()
        service = self.make_service(embedding_provider=provider, story_extractor=FakeExtractor())
        service.ingest_book("book", "v1", units())
        service.extract_story_graph(book_id="book", reading_boundary=boundary())
        first_graph = service.story_graph.get_book("book")
        service.ingest_book("book", "v1", units())
        self.assertEqual(service.story_graph.get_book("book"), first_graph)
        with self.assertRaisesRegex(ContentVersionConflictError, "use a new content version"):
            service.ingest_book("book", "v1", units("Different content."))

        service.ingest_book("book", "v2", units("Jonas left the city."))
        self.assertEqual(service.story_graph.get_book("book").content_version_id, "v2")
        self.assertEqual(service.story_graph.get_book("book").entities, ())
        service.close()
        restored = self.make_service(embedding_provider=FakeEmbeddings())
        self.assertEqual(restored.store.get_book("book")[0], "v2")
        self.assertEqual(restored.story_graph.get_book("book").entities, ())
        result = restored.search_book(
            book_id="book",
            query="who",
            retrieval_mode="semantic",
            reading_boundary=boundary(),
        )
        self.assertEqual(result[0].chunk.text, "Jonas left the city.")

    def test_failed_sql_write_rolls_back_database_and_live_state(self) -> None:
        service = self.make_service()
        old_chunks = service.ingest_book("book", "v1", units())
        with service.store._connection:
            service.store._connection.execute(
                "CREATE TRIGGER reject_new_chunks BEFORE INSERT ON chunks "
                "BEGIN SELECT RAISE(ABORT, 'injected write failure'); END"
            )
        with self.assertRaises(sqlite3.IntegrityError):
            service.ingest_book("book", "v2", units("New version."))
        self.assertEqual(service.store.get_book("book"), ("v1", old_chunks))
        service.close()
        reopened = self.make_service()
        self.assertEqual(reopened.store.get_book("book"), ("v1", old_chunks))

    def test_vectors_with_wrong_identity_or_dimensions_are_not_used(self) -> None:
        service = self.make_service(embedding_provider=FakeEmbeddings())
        service.ingest_book("book", "v1", units())
        service.ingest_book("healthy", "v1", units())
        service.close()

        connection = sqlite3.connect(self.path)
        try:
            connection.execute(
                "UPDATE semantic_vectors SET vector_json = ? WHERE book_id = 'book'",
                (json.dumps([1.0]),),
            )
            connection.commit()
        finally:
            connection.close()
        reopened = self.make_service(embedding_provider=FakeEmbeddings())
        healthy_result = reopened.search_book(
            book_id="healthy",
            query="q",
            retrieval_mode="semantic",
            reading_boundary=ReadingBoundary("healthy", 0, 0, "u1"),
        )
        self.assertTrue(healthy_result)
        with self.assertRaises(EmbeddingIndexError):
            reopened.search_book(
                book_id="book",
                query="q",
                retrieval_mode="semantic",
                reading_boundary=boundary(),
            )

        reopened.close()
        mismatched = FakeEmbeddings()
        mismatched.provider_id = "different-model"
        mismatch_service = self.make_service(embedding_provider=mismatched)
        with self.assertRaises(EmbeddingIndexError):
            mismatch_service.search_book(
                book_id="book",
                query="q",
                retrieval_mode="semantic",
                reading_boundary=boundary(),
            )

    def test_stale_graph_evidence_fails_closed_during_recovery(self) -> None:
        service = self.make_service(story_extractor=FakeExtractor())
        service.ingest_book("book", "v1", units())
        service.extract_story_graph(book_id="book", reading_boundary=boundary())
        service.close()
        connection = sqlite3.connect(self.path)
        try:
            row = connection.execute("SELECT payload FROM graph_snapshots WHERE book_id='book'").fetchone()
            payload = json.loads(row[0])
            payload["entities"][0]["evidence"][0]["source_chunk_id"] = "stale-chunk"
            connection.execute(
                "UPDATE graph_snapshots SET payload=? WHERE book_id='book'",
                (json.dumps(payload),),
            )
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(RuntimeError, "missing canonical source"):
            InMemoryRagService(database_path=self.path)

    def test_schema_version_and_database_scoped_boundary_recovery(self) -> None:
        service = self.make_service()
        service.ingest_book("book", "v1", units())
        service.ingest_book("other", "other-v1", units("A separate book."))
        service.close()
        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 5)
        finally:
            connection.close()
        reopened = self.make_service()
        self.assertEqual(reopened.store.book_ids, ("book", "other"))
        with self.assertRaisesRegex(ValueError, "every indexed book"):
            reopened.search_all_books(
                query="book",
                retrieval_mode="keyword",
                boundaries={"book": boundary()},
            )
        results = reopened.search_all_books(
            query="book",
            retrieval_mode="keyword",
            boundaries={
                "book": boundary(),
                "other": ReadingBoundary("other", 0, 0, "u1"),
            },
        )
        self.assertTrue(results)

    def test_v4_migration_seeds_existing_book_progress_without_reindexing(self) -> None:
        service = self.make_service()
        chunks = service.ingest_book("book", "v1", units())
        service.close()
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("DROP TABLE book_preparation_jobs")
            connection.execute("PRAGMA user_version = 4")
            connection.commit()
        finally:
            connection.close()

        reopened = self.make_service(embedding_provider=FakeEmbeddings())
        self.assertEqual(reopened.store.get_book("book"), ("v1", chunks))
        status = reopened.store.book_preparation_status("book", "v1")
        self.assertIsNotNone(status)
        self.assertEqual(status["indexing_status"], "complete")
        self.assertEqual(status["indexing_completed"], 1)
        self.assertEqual(status["indexing_total"], 1)
        self.assertEqual(status["embedding_status"], "pending")

    def test_newer_schema_is_rejected_without_resetting_existing_data(self) -> None:
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("CREATE TABLE preserved_data (value TEXT)")
            connection.execute("INSERT INTO preserved_data(value) VALUES ('keep')")
            connection.execute("PRAGMA user_version = 99")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaisesRegex(RuntimeError, "newer than supported"):
            InMemoryRagService(database_path=self.path)
        connection = sqlite3.connect(self.path)
        try:
            self.assertEqual(
                connection.execute("SELECT value FROM preserved_data").fetchone()[0],
                "keep",
            )
            self.assertEqual(connection.execute("PRAGMA user_version").fetchone()[0], 99)
        finally:
            connection.close()

    def test_recovered_chunks_keep_boundary_and_context_provenance(self) -> None:
        service = self.make_service()
        chunks = service.ingest_book("book", "v1", units())
        service.close()
        reopened = self.make_service()
        results = reopened.search_book(
            book_id="book",
            query="garden",
            retrieval_mode="keyword",
            reading_boundary=boundary(),
        )
        context = assemble_context(
            results=results,
            chunks_by_book={"book": reopened.store.get_book("book")[1]},
            graphs={},
            boundaries={"book": boundary()},
            options=ContextOptions(),
        )
        answer = generate_grounded_answer(
            provider=CitingAnswerProvider(), question="What happened?", context=context
        )
        self.assertEqual(context.passages[0].chunk.id, chunks[0].id)
        self.assertEqual(context.passages[0].chunk.source_positions[0].cfi, units()[0].cfi)
        self.assertEqual(answer.status, "answered")
        self.assertEqual(answer.citations[0].passage.chunk.id, chunks[0].id)
        reopened.close()

    def test_confirmed_boundary_filters_recovered_future_chunks(self) -> None:
        source_units = [
            SourceUnitInput(
                source_unit_id="early",
                section_index=0,
                unit_index=0,
                kind="paragraph",
                text="safe opening " + ("ordinary text " * 50),
            ),
            SourceUnitInput(
                source_unit_id="future",
                section_index=0,
                unit_index=1,
                kind="paragraph",
                text="future reveal keyword " + ("later text " * 60),
            ),
        ]
        service = self.make_service()
        chunks = service.ingest_book("book", "v1", source_units)
        self.assertGreater(len(chunks), 1)
        service.close()
        reopened = self.make_service()
        results = reopened.search_book(
            book_id="book",
            query="future reveal keyword",
            retrieval_mode="keyword",
            reading_boundary=ReadingBoundary("book", 0, 0, "early"),
        )
        self.assertEqual(results, [])
        reopened.close()

    def test_versioned_api_factory_uses_durable_store_across_restart(self) -> None:
        first_app = create_app(database_path=str(self.path))
        first_client = TestClient(first_app)
        indexed = first_client.post(
            "/api/v1/books/book/source-units",
            json={
                "content_version_id": "v1",
                "source_units": [
                    {
                        "source_unit_id": "u1",
                        "section_index": 0,
                        "unit_index": 0,
                        "kind": "paragraph",
                        "text": "Mara entered the garden.",
                    }
                ],
            },
        )
        self.assertEqual(indexed.status_code, 202)
        first_app.state.rag_service.close()

        second_app = create_app(database_path=str(self.path))
        second_client = TestClient(second_app)
        queried = second_client.post(
            "/api/v1/books/book/query",
            json={
                "query": "garden",
                "retrieval_mode": "keyword",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
            },
        )
        self.assertEqual(queried.status_code, 200)
        self.assertEqual(
            queried.json()["results"][0]["chunk"]["source_unit_ids"], ["u1"]
        )
        second_app.state.rag_service.close()


if __name__ == "__main__":
    unittest.main()
