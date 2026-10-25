"""Cross-module tests for ingestion, provider injection, and query contracts."""

import unittest
from collections.abc import Sequence

from fastapi.testclient import TestClient

from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService


class FixtureEmbeddingProvider:
    dimensions = 2

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [self.vectors[text] for text in texts]


def source_unit(unit_id: str, index: int, text: str) -> dict[str, object]:
    return {
        "source_unit_id": unit_id,
        "section_index": 0,
        "unit_index": index,
        "kind": "paragraph",
        "text": text,
        "href": "text/chapter.xhtml",
        "cfi": f"epubcfi(/6/2!/4/2/{index + 1}:0,/1:12)",
    }


def confirmed_boundary(book_id: str, source_id: str, index: int) -> dict[str, object]:
    return {
        "book_id": book_id,
        "confirmed": True,
        "section_index": 0,
        "unit_index": index,
        "source_unit_id": source_id,
    }


def ingest(client: TestClient, book_id: str, version: str, units: list[dict[str, object]]):
    return client.post(
        f"/api/v1/books/{book_id}/source-units",
        json={"content_version_id": version, "source_units": units},
    )


class RetrievalApiTests(unittest.TestCase):
    def test_ingestion_chunks_for_keyword_and_semantic_indexes(self) -> None:
        provider = FixtureEmbeddingProvider(
            {
                "Mara found the silver key.": [1, 0],
                "The old gate opened.": [0, 1],
                "Mara found the silver key.\n\nThe old gate opened.": [1, 0],
                "Who found a key?": [1, 0],
            }
        )
        service = InMemoryRagService(embedding_provider=provider)
        client = TestClient(create_app(service))
        response = ingest(
            client,
            "novel",
            "rev-1",
            [
                source_unit("u1", 0, "Mara found the silver key."),
                source_unit("u2", 1, "The old gate opened."),
            ],
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(response.json()["chunk_count"], 1)
        self.assertEqual(service.store.get_book("novel")[0], "rev-1")

        keyword = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "Mara key",
                "retrieval_mode": "keyword",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 1,
                    "source_unit_id": "u2",
                },
            },
        )
        semantic = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "Who found a key?",
                "retrieval_mode": "semantic",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 1,
                    "source_unit_id": "u2",
                },
            },
        )

        self.assertEqual(keyword.status_code, 200)
        self.assertEqual(semantic.status_code, 200)
        keyword_result = keyword.json()["results"][0]
        semantic_result = semantic.json()["results"][0]
        self.assertEqual(keyword_result["chunk"]["id"], semantic_result["chunk"]["id"])
        self.assertEqual(keyword_result["chunk"]["source_unit_ids"], ["u1", "u2"])
        self.assertEqual(semantic_result["retrieval_method"], "semantic")
        self.assertIn("cfi", semantic_result["chunk"]["source_positions"][0])

    def test_reingestion_replaces_keyword_and_semantic_state_together(self) -> None:
        provider = FixtureEmbeddingProvider(
            {
                "oldword passage": [1, 0],
                "newword passage": [0, 1],
                "oldword": [1, 0],
                "newword": [0, 1],
            }
        )
        service = InMemoryRagService(embedding_provider=provider)
        client = TestClient(create_app(service))
        ingest(client, "novel", "v1", [source_unit("old", 0, "oldword passage")])
        ingest(client, "novel", "v2", [source_unit("new", 0, "newword passage")])
        current_boundary = confirmed_boundary("novel", "new", 0)

        old_keyword = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "oldword",
                "reading_boundary": {key: value for key, value in current_boundary.items() if key != "book_id"},
            },
        )
        new_keyword = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "newword",
                "reading_boundary": {key: value for key, value in current_boundary.items() if key != "book_id"},
            },
        )
        new_semantic = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "newword",
                "retrieval_mode": "semantic",
                "reading_boundary": {key: value for key, value in current_boundary.items() if key != "book_id"},
            },
        )
        new_hybrid = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "newword",
                "retrieval_mode": "hybrid",
                "reading_boundary": {
                    key: value
                    for key, value in current_boundary.items()
                    if key != "book_id"
                },
            },
        )

        self.assertEqual(old_keyword.json()["results"], [])
        self.assertEqual(new_keyword.json()["results"][0]["chunk"]["content_version_id"], "v2")
        self.assertEqual(new_semantic.json()["results"][0]["chunk"]["source_unit_ids"], ["new"])
        self.assertEqual(new_hybrid.json()["results"][0]["chunk"]["content_version_id"], "v2")
        self.assertEqual(service.store.get_book("novel")[1][0].source_unit_ids, ("new",))

    def test_semantic_and_hybrid_modes_without_provider_fail_explicitly(self) -> None:
        service = InMemoryRagService()
        client = TestClient(create_app(service))
        ingest(client, "novel", "v1", [source_unit("u1", 0, "A passage.")])
        body = {
            "query": "passage",
            "reading_boundary": {
                "confirmed": True,
                "section_index": 0,
                "unit_index": 0,
                "source_unit_id": "u1",
            },
        }

        semantic = client.post(
            "/api/v1/books/novel/query",
            json={**body, "retrieval_mode": "semantic"},
        )
        hybrid = client.post(
            "/api/v1/books/novel/query",
            json={**body, "retrieval_mode": "hybrid"},
        )
        explicitly_keyword_only = client.post(
            "/api/v1/books/novel/query",
            json={
                **body,
                "retrieval_mode": "hybrid",
                "hybrid_options": {"semantic_weight": 0},
            },
        )

        self.assertEqual(semantic.status_code, 503)
        self.assertEqual(semantic.json()["error"]["code"], "semantic_unavailable")
        self.assertEqual(hybrid.status_code, 503)
        self.assertEqual(hybrid.json()["error"]["code"], "semantic_unavailable")
        self.assertEqual(explicitly_keyword_only.status_code, 200)
        fused = explicitly_keyword_only.json()["results"][0]
        self.assertEqual(fused["retrieval_method"], "hybrid")
        self.assertEqual(fused["keyword_rank"], 1)
        self.assertIsNone(fused["semantic_rank"])

    def test_hybrid_query_returns_component_ranks_and_fusion_score(self) -> None:
        text = "Mara found the silver key."
        provider = FixtureEmbeddingProvider(
            {text: [1, 0], "Who found the key?": [1, 0]}
        )
        client = TestClient(create_app(InMemoryRagService(embedding_provider=provider)))
        ingest(client, "novel", "v1", [source_unit("u1", 0, text)])

        response = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "Who found the key?",
                "retrieval_mode": "hybrid",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
                "hybrid_options": {
                    "k": 10,
                    "keyword_weight": 2,
                    "semantic_weight": 1,
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["retrieval_mode"], "hybrid")
        result = body["results"][0]
        self.assertEqual(result["retrieval_method"], "hybrid")
        self.assertEqual((result["keyword_rank"], result["semantic_rank"]), (1, 1))
        self.assertIsNotNone(result["keyword_score"])
        self.assertEqual(result["semantic_score"], 1)
        self.assertAlmostEqual(result["score"], 3 / 11)
        self.assertEqual(result["chunk"]["source_unit_ids"], ["u1"])

    def test_hybrid_request_rejects_invalid_fusion_configuration(self) -> None:
        client = TestClient(create_app(InMemoryRagService()))
        response = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "question",
                "retrieval_mode": "hybrid",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
                "hybrid_options": {
                    "keyword_weight": 0,
                    "semantic_weight": 0,
                },
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_hybrid_boundary_excludes_future_chunks_from_component_ranks(self) -> None:
        early = "needle " + ("early " * 220)
        future = "needle " * 40 + "future revelation"
        provider = FixtureEmbeddingProvider(
            {
                early: [0, 1],
                future: [1, 0],
                early: [0, 1],
                f"{early}\n\n{future}": [0, 1],
                "needle": [1, 0],
            }
        )
        service = InMemoryRagService(embedding_provider=provider)
        client = TestClient(create_app(service))
        ingest(
            client,
            "novel",
            "v1",
            [source_unit("early", 0, early), source_unit("future", 1, future)],
        )
        response = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "needle",
                "retrieval_mode": "hybrid",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "early",
                },
            },
        )

        self.assertEqual(response.status_code, 200, response.text)
        results = response.json()["results"]
        self.assertTrue(results)
        self.assertEqual(
            {source_id for item in results for source_id in item["chunk"]["source_unit_ids"]},
            {"early"},
        )
        self.assertTrue(all(item["keyword_rank"] == 1 for item in results))
        self.assertTrue(all(item["semantic_rank"] == 1 for item in results))

    def test_hybrid_excludes_a_chunk_that_crosses_the_boundary(self) -> None:
        combined = "The clue appears.\n\nThe later answer is hidden."
        provider = FixtureEmbeddingProvider(
            {combined: [1, 0], "clue": [1, 0]}
        )
        client = TestClient(create_app(InMemoryRagService(embedding_provider=provider)))
        ingest(
            client,
            "novel",
            "v1",
            [
                source_unit("seen", 0, "The clue appears."),
                source_unit("unread", 1, "The later answer is hidden."),
            ],
        )
        response = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "clue",
                "retrieval_mode": "hybrid",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "seen",
                },
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["results"], [])

    def test_all_books_requires_explicit_opt_in_and_boundary_for_each_book(self) -> None:
        provider = FixtureEmbeddingProvider(
            {
                "shared apple": [1, 0],
                "shared banana": [0, 1],
                "shared": [1, 0],
            }
        )
        service = InMemoryRagService(embedding_provider=provider)
        client = TestClient(create_app(service))
        ingest(client, "a", "v1", [source_unit("a1", 0, "shared apple")])
        ingest(client, "b", "v1", [source_unit("b1", 0, "shared banana")])
        body = {
            "query": "shared",
            "retrieval_mode": "hybrid",
            "all_books": True,
            "reading_boundaries": [
                confirmed_boundary("a", "a1", 0),
                confirmed_boundary("b", "b1", 0),
            ],
        }

        response = client.post("/api/v1/query", json=body)
        missing = client.post(
            "/api/v1/query",
            json={
                **body,
                "reading_boundaries": [confirmed_boundary("a", "a1", 0)],
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {item["chunk"]["book_id"] for item in response.json()["results"]},
            {"a", "b"},
        )
        self.assertTrue(
            all(item["retrieval_method"] == "hybrid" for item in response.json()["results"])
        )
        self.assertEqual(missing.status_code, 422)
        self.assertEqual(missing.json()["error"]["code"], "invalid_reading_boundary")

    def test_invalid_boundary_and_hybrid_errors_are_structured(self) -> None:
        client = TestClient(create_app(InMemoryRagService()))
        ingest(client, "novel", "v1", [source_unit("u1", 0, "A passage.")])
        wrong_boundary = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "passage",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "future-or-unknown",
                },
            },
        )
        malformed = client.post(
            "/api/v1/books/novel/query",
            json={
                "query": "passage",
                "reading_boundary": {
                    "confirmed": False,
                    "section_index": 0,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
            },
        )

        self.assertEqual(wrong_boundary.status_code, 422)
        self.assertEqual(wrong_boundary.json()["error"]["code"], "invalid_reading_boundary")
        self.assertEqual(malformed.status_code, 422)
        self.assertEqual(malformed.json()["error"]["code"], "validation_error")

    def test_failed_embedding_build_does_not_partially_replace_book(self) -> None:
        class FailingProvider(FixtureEmbeddingProvider):
            def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
                if "new passage" in texts:
                    raise RuntimeError("test provider failure")
                return super().embed_texts(texts)

        provider = FailingProvider(
            {"old passage": [1, 0], "old": [1, 0]}
        )
        service = InMemoryRagService(embedding_provider=provider)
        old_chunks = service.ingest_book(
            "novel",
            "v1",
            [
                SourceUnitInput(
                    source_unit_id="old",
                    section_index=0,
                    unit_index=0,
                    kind="paragraph",
                    text="old passage",
                )
            ],
        )
        with self.assertRaises(RuntimeError):
            service.ingest_book(
                "novel",
                "v2",
                [
                    SourceUnitInput(
                        source_unit_id="new",
                        section_index=0,
                        unit_index=0,
                        kind="paragraph",
                        text="new passage",
                    )
                ],
            )

        stored = service.store.get_book("novel")
        self.assertEqual(stored[0], "v1")
        self.assertIs(stored[1][0], old_chunks[0])
        boundary = ReadingBoundary("novel", 0, 0, "old")
        self.assertEqual(
            service.search_book(
                book_id="novel",
                query="old",
                retrieval_mode="keyword",
                reading_boundary=boundary,
            )[0].chunk.source_unit_ids,
            ("old",),
        )
        self.assertEqual(
            service.search_book(
                book_id="novel",
                query="old",
                retrieval_mode="semantic",
                reading_boundary=boundary,
            )[0].chunk.source_unit_ids,
            ("old",),
        )


if __name__ == "__main__":
    unittest.main()
