"""Contract tests for the initial Lector API scaffold."""

import unittest

from fastapi.testclient import TestClient

from lector_rag_api.main import create_app
from lector_rag_api.service import InMemoryRagService


class ApiScaffoldTests(unittest.TestCase):
    def setUp(self) -> None:
        self.service = InMemoryRagService()
        self.client = TestClient(create_app(self.service))

    def test_health_reports_readiness(self) -> None:
        response = self.client.get("/api/v1/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ready",
                "service": "lector-rag-api",
                "api_version": "v1",
                "local_embeddings": "unconfigured",
                "local_story_extraction": "unconfigured",
            },
        )

    def test_reader_development_origin_can_make_json_preflight_request(self) -> None:
        response = self.client.options(
            "/api/v1/answer",
            headers={
                "Origin": "http://localhost:7127",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.headers["access-control-allow-origin"],
            "http://localhost:7127",
        )
        self.assertIn("POST", response.headers["access-control-allow-methods"])

    def test_reader_origin_can_preflight_local_epub_upload(self) -> None:
        response = self.client.options(
            "/api/v1/books/book/epub",
            headers={
                "Origin": "http://127.0.0.1:7127",
                "Access-Control-Request-Method": "PUT",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://127.0.0.1:7127")
        self.assertIn("PUT", response.headers["access-control-allow-methods"])

    def test_source_unit_contract_rejects_duplicate_ids(self) -> None:
        response = self.client.post(
            "/api/v1/books/book-1/source-units",
            json={
                "content_version_id": "version-1",
                "source_units": [
                    {
                        "source_unit_id": "unit-1",
                        "section_index": 0,
                        "unit_index": 0,
                        "kind": "paragraph",
                        "text": "First.",
                    },
                    {
                        "source_unit_id": "unit-1",
                        "section_index": 0,
                        "unit_index": 1,
                        "kind": "paragraph",
                        "text": "Second.",
                    },
                ],
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_ingestion_validates_and_chunks_source_units(
        self,
    ) -> None:
        response = self.client.post(
            "/api/v1/books/book-1/source-units",
            json={
                "content_version_id": "version-1",
                "source_units": [
                    {
                        "source_unit_id": "unit-1",
                        "section_index": 0,
                        "unit_index": 0,
                        "kind": "paragraph",
                        "text": "A source passage.",
                        "href": "text/chapter-1.xhtml",
                        "cfi": "epubcfi(/6/2[chapter01]!/4/2/1:0,/1:16)",
                    }
                ],
            },
        )

        self.assertEqual(response.status_code, 202)
        self.assertEqual(
            response.json(),
            {
                "book_id": "book-1",
                "content_version_id": "version-1",
                "accepted_source_units": 1,
                "chunk_count": 1,
            },
        )

    def test_query_requires_explicit_confirmed_boundary(self) -> None:
        response = self.client.post(
            "/api/v1/books/book-1/query",
            json={
                "query": "Who is Mara?",
                "reading_boundary": {
                    "section_index": 2,
                    "unit_index": 18,
                },
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_query_returns_keyword_retrieval_results(self) -> None:
        self.client.post(
            "/api/v1/books/book-1/source-units",
            json={
                "content_version_id": "version-1",
                "source_units": [
                    {
                        "source_unit_id": "unit-18",
                        "section_index": 2,
                        "unit_index": 18,
                        "kind": "paragraph",
                        "text": "Mara found the silver key.",
                    }
                ],
            },
        )
        response = self.client.post(
            "/api/v1/books/book-1/query",
            json={
                "query": "Who is Mara?",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 2,
                    "unit_index": 18,
                    "source_unit_id": "unit-18",
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        result = response.json()["results"][0]
        self.assertEqual(result["retrieval_method"], "keyword")
        self.assertEqual(result["chunk"]["source_unit_ids"], ["unit-18"])
        self.assertIn("mara", result["matched_terms"])

    def test_ingestion_replaces_previous_in_memory_content_version(self) -> None:
        def send(version: str, unit_id: str, text: str) -> None:
            response = self.client.post(
                "/api/v1/books/replacement-test/source-units",
                json={
                    "content_version_id": version,
                    "source_units": [
                        {
                            "source_unit_id": unit_id,
                            "section_index": 0,
                            "unit_index": 0,
                            "kind": "paragraph",
                            "text": text,
                        }
                    ],
                },
            )
            self.assertEqual(response.status_code, 202)

        send("old-version", "old-unit", "Old contents")
        send("new-version", "new-unit", "New contents")

        stored = self.service.store.get_book("replacement-test")
        self.assertIsNotNone(stored)
        self.assertEqual(stored[0], "new-version")
        self.assertEqual(len(stored[1]), 1)
        self.assertEqual(stored[1][0].source_unit_ids, ("new-unit",))


if __name__ == "__main__":
    unittest.main()
