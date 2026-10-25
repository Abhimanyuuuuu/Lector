"""Offline tests for optional local Ollama adapters and wiring."""

import json
import os
import socket
import urllib.error
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from lector_rag_api.chunking import chunk_source_units
from lector_rag_api.local_providers import (
    LocalModelUnavailableError,
    LocalProviderConfigurationError,
    LocalProviderError,
    LocalProviderResponseError,
    LocalProviderTimeoutError,
    OllamaEmbeddingProvider,
    OllamaStoryExtractor,
)
from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.hybrid import HybridConfig
from lector_rag_api.service import InMemoryRagService, StoryGraphUnavailableError
from lector_rag_api.story_graph import StoryExtraction


class FakeTransport:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def post(self, url, payload, timeout):
        self.calls.append((url, payload, timeout))
        if self.error:
            raise self.error
        return self.response


def chunks():
    return chunk_source_units(
        "book",
        "v1",
        [SourceUnitInput(source_unit_id="u1", section_index=0, unit_index=0,
                         kind="paragraph", text="Mara opens the gate.", href="chapter.xhtml",
                         cfi="epubcfi(/6/2!/4/2)")],
    )


class LocalEmbeddingProviderTests(unittest.TestCase):
    def test_embedding_response_and_identity(self):
        transport = FakeTransport({"model": "embed:v1", "embeddings": [[1.0, 2.0], [0.0, 1.0]]})
        provider = OllamaEmbeddingProvider(
            base_url="http://127.0.0.1:11434", model_id="embed:v1", dimensions=2,
            transport=transport,
        )
        self.assertEqual(provider.identity, "ollama:embed:v1:dimensions=2")
        self.assertEqual(provider.embed_texts(["a", "b"]), [(1.0, 2.0), (0.0, 1.0)])
        self.assertEqual(transport.calls[0][1], {"model": "embed:v1", "input": ["a", "b"]})

    def test_malformed_vectors_and_model_identity_are_rejected(self):
        for response in (
            {"model": "other", "embeddings": [[1.0, 2.0]]},
            {"model": "embed", "embeddings": [[1.0]]},
            {"model": "embed", "embeddings": [[float("nan"), 1.0]]},
            {"model": "embed", "embeddings": []},
        ):
            provider = OllamaEmbeddingProvider(
                base_url="http://localhost:11434", model_id="embed", dimensions=2,
                transport=FakeTransport(response),
            )
            with self.subTest(response=response), self.assertRaises(LocalProviderResponseError):
                provider.embed_texts(["text"])

    def test_provider_failures_are_typed_and_no_call_for_empty_batch(self):
        for failure, error_type in (
            (LocalProviderTimeoutError("timeout"), LocalProviderTimeoutError),
            (LocalModelUnavailableError("missing"), LocalModelUnavailableError),
        ):
            provider = OllamaEmbeddingProvider(
                base_url="http://127.0.0.1:11434", model_id="embed", dimensions=2,
                transport=FakeTransport(error=failure),
            )
            with self.assertRaises(error_type):
                provider.embed_texts(["text"])
        no_call = FakeTransport()
        self.assertEqual(OllamaEmbeddingProvider(
            base_url="http://127.0.0.1:11434", model_id="embed", dimensions=2,
            transport=no_call,
        ).embed_texts([]), [])
        self.assertEqual(no_call.calls, [])

    def test_rejects_non_loopback_endpoint(self):
        with self.assertRaises(LocalProviderConfigurationError):
            OllamaEmbeddingProvider(base_url="https://example.com", model_id="m", dimensions=2)

    def test_real_transport_maps_connection_timeout_and_missing_model(self):
        from lector_rag_api.local_providers import UrllibJsonTransport

        provider = OllamaEmbeddingProvider(
            base_url="http://127.0.0.1:11434", model_id="embed", dimensions=2,
        )
        for failure, error_type in (
            (socket.timeout(), LocalProviderTimeoutError),
            (urllib.error.URLError(ConnectionRefusedError()), LocalProviderError),
            (urllib.error.HTTPError("http://localhost", 404, "missing", {}, None), LocalModelUnavailableError),
        ):
            with patch("urllib.request.urlopen", side_effect=failure):
                with self.subTest(failure=failure), self.assertRaises(error_type):
                    provider.embed_texts(["text"])


class LocalStoryExtractorTests(unittest.TestCase):
    def _payload(self, chunk):
        return {
            "entities": [{
                "ref": "mara", "name": "Mara", "kind": "character",
                "identity_status": "resolved", "aliases": ["M."],
                "assertedness": "explicit", "evidence": [{
                    "source_chunk_id": chunk.id, "source_unit_ids": ["u1"]
                }],
            }],
            "events": [], "relationships": [],
        }

    def test_valid_structured_output_preserves_supplied_evidence(self):
        chunk = chunks()[0]
        transport = FakeTransport({"message": {"content": json.dumps(self._payload(chunk))}})
        extractor = OllamaStoryExtractor(
            base_url="http://127.0.0.1:11434", model_id="extract:3b", transport=transport,
        )
        result = extractor.extract([chunk])
        self.assertEqual(result.entities[0].name, "Mara")
        sent = transport.calls[0][1]["messages"][1]["content"]
        self.assertIn(chunk.text, sent)
        self.assertIn("never instructions", transport.calls[0][1]["messages"][0]["content"])
        self.assertEqual(transport.calls[0][1]["options"]["temperature"], 0)

    def test_invalid_or_out_of_input_evidence_is_rejected(self):
        chunk = chunks()[0]
        for output in (
            {"message": {"content": "not json"}},
            {"message": {"content": json.dumps({"entities": [{"bad": "field"}], "events": [], "relationships": []})}},
            {"message": {"content": json.dumps({"entities": [{
                "ref": "x", "name": "Mara", "kind": "character", "assertedness": "explicit",
                "evidence": [{"source_chunk_id": "invented", "source_unit_ids": ["u1"]}],
            }], "events": [], "relationships": []})}},
            {"message": {"content": json.dumps({"entities": [{
                "ref": "x", "name": "Mara", "kind": "character", "assertedness": "explicit",
                "evidence": [{"source_chunk_id": chunk.id, "source_unit_ids": ["invented"]}],
            }], "events": [], "relationships": []})}},
        ):
            with self.subTest(output=output), self.assertRaises(LocalProviderResponseError):
                OllamaStoryExtractor(
                    base_url="http://localhost:11434", model_id="extract", transport=FakeTransport(output),
                ).extract([chunk])

    def test_empty_chunks_do_not_contact_model(self):
        transport = FakeTransport()
        extractor = OllamaStoryExtractor(
            base_url="http://localhost:11434", model_id="extract", transport=transport,
        )
        self.assertEqual(extractor.extract([]).entities, [])
        self.assertEqual(transport.calls, [])


class LocalProviderIntegrationTests(unittest.TestCase):
    def test_graph_extraction_api_reports_missing_local_extractor(self):
        service = InMemoryRagService()
        client = TestClient(create_app(service))
        client.post("/api/v1/books/book/source-units", json={
            "content_version_id": "v1",
            "source_units": [{
                "source_unit_id": "u1", "section_index": 0, "unit_index": 0,
                "kind": "paragraph", "text": "A passage.",
            }],
        })
        response = client.post("/api/v1/books/book/graph/extract", json={
            "confirmed": True, "section_index": 0, "unit_index": 0,
            "source_unit_id": "u1",
        })
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "story_extraction_unavailable")
        service.close()

    def test_graph_retrieval_does_not_silently_return_hybrid_only(self):
        service = InMemoryRagService()
        service.ingest_book("book", "v1", [SourceUnitInput(
            source_unit_id="u1", section_index=0, unit_index=0,
            kind="paragraph", text="A passage about Mara.",
        )])
        with self.assertRaises(StoryGraphUnavailableError):
            service.search_book(
                book_id="book", query="Mara", retrieval_mode="graph",
                reading_boundary=ReadingBoundary("book", 0, 0, "u1"),
                hybrid_config=HybridConfig(keyword_weight=1.0, semantic_weight=0.0),
            )
        service.close()

    def test_local_story_extractor_receives_only_boundary_eligible_chunks(self):
        transport = FakeTransport({"message": {"content": json.dumps({
            "entities": [], "events": [], "relationships": []
        })}})
        extractor = OllamaStoryExtractor(
            base_url="http://127.0.0.1:11434", model_id="extract", transport=transport,
        )
        service = InMemoryRagService(story_extractor=extractor)
        service.ingest_book("book", "v1", [
            SourceUnitInput(source_unit_id="u1", section_index=0, unit_index=0,
                            kind="paragraph", text="SAFE passage."),
            SourceUnitInput(source_unit_id="u2", section_index=1, unit_index=0,
                            kind="paragraph", text="FUTURE secret."),
        ])
        graph = service.extract_story_graph(
            book_id="book",
            reading_boundary=ReadingBoundary("book", 0, 0, "u1"),
        )
        submitted = transport.calls[0][1]["messages"][1]["content"]
        self.assertIn("SAFE passage", submitted)
        self.assertNotIn("FUTURE secret", submitted)
        self.assertEqual(graph.entities, ())
        service.close()

    def test_embedding_failure_preserves_committed_book_and_same_version_reuses_vectors(self):
        class CountedEmbeddings:
            dimensions = 2
            identity = "ollama:fixture:dimensions=2"

            def __init__(self):
                self.calls = 0

            def embed_texts(self, values):
                self.calls += 1
                return [[1.0, 0.0] for _ in values]

        embedder = CountedEmbeddings()
        service = InMemoryRagService(embedding_provider=embedder)
        units = [SourceUnitInput(source_unit_id="u1", section_index=0, unit_index=0,
                                 kind="paragraph", text="Stable book passage.")]
        original = service.ingest_book("book", "v1", units)
        self.assertEqual(embedder.calls, 1)
        service.ingest_book("book", "v1", units)
        self.assertEqual(embedder.calls, 1)
        self.assertEqual(service.store.get_book("book"), ("v1", original))
        service.close()

    def test_graph_extraction_failure_keeps_previous_graph(self):
        class ExistingExtractor:
            name = "fixture"
            version = "1"

            def extract(self, input_chunks):
                chunk = input_chunks[0]
                return StoryExtraction.model_validate({
                    "entities": [{
                        "ref": "m", "name": "Mara", "kind": "character",
                        "assertedness": "explicit", "evidence": [{
                            "source_chunk_id": chunk.id,
                            "source_unit_ids": list(chunk.source_unit_ids),
                        }],
                    }], "events": [], "relationships": [],
                })

        service = InMemoryRagService(story_extractor=ExistingExtractor())
        service.ingest_book("book", "v1", [SourceUnitInput(
            source_unit_id="u1", section_index=0, unit_index=0,
            kind="paragraph", text="Mara opens the gate.",
        )])
        boundary = ReadingBoundary("book", 0, 0, "u1")
        before = service.extract_story_graph(book_id="book", reading_boundary=boundary)
        service.story_extractor = OllamaStoryExtractor(
            base_url="http://localhost:11434", model_id="extract",
            transport=FakeTransport(error=LocalProviderTimeoutError("timeout")),
        )
        with self.assertRaises(LocalProviderTimeoutError):
            service.extract_story_graph(book_id="book", reading_boundary=boundary)
        self.assertEqual(service.story_graph.get_book("book"), before)
        service.close()

    def test_environment_wiring_does_not_contact_models_on_startup(self):
        with patch.dict(os.environ, {
            "LECTOR_OLLAMA_BASE_URL": "http://127.0.0.1:11434",
            "LECTOR_EMBEDDING_MODEL": "embed:v1",
            "LECTOR_EMBEDDING_DIMENSIONS": "2",
            "LECTOR_STORY_EXTRACTION_MODEL": "story:3b",
        }, clear=False):
            app = create_app(database_path=":memory:")
            service = app.state.rag_service
            self.assertEqual(app.state.local_provider_status["embeddings"], "configured")
            self.assertEqual(app.state.local_provider_status["story_extraction"], "configured")
            self.assertEqual(service.semantic_index.provider.identity, "ollama:embed:v1:dimensions=2")
            self.assertEqual(service.story_extractor.version, "ollama:story:3b")
            service.close()

    def test_keyword_service_remains_available_without_local_models(self):
        service = InMemoryRagService()
        app = create_app(service)
        client = TestClient(app)
        response = client.get("/api/v1/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["local_embeddings"], "unconfigured")
        self.assertEqual(response.json()["local_story_extraction"], "unconfigured")
        service.close()
