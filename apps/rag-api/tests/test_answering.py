"""Offline answer-context, citation-validation, provider, and API tests."""

import json
import os
import tempfile
import urllib.error
import unittest
from dataclasses import replace
from typing import Sequence

from fastapi.testclient import TestClient

from lector_rag_api.answering import (
    GenerationAuthenticationError,
    GenerationConfigurationError,
    GenerationMalformedResponseError,
    GenerationRateLimitError,
    GenerationStructuredOutputError,
    GenerationTimeoutError,
    GenerationUnavailableError,
    GroqAnswerProvider,
    generate_grounded_answer,
)
from lector_rag_api.context import ContextOptions, assemble_context
from lector_rag_api.hybrid import HybridConfig
from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary, RetrievalResult
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService
from lector_rag_api.chunking import BookChunk, ChunkSourcePosition


def unit(unit_id: str, section: int, text: str) -> SourceUnitInput:
    return SourceUnitInput(
        source_unit_id=unit_id,
        section_index=section,
        unit_index=0,
        kind="paragraph",
        text=text,
        href=f"chapter-{section}.xhtml",
        cfi=f"epubcfi(/6/{section + 2}!/4/2/1:0,/1:{len(text)})",
    )


class FakeProvider:
    def __init__(self, *, response: str | None = None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def generate(self, *, question: str, context_json: str) -> str:
        self.calls.append((question, context_json))
        if self.error is not None:
            raise self.error
        if self.response is not None:
            return self.response
        evidence = json.loads(context_json)
        return json.dumps(
            {
                "answer": "Mara found a silver key.",
                "citations": [evidence["passages"][0]["passage_id"]],
                "insufficient_evidence": False,
                "uncertainty": None,
            }
        )


class FakeExtractor:
    name = "fixture"
    version = "1"

    def extract(self, chunks: Sequence[BookChunk]) -> dict[str, object]:
        by_section = {chunk.section_index: chunk for chunk in chunks}
        early, later = by_section[0], by_section[1]
        return {
            "entities": [
                {
                    "ref": "mara",
                    "name": "Mara",
                    "kind": "character",
                    "identity_status": "resolved",
                    "assertedness": "explicit",
                    "evidence": [
                        {"source_chunk_id": early.id, "source_unit_ids": ["u0"]}
                    ],
                },
                {
                    "ref": "jonas",
                    "name": "Jonas",
                    "kind": "character",
                    "assertedness": "explicit",
                    "evidence": [
                        {"source_chunk_id": later.id, "source_unit_ids": ["u1"]}
                    ],
                },
            ],
            "relationships": [
                {
                    "subject_ref": "mara",
                    "object_ref": "jonas",
                    "relationship_type": "trusts",
                    "assertedness": "inferred",
                    "evidence": [
                        {"source_chunk_id": later.id, "source_unit_ids": ["u1"]}
                    ],
                }
            ],
        }


def make_service() -> InMemoryRagService:
    service = InMemoryRagService()
    service.ingest_book(
        "book",
        "v1",
        [
            unit("u0", 0, "Mara found a silver key."),
            unit("u1", 1, "Jonas waits outside the station."),
            unit("u2", 2, "Mara learns the secret identity."),
        ],
    )
    return service


def safe_boundary(section: int = 1, source_id: str = "u1") -> ReadingBoundary:
    return ReadingBoundary("book", section, 0, source_id)


def _context_with_long_ids(context, count: int, prefix: str = "book-content-v1"):
    base = context.passages[0]
    passages = []
    ids = [f"{prefix}-{'source-unit-' * 14}-chunk-{index:03d}" for index in range(count)]
    for index, canonical_id in enumerate(ids):
        chunk = replace(base.chunk, id=canonical_id, text=f"Evidence passage {index + 1}.")
        passages.append(replace(base, passage_id=canonical_id, chunk=chunk))
    prompt = json.dumps({
        "passages": [
            {
                "passage_id": passage.passage_id,
                "source": {"section_index": passage.chunk.section_index,
                           "chunk_index": passage.chunk.chunk_index,
                           "href": passage.chunk.href},
                "text": passage.chunk.text,
            }
            for passage in passages
        ],
        "graph_facts": ([{"fact_id": "fact-1", "kind": "relationship",
                          "label": "supports", "assertedness": "explicit",
                          "evidence_passage_ids": [ids[-1]]}] if ids else []),
    }, ensure_ascii=False, separators=(",", ":"))
    return replace(context, passages=tuple(passages), prompt_json=prompt), ids


def _groq_response_transport(answer: dict[str, object]):
    envelope = {"choices": [{"message": {"content": json.dumps(answer)}}]}
    return lambda *_args: json.dumps(envelope).encode()


def keyword_results(service: InMemoryRagService, boundary: ReadingBoundary | None = None):
    return service.search_book(
        book_id="book",
        query="Mara",
        retrieval_mode="keyword",
        reading_boundary=boundary or safe_boundary(),
    )


class ContextAssemblyTests(unittest.TestCase):
    def test_context_is_deterministic_deduplicated_and_source_mapped(self) -> None:
        service = make_service()
        results = keyword_results(service)
        stored = service.store.get_book("book")[1]
        first = assemble_context(
            results=results + results,
            chunks_by_book={"book": stored},
            graphs={},
            boundaries={"book": safe_boundary()},
        )
        second = assemble_context(
            results=results,
            chunks_by_book={"book": stored},
            graphs={},
            boundaries={"book": safe_boundary()},
        )
        self.assertEqual(first.prompt_json, second.prompt_json)
        self.assertEqual(len(first.passages), 1)
        chunk = first.passages[0].chunk
        self.assertEqual(chunk.book_id, "book")
        self.assertEqual(chunk.content_version_id, "v1")
        self.assertEqual(chunk.source_positions[0].href, "chapter-0.xhtml")
        self.assertTrue(chunk.start_cfi.startswith("epubcfi("))

    def test_context_budget_skips_oversized_whole_passage(self) -> None:
        service = make_service()
        results = keyword_results(service)
        context = assemble_context(
            results=results,
            chunks_by_book={"book": service.store.get_book("book")[1]},
            graphs={},
            boundaries={"book": safe_boundary()},
            options=ContextOptions(max_chars=10, max_passages=4, max_graph_facts=2),
        )
        self.assertFalse(context.sufficient)
        self.assertEqual(context.passages, ())

    def test_invalid_or_missing_boundary_fails_closed(self) -> None:
        service = make_service()
        results = keyword_results(service)
        stored = service.store.get_book("book")[1]
        with self.assertRaises(ValueError):
            assemble_context(
                results=results,
                chunks_by_book={"book": stored},
                graphs={},
                boundaries={},
            )
        with self.assertRaises(ValueError):
            assemble_context(
                results=results,
                chunks_by_book={"book": stored},
                graphs={},
                boundaries={"book": ReadingBoundary("book", 99, 0, "missing")},
            )

    def test_boundary_crossing_canonical_chunk_is_not_assembled(self) -> None:
        service = InMemoryRagService()
        chunks = service.ingest_book(
            "book",
            "v1",
            [
                unit("u0", 0, "Mara reaches the door."),
                SourceUnitInput(
                    source_unit_id="u1",
                    section_index=0,
                    unit_index=1,
                    kind="paragraph",
                    text="A later passage reveals the answer.",
                ),
            ],
        )
        crossing = chunks[0]
        self.assertEqual(crossing.source_unit_ids, ("u0", "u1"))
        result = RetrievalResult(crossing, 1.0, "hybrid")
        context = assemble_context(
            results=[result],
            chunks_by_book={"book": chunks},
            graphs={},
            boundaries={"book": ReadingBoundary("book", 0, 0, "u0")},
        )
        self.assertFalse(context.sufficient)
        self.assertEqual(context.passages, ())

    def test_ineligible_graph_fact_is_omitted_from_context(self) -> None:
        service = InMemoryRagService(story_extractor=FakeExtractor())
        service.ingest_book(
            "book",
            "v1",
            [unit("u0", 0, "Mara waits."), unit("u1", 1, "Jonas appears.")],
        )
        service.extract_story_graph(book_id="book", reading_boundary=safe_boundary())
        results = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=safe_boundary(),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )
        context = assemble_context(
            results=results,
            chunks_by_book={"book": service.store.get_book("book")[1]},
            graphs={"book": service.story_graph.get_book("book")},
            boundaries={"book": ReadingBoundary("book", 0, 0, "u0")},
        )
        self.assertFalse(
            any(fact.fact_kind == "relationship" for fact in context.graph_facts)
        )
        self.assertTrue(all(fact.source_unit_ids == ("u0",) for fact in context.graph_facts))


class GenerationValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        service = make_service()
        self.context = assemble_context(
            results=keyword_results(service),
            chunks_by_book={"book": service.store.get_book("book")[1]},
            graphs={},
            boundaries={"book": safe_boundary()},
        )

    def test_valid_citation_resolves_to_selected_passage(self) -> None:
        provider = FakeProvider()
        answer = generate_grounded_answer(
            provider=provider, question="What did Mara find?", context=self.context
        )
        self.assertEqual(answer.status, "answered")
        self.assertIsNotNone(answer.uncertainty)
        self.assertEqual(
            answer.citations[0].passage.passage_id,
            self.context.passages[0].passage_id,
        )
        self.assertEqual(provider.calls[0][0], "What did Mara find?")

    def test_unknown_citation_fails_closed_and_insufficient_answer_abstains(self) -> None:
        unknown = FakeProvider(
            response=json.dumps(
                {
                    "answer": "Unsupported claim.",
                    "citations": ["made-up-id"],
                    "insufficient_evidence": False,
                    "uncertainty": None,
                }
            )
        )
        answer = generate_grounded_answer(provider=unknown, question="Q", context=self.context)
        self.assertEqual(answer.status, "abstained")
        self.assertEqual(answer.citations, ())

        insufficient = FakeProvider(
            response=json.dumps(
                {
                    "answer": "A speculative response.",
                    "citations": [],
                    "insufficient_evidence": True,
                    "uncertainty": "The source does not say.",
                }
            )
        )
        answer = generate_grounded_answer(
            provider=insufficient, question="Q", context=self.context
        )
        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(
            answer.answer,
            "The available passages do not establish a reliable answer.",
        )

    def test_no_evidence_does_not_call_provider(self) -> None:
        provider = FakeProvider()
        empty = assemble_context(
            results=[], chunks_by_book={}, graphs={}, boundaries={}
        )
        answer = generate_grounded_answer(provider=provider, question="Q", context=empty)
        self.assertEqual(answer.status, "insufficient_evidence")
        self.assertEqual(provider.calls, [])

    def test_malformed_generated_contract_is_rejected(self) -> None:
        provider = FakeProvider(response="not json")
        with self.assertRaises(GenerationMalformedResponseError):
            generate_grounded_answer(provider=provider, question="Q", context=self.context)

    def test_groq_short_aliases_resolve_long_canonical_ids_and_graph_evidence(self) -> None:
        context, canonical_ids = _context_with_long_ids(self.context, 2)
        captured = {}
        response_object = {
            "answer": "Tess accompanies the narrator.",
            "citations": ["P01", "P02"],
            "insufficient_evidence": False,
            "uncertainty": None,
        }

        def transport(request, _timeout):
            body = json.loads(request.data)
            captured["payload"] = body
            response = {"choices": [{"message": {"content": json.dumps(response_object)}}]}
            return json.dumps(response).encode()

        provider = GroqAnswerProvider(
            api_key="test-key", model="openai/gpt-oss-120b", transport=transport
        )
        answer = generate_grounded_answer(
            provider=provider, question="who is tess?", context=context
        )
        self.assertEqual(answer.status, "answered")
        self.assertEqual(
            [item.passage.passage_id for item in answer.citations], canonical_ids
        )
        user_message = json.loads(captured["payload"]["messages"][1]["content"])
        sent_ids = [item["passage_id"] for item in user_message["evidence"]["passages"]]
        self.assertEqual(sent_ids, ["P01", "P02"])
        self.assertFalse(any(item in json.dumps(user_message) for item in canonical_ids))
        self.assertEqual(
            user_message["evidence"]["graph_facts"][0]["evidence_passage_ids"],
            ["P02"],
        )

    def test_groq_unknown_modified_duplicate_and_request_local_aliases_fail_closed(self) -> None:
        context, canonical_ids = _context_with_long_ids(self.context, 2)
        for aliases in (["P0"], ["P01", "P01"], ["P03"]):
            with self.subTest(aliases=aliases):
                provider = GroqAnswerProvider(
                    api_key="key", model="openai/gpt-oss-120b",
                    transport=_groq_response_transport({
                        "answer": "Tess is younger.", "citations": aliases,
                        "insufficient_evidence": False, "uncertainty": None,
                    }),
                )
                answer = generate_grounded_answer(
                    provider=provider, question="Q", context=context
                )
                self.assertEqual(answer.status, "abstained")
                self.assertEqual(answer.citations, ())

        second_context, second_ids = _context_with_long_ids(self.context, 1, prefix="other-book")
        provider = GroqAnswerProvider(
            api_key="key", model="openai/gpt-oss-120b",
            transport=_groq_response_transport({
                "answer": "Supported.", "citations": ["P01"],
                "insufficient_evidence": False, "uncertainty": None,
            }),
        )
        answer = generate_grounded_answer(
            provider=provider, question="Q", context=second_context
        )
        self.assertEqual(answer.citations[0].passage.passage_id, second_ids[0])
        self.assertNotEqual(canonical_ids[0], second_ids[0])

    def test_insufficient_groq_answer_with_no_citations_is_valid(self) -> None:
        provider = GroqAnswerProvider(
            api_key="key", model="openai/gpt-oss-120b",
            transport=_groq_response_transport({
                "answer": "", "citations": [], "insufficient_evidence": True,
                "uncertainty": "The supplied passages do not establish this.",
            }),
        )
        answer = generate_grounded_answer(
            provider=provider, question="Q", context=self.context
        )
        self.assertEqual(answer.status, "insufficient_evidence")

    def test_aliases_expand_past_two_digits_without_colliding(self) -> None:
        context, canonical_ids = _context_with_long_ids(self.context, 101)
        captured = {}

        def transport(request, _timeout):
            captured["request"] = json.loads(request.data)
            return _groq_response_transport({
                "answer": "Supported.", "citations": ["P100"],
                "insufficient_evidence": False, "uncertainty": None,
            })(request, _timeout)

        answer = generate_grounded_answer(
            provider=GroqAnswerProvider(
                api_key="key", model="openai/gpt-oss-120b", transport=transport
            ),
            question="Q", context=context,
        )
        sent = json.loads(captured["request"]["messages"][1]["content"])
        sent_ids = [item["passage_id"] for item in sent["evidence"]["passages"]]
        self.assertEqual(len(sent_ids), 101)
        self.assertEqual(len(set(sent_ids)), 101)
        self.assertEqual(sent_ids[99], "P100")
        self.assertEqual(answer.citations[0].passage.passage_id, canonical_ids[99])


class GroqProviderTests(unittest.TestCase):
    def test_missing_configuration_fails_before_transport(self) -> None:
        calls = []
        provider = GroqAnswerProvider(
            api_key="", model="", transport=lambda *_: calls.append(1)
        )
        with self.assertRaises(GenerationConfigurationError):
            provider.generate(question="Q", context_json='{"passages":[]}')
        self.assertEqual(calls, [])

    def test_transport_receives_only_compact_selected_context(self) -> None:
        seen = {}

        def transport(request, timeout):
            seen["url"] = request.full_url
            seen["headers"] = request.headers
            seen["body"] = json.loads(request.data)
            message = json.loads(seen["body"]["messages"][1]["content"])
            passage_id = message["evidence"]["passages"][0]["passage_id"]
            return json.dumps(
                {"choices": [{"message": {"content": json.dumps({
                    "answer": "Evidence answer.", "citations": [passage_id],
                    "insufficient_evidence": False, "uncertainty": None,
                })}}]}
            ).encode()

        provider = GroqAnswerProvider(
            api_key="secret-test-key", model="test-model", transport=transport
        )
        content = provider.generate(
            question="Question?",
            context_json=json.dumps(
                {
                    "passages": [{"passage_id": "p1", "text": "Selected passage"}],
                    "graph_facts": [],
                }
            ),
        )
        self.assertEqual(seen["url"], GroqAnswerProvider.endpoint)
        self.assertNotIn("secret-test-key", str(seen["body"]))
        self.assertEqual(json.loads(content)["citations"], ["p1"])
        response_format = seen["body"]["response_format"]
        self.assertEqual(response_format["type"], "json_schema")
        self.assertTrue(response_format["json_schema"]["strict"])
        self.assertEqual(
            response_format["json_schema"]["schema"]["required"],
            ["answer", "citations", "insufficient_evidence", "uncertainty"],
        )
        self.assertFalse(response_format["json_schema"]["schema"]["additionalProperties"])
        self.assertEqual(seen["body"]["reasoning_effort"], "low")
        self.assertEqual(seen["body"]["max_completion_tokens"], 768)
        system = seen["body"]["messages"][0]["content"]
        self.assertIn("reading companion", system)
        self.assertIn("only factual evidence", system)
        self.assertIn("partial", system)
        self.assertIn("interpretations", system)
        self.assertIn("passage_id", system)
        self.assertIn("one concise sentence", system)
        self.assertEqual(seen["headers"].get("User-agent"), "Lector/1.0")

    def test_provider_maps_timeout_auth_rate_limit_and_malformed_output(self) -> None:
        def raising(error):
            return lambda request, timeout: (_ for _ in ()).throw(error)

        cases = [
            (TimeoutError(), GenerationTimeoutError),
            (
                urllib.error.HTTPError("url", 401, "unauthorized", {}, None),
                GenerationAuthenticationError,
            ),
            (
                urllib.error.HTTPError("url", 429, "limited", {}, None),
                GenerationRateLimitError,
            ),
        ]
        for transport_error, expected in cases:
            with self.subTest(expected=expected.__name__):
                provider = GroqAnswerProvider(
                    api_key="key", model="model", transport=raising(transport_error)
                )
                with self.assertRaises(expected):
                    provider.generate(question="Q", context_json='{"passages":[]}')

        provider = GroqAnswerProvider(
            api_key="key", model="model", transport=lambda *_: b"not json"
        )
        with self.assertRaises(GenerationMalformedResponseError):
            provider.generate(question="Q", context_json='{"passages":[]}')

    def test_http_400_json_failure_and_completion_exhaustion_are_structured_output_errors(self):
        error_bodies = (
            b'{"error":{"code":"json_validate_failed","message":"Failed to validate JSON"}}',
            b'{"error":{"code":"json_validate_failed","failed_generation":"max completion tokens reached before generating a valid document"}}',
        )
        for body in error_bodies:
            json_error = urllib.error.HTTPError(
                "url", 400, "bad request", {}, __import__("io").BytesIO(body)
            )
            provider = GroqAnswerProvider(
                api_key="key", model="openai/gpt-oss-120b",
                transport=lambda *_args, error=json_error: (_ for _ in ()).throw(error),
            )
            with self.assertRaises(GenerationStructuredOutputError) as raised:
                provider.generate(question="Q", context_json='{"passages":[]}')
            self.assertIn("Lector", str(raised.exception))

        rate_error = urllib.error.HTTPError("url", 429, "limited", {"Retry-After": "7"}, None)
        provider = GroqAnswerProvider(api_key="key", model="model", transport=lambda *_: (_ for _ in ()).throw(rate_error))
        with self.assertRaises(GenerationRateLimitError) as raised:
            provider.generate(question="Q", context_json='{"passages":[]}')
        self.assertIn("7 seconds", str(raised.exception))

    def test_simple_factual_answer_and_insufficient_evidence_keep_contract(self):
        for response, expected_status in (
            ({"answer": "Tess accompanies the narrator and is two years younger.",
              "citations": ["p1"], "insufficient_evidence": False, "uncertainty": None}, "answered"),
            ({"answer": "", "citations": [], "insufficient_evidence": True,
              "uncertainty": "The supplied passages do not identify Tess."}, "insufficient_evidence"),
        ):
            with self.subTest(status=expected_status):
                result = json.dumps(response)
                provider = FakeProvider(response=result)
                answer = generate_grounded_answer(
                    provider=provider, question="who is tess?", context=self._context_with_p1()
                )
                self.assertEqual(answer.status, expected_status)

    def _context_with_p1(self):
        from lector_rag_api.chunking import BookChunk
        from lector_rag_api.retrieval_types import RetrievalResult

        chunk = BookChunk(
            id="p1", book_id="book", content_version_id="v1", section_index=0,
            chunk_index=0, text="Tess accompanies the narrator and is two years younger.",
            source_unit_ids=("u1",),
            source_positions=(ChunkSourcePosition("u1", 0, "paragraph", None, None),),
            start_unit_index=0,
            end_unit_index=0, href=None, heading=None, start_cfi=None, end_cfi=None,
            cfi_status="unavailable",
        )
        boundary = ReadingBoundary("book", 0, 0, "u1")
        return assemble_context(
            results=[RetrievalResult(chunk=chunk, score=1, retrieval_method="keyword")],
            chunks_by_book={"book": [chunk]}, graphs={}, boundaries={"book": boundary},
        )


class AnswerApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._qa_temp = tempfile.TemporaryDirectory()
        self._old_qa_log_dir = os.environ.get("LECTOR_QA_LOG_DIR")
        os.environ["LECTOR_QA_LOG_DIR"] = self._qa_temp.name
        self.service = make_service()
        self.provider = FakeProvider()
        self.client = TestClient(create_app(self.service, self.provider))

    def tearDown(self) -> None:
        self.client.close()
        if self._old_qa_log_dir is None:
            os.environ.pop("LECTOR_QA_LOG_DIR", None)
        else:
            os.environ["LECTOR_QA_LOG_DIR"] = self._old_qa_log_dir
        self._qa_temp.cleanup()

    def payload(self, **changes):
        payload = {
            "query": "What did Mara find?",
            "book_id": "book",
            "reading_boundary": {
                "confirmed": True,
                "section_index": 1,
                "unit_index": 0,
                "source_unit_id": "u1",
            },
            "retrieval_mode": "keyword",
        }
        payload.update(changes)
        return payload

    def test_answer_endpoint_returns_validated_citation_metadata(self) -> None:
        response = self.client.post("/api/v1/answer", json=self.payload())
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "answered")
        self.assertEqual(body["retrieval_mode"], "keyword")
        self.assertEqual(body["citations"][0]["book_id"], "book")
        self.assertEqual(body["citations"][0]["content_version_id"], "v1")
        self.assertTrue(body["citations"][0]["start_cfi"].startswith("epubcfi("))
        sent = json.loads(self.provider.calls[0][1])
        self.assertEqual(len(sent["passages"]), 1)
        self.assertNotIn("secret identity", json.dumps(sent))

    def test_structured_output_provider_failure_has_distinct_api_error(self) -> None:
        provider = FakeProvider(error=GenerationStructuredOutputError(
            "Lector could not complete a structured answer from the selected evidence."
        ))
        client = TestClient(create_app(self.service, provider))
        try:
            response = client.post("/api/v1/answer", json=self.payload())
            self.assertEqual(response.status_code, 502)
            self.assertEqual(
                response.json()["error"]["code"],
                "generation_structured_output_failed",
            )
            self.assertIn("Lector", response.json()["error"]["message"])
        finally:
            client.close()

    def test_later_chapter_evidence_requires_explicit_boundary_advance(self) -> None:
        early = self.client.post(
            "/api/v1/answer",
            json=self.payload(
                query="What is the secret identity?",
                reading_boundary={
                    "confirmed": True,
                    "section_index": 1,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
            ),
        )
        self.assertEqual(early.status_code, 200)
        self.assertEqual(early.json()["status"], "answered")
        self.assertEqual(len(self.provider.calls), 1)
        self.assertNotIn("secret identity", self.provider.calls[0][1])

        confirmed_later = self.client.post(
            "/api/v1/answer",
            json=self.payload(
                query="What is the secret identity?",
                reading_boundary={
                    "confirmed": True,
                    "section_index": 2,
                    "unit_index": 0,
                    "source_unit_id": "u2",
                },
            ),
        )
        self.assertEqual(confirmed_later.status_code, 200)
        self.assertEqual(confirmed_later.json()["status"], "answered")
        self.assertIn("secret identity", self.provider.calls[1][1])

    def test_story_so_far_uses_ordered_boundary_eligible_summary_evidence(self) -> None:
        response = self.client.post(
            "/api/v1/answer", json=self.payload(query="Tell me the story so far")
        )
        self.assertEqual(response.status_code, 200)
        evidence = json.loads(self.provider.calls[0][1])["passages"]
        self.assertEqual(
            [item["text"] for item in evidence],
            ["Mara found a silver key.", "Jonas waits outside the station."],
        )
        self.assertNotIn("secret identity", json.dumps(evidence))

    def test_missing_invalid_boundaries_and_content_versions_fail_closed(self) -> None:
        missing = self.payload()
        missing.pop("reading_boundary")
        self.assertEqual(self.client.post("/api/v1/answer", json=missing).status_code, 422)

        mismatch = self.payload(content_version_id="old-version")
        response = self.client.post("/api/v1/answer", json=mismatch)
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "content_version_mismatch")

    def test_insufficient_evidence_skips_generation(self) -> None:
        response = self.client.post(
            "/api/v1/answer", json=self.payload(query="An absent phrase")
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "insufficient_evidence")
        self.assertEqual(self.provider.calls, [])

    def test_missing_groq_configuration_is_structured_and_offline(self) -> None:
        client = TestClient(
            create_app(
                self.service,
                GroqAnswerProvider(
                    api_key="",
                    model="",
                    transport=lambda *_: self.fail("network called"),
                ),
            )
        )
        response = client.post("/api/v1/answer", json=self.payload())
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["error"]["code"], "generation_not_configured")

    def test_all_books_requires_explicit_boundary_for_every_indexed_book(self) -> None:
        self.service.ingest_book("other", "v1", [unit("other-unit", 0, "Other story.")])
        response = self.client.post(
            "/api/v1/answer",
            json={
                "query": "Mara",
                "all_books": True,
                "reading_boundaries": [
                    {
                        "book_id": "book",
                        "confirmed": True,
                        "section_index": 1,
                        "unit_index": 0,
                        "source_unit_id": "u1",
                    }
                ],
                "retrieval_mode": "keyword",
            },
        )
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "invalid_reading_boundary")

    def test_reingestion_during_generation_invalidates_the_response(self) -> None:
        service = self.service

        class ReplacingProvider:
            def generate(self, *, question: str, context_json: str) -> str:
                service.ingest_book("book", "v2", [unit("new-unit", 0, "Replacement text.")])
                return json.dumps(
                    {
                        "answer": "Mara found a key.",
                        "citations": [json.loads(context_json)["passages"][0]["passage_id"]],
                        "insufficient_evidence": False,
                        "uncertainty": None,
                    }
                )

        client = TestClient(create_app(service, ReplacingProvider()))
        response = client.post("/api/v1/answer", json=self.payload())
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["error"]["code"], "answer_evidence_changed")

    def test_provider_failures_use_sanitized_structured_http_errors(self) -> None:
        cases = [
            (GenerationAuthenticationError("credentials rejected"), 502),
            (GenerationRateLimitError("limited"), 429),
            (GenerationTimeoutError("timed out"), 504),
            (GenerationMalformedResponseError("invalid JSON"), 502),
            (GenerationUnavailableError("unavailable"), 502),
        ]
        for error, status in cases:
            with self.subTest(error=type(error).__name__):
                client = TestClient(create_app(self.service, FakeProvider(error=error)))
                response = client.post("/api/v1/answer", json=self.payload())
                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], error.code)


if __name__ == "__main__":
    unittest.main()
