from __future__ import annotations

import json
import os
import tempfile
import unittest
import urllib.error

from lector_rag_api.answering import (
    GenerationMalformedResponseError,
    GenerationStructuredOutputError,
    GroqAnswerProvider,
    generate_grounded_answer,
)
from lector_rag_api.context import assemble_context
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService
from lector_rag_api.qa_logging import QARequestLog


class QALoggingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("LECTOR_QA_LOG_DIR")
        os.environ["LECTOR_QA_LOG_DIR"] = self.temp.name

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("LECTOR_QA_LOG_DIR", None)
        else:
            os.environ["LECTOR_QA_LOG_DIR"] = self.previous
        self.temp.cleanup()

    def make_log(self):
        return QARequestLog.begin(query="Who is Zoë?", retrieval_mode="keyword",
            book_id="book-1", content_version_id="v1",
            boundaries={"section_index": 0, "unit_index": 2})

    def test_raw_success_response_is_written_before_json_parsing(self):
        log = self.make_log()
        raw = "not outer JSON: raw response preserved"
        provider = GroqAnswerProvider(api_key="secret-test-key", model="test", transport=lambda *_: raw.encode())
        with self.assertRaises(GenerationMalformedResponseError):
            provider.generate(question="Who is Zoë?", context_json='{"passages":[]} ', audit_log=log)
        text = log.path.read_text(encoding="utf-8")
        self.assertIn(raw, text)
        self.assertIn("EVIDENCE SENT TO GROQ:\n{\"passages\":[]} ", text)
        self.assertNotIn("secret-test-key", text)
        self.assertNotIn("Authorization", text)

    def test_http_errors_and_timeout_log_outcome_and_raw_body(self):
        for status in (400, 401, 403, 429):
            log = self.make_log()
            body = f"raw provider error {status}"

            def fail(*_args, code=status, payload=body):
                raise urllib.error.HTTPError("https://example.invalid", code, "failure", {}, __import__("io").BytesIO(payload.encode()))

            provider = GroqAnswerProvider(api_key="secret", model="test", transport=fail)
            with self.assertRaises(Exception):
                provider.generate(question="Q", context_json="{}", audit_log=log)
            text = log.path.read_text(encoding="utf-8")
            self.assertIn(body, text)
            self.assertIn(f"GROQ HTTP STATUS: {status}", text)
        log = self.make_log()
        provider = GroqAnswerProvider(api_key="secret", model="test", transport=lambda *_: (_ for _ in ()).throw(TimeoutError()))
        with self.assertRaises(Exception):
            provider.generate(question="Q", context_json="{}", audit_log=log)
        self.assertIn("REQUEST OUTCOME: timeout", log.path.read_text(encoding="utf-8"))

    def test_json_validation_400_raw_body_is_audited_before_error_mapping(self):
        log = self.make_log()
        raw = json.dumps({
            "error": {
                "code": "json_validate_failed",
                "failed_generation": "max completion tokens reached before generating a valid document",
            }
        })

        def fail(*_args):
            raise urllib.error.HTTPError(
                "https://example.invalid", 400, "bad request", {},
                __import__("io").BytesIO(raw.encode()),
            )

        provider = GroqAnswerProvider(api_key="secret", model="test", transport=fail)
        with self.assertRaises(GenerationStructuredOutputError):
            provider.generate(question="Q", context_json='{"passages":[]}', audit_log=log)
        text = log.path.read_text(encoding="utf-8")
        self.assertIn(raw, text)
        self.assertIn("GROQ HTTP STATUS: 400", text)
        self.assertNotIn("secret", text)

    def test_alias_mapping_and_validation_are_audited_separately_from_raw_response(self):
        log = self.make_log()
        service = InMemoryRagService()
        try:
            chunks = service.ingest_book("book-1", "v1", [SourceUnitInput(
                source_unit_id="u1", section_index=0, unit_index=0,
                kind="paragraph", text="Tess is two years younger than the narrator.",
            )])
            boundary = ReadingBoundary("book-1", 0, 0, "u1")
            results = service.search_book(
                book_id="book-1", query="Tess narrator", retrieval_mode="keyword",
                reading_boundary=boundary,
            )
            context = assemble_context(
                results=results, chunks_by_book={"book-1": chunks}, graphs={},
                boundaries={"book-1": boundary},
            )
            passage_id = context.passages[0].passage_id
            raw_completion = json.dumps({
                "answer": "Tess is two years younger.", "citations": ["P01"],
                "insufficient_evidence": False, "uncertainty": None,
            })
            raw_envelope = json.dumps({"choices": [{"message": {"content": raw_completion}}]})
            provider = GroqAnswerProvider(
                api_key="secret-test-key", model="openai/gpt-oss-120b",
                transport=lambda *_: raw_envelope.encode(),
            )
            answer = generate_grounded_answer(
                provider=provider, question="who is Tess?", context=context, audit_log=log
            )
            self.assertEqual(answer.citations[0].passage.passage_id, passage_id)
            log.finish(answer.status)
            text = log.path.read_text(encoding="utf-8")
            self.assertIn(raw_envelope, text)
            self.assertIn("CITATION ALIASES SENT TO MODEL:\nP01 -> " + passage_id, text)
            self.assertIn('RAW CITATION ALIASES:\n["P01"]', text)
            self.assertIn('"accepted":true', text)
            self.assertIn('"canonical_id":"' + passage_id + '"', text)
            self.assertIn("CITATION VALIDATION: accepted", text)
            self.assertNotIn("secret-test-key", text)
        finally:
            service.close()

    def test_invalid_citation_is_logged_as_postprocessing(self):
        # Reuse the existing deterministic context fixture without a network call.
        service = InMemoryRagService()
        chunks = service.ingest_book("book", "v1", [SourceUnitInput(
            source_unit_id="u1", section_index=0, unit_index=0,
            kind="paragraph", text="A supported passage.",
        )])
        boundary = ReadingBoundary("book", 0, 0, "u1")
        result = service.search_book(
            book_id="book", query="supported", retrieval_mode="keyword",
            reading_boundary=boundary,
        )
        context = assemble_context(
            results=result, chunks_by_book={"book": chunks}, graphs={},
            boundaries={"book": boundary},
        )
        try:
            log = self.make_log()
            raw_completion = json.dumps({
                "answer": "Unsupported claim", "citations": ["made-up"],
                "insufficient_evidence": False, "uncertainty": None,
            })
            raw_envelope = json.dumps({"choices": [{"message": {"content": raw_completion}}]})
            provider = GroqAnswerProvider(api_key="secret", model="test", transport=lambda *_: raw_envelope.encode())
            answer = generate_grounded_answer(provider=provider, question="Q", context=context, audit_log=log)
            self.assertEqual(answer.status, "abstained")
            log.finish(answer.status, "invalid citation IDs rejected")
            text = log.path.read_text(encoding="utf-8")
            self.assertIn(raw_envelope, text)
            self.assertIn("invalid citation IDs rejected", text)
            self.assertNotIn("Authorization", text)
        finally:
            service.close()


if __name__ == "__main__":
    unittest.main()
