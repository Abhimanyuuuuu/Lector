"""BookNLP graph views and retrieval labels obey confirmed boundaries."""

import re
import unittest
from copy import deepcopy
from threading import Event

from lector_rag_api.booknlp_graph import graph_from_artifacts, bundle_to_json
from fastapi.testclient import TestClient
from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService


def token_rows(text: str) -> list[dict[str, object]]:
    rows = []
    sentence = 0
    for token_id, match in enumerate(re.finditer(r"[A-Za-z']+|[.!?]", text)):
        word = match.group()
        lemma = {"met": "meet", "helped": "help"}.get(word.casefold(), word.casefold())
        rows.append({
            "paragraph_ID": str(sentence),
            "sentence_ID": str(sentence),
            "token_ID_within_document": str(token_id),
            "word": word,
            "lemma": lemma,
            "byte_onset": str(len(text[:match.start()].encode("utf-8"))),
            "byte_offset": str(len(text[:match.end()].encode("utf-8"))),
            "POS_tag": "VERB" if word.casefold() in {"met", "helped"} else "NOUN",
        })
        if word in {".", "!", "?"}:
            sentence += 1
    return rows


class BookNLPBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.service = InMemoryRagService()
        self.addCleanup(self.service.close)
        self.units = [
            SourceUnitInput(
                source_unit_id="early", section_index=0, unit_index=0,
                kind="paragraph", text="the girl met John.",
            ),
            SourceUnitInput(
                source_unit_id="late", section_index=1, unit_index=0,
                kind="paragraph", text="Tess helped John. Mara watched.",
            ),
        ]
        chunks = self.service.ingest_book("book", "v1", self.units)
        self.service.store.schedule_book_preparation(
            book_id="book", content_version_id="v1",
            source_units=[], graph_rag_enabled=True,
        )
        self.service.store.schedule_booknlp_job(
            book_id="book", content_version_id="v1", total_chunks=len(chunks),
            extraction_version="fixture-booknlp",
        )
        full_text = "\n\n".join(chunk.text for chunk in chunks)
        bundle = graph_from_artifacts(
            book_id="book", content_version_id="v1", chunks=chunks,
            book_data={"characters": [
                {"id": "1", "count": 2, "mentions": {
                    "proper": [{"n": "Tess", "c": 1}],
                    "common": [{"n": "the girl", "c": 1}],
                }},
                {"id": "2", "count": 2, "mentions": {
                    "proper": [{"n": "John", "c": 2}],
                }},
                {"id": "3", "count": 1, "mentions": {
                    "proper": [{"n": "Mara", "c": 1}],
                }},
            ]},
            entities=[
                {"COREF": "1", "start_token": 0, "end_token": 1, "prop": "NOM", "text": "the girl"},
                {"COREF": "2", "start_token": 3, "end_token": 3, "prop": "PROP", "text": "John"},
                {"COREF": "1", "start_token": 5, "end_token": 5, "prop": "PROP", "text": "Tess"},
                {"COREF": "2", "start_token": 7, "end_token": 7, "prop": "PROP", "text": "John"},
                {"COREF": "3", "start_token": 9, "end_token": 9, "prop": "PROP", "text": "Mara"},
            ],
            tokens=token_rows(full_text),
        )
        payload = bundle_to_json(bundle)
        self.service.store.persist_booknlp_bundle(
            book_id="book", content_version_id="v1", bundle=payload,
            total_chunks=len(chunks),
        )
        for index, chunk in enumerate(chunks):
            self.service.store.commit_booknlp_batch(
                book_id="book", content_version_id="v1", bundle=payload,
                chunk_ids=[chunk.id], completed_chunks=index + 1,
                total_chunks=len(chunks), complete=index + 1 == len(chunks),
            )

    def test_graph_view_hides_future_relationship_evidence_and_identity(self):
        early = self.service.eligible_character_graph(
            book_id="book",
            reading_boundary=ReadingBoundary("book", 0, 0, "early"),
        )
        self.assertEqual(len(early["edges"]), 1)
        self.assertEqual(early["edges"][0]["evidence_count"], 1)
        early_names = {node["name"] for node in early["nodes"]}
        self.assertIn("the girl", early_names)
        self.assertNotIn("Tess", early_names)
        self.assertNotIn("Mara", early_names)
        self.assertEqual(
            {
                item["source_chunk_id"]
                for edge in early["edges"]
                for item in edge["evidence"]
            },
            {self.service.store.get_book("book")[1][0].id},
        )
        evidence = early["edges"][0]["evidence"][0]
        self.assertEqual(evidence["source_unit_id"], "early")
        self.assertEqual(evidence["source_unit_ids"], ["early"])
        self.assertEqual(
            [position["source_unit_id"] for position in evidence["source_positions"]],
            ["early"],
        )
        self.assertIsInstance(evidence["start_character"], int)
        self.assertIsInstance(evidence["end_character"], int)

        later = self.service.eligible_character_graph(
            book_id="book",
            reading_boundary=ReadingBoundary("book", 1, 0, "late"),
        )
        self.assertEqual(later["edges"][0]["evidence_count"], 2)
        self.assertIn("Tess", {node["name"] for node in later["nodes"]})

    def test_graph_retrieval_entities_use_only_eligible_names_and_aliases(self):
        graph = self.service._booknlp_story_graph(
            "book", "v1", ReadingBoundary("book", 0, 0, "early")
        )
        assert graph is not None
        entities, _, _ = graph.eligible_facts(ReadingBoundary("book", 0, 0, "early"))
        girl = next(entity for entity in entities if entity.name == "the girl")
        self.assertEqual(girl.alias_names, ("the girl",))
        self.assertNotIn("Tess", girl.alias_names)

    def test_graph_view_api_returns_boundary_filtered_payload(self):
        client = TestClient(create_app(self.service))
        response = client.post("/api/v1/books/book/graph/view", json={
            "content_version_id": "v1",
            "view_mode": "till_now",
            "confirmed": True,
            "section_index": 0,
            "unit_index": 0,
            "source_unit_id": "early",
        })
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["content_version_id"], "v1")
        self.assertEqual(payload["view_mode"], "till_now")
        self.assertEqual(payload["edges"][0]["evidence_count"], 1)
        self.assertNotIn("Tess", {node["name"] for node in payload["nodes"]})
        self.assertTrue(all("profile_summary" not in node for node in payload["nodes"]))

    def test_full_book_graph_api_does_not_require_or_return_a_reading_boundary(self):
        client = TestClient(create_app(self.service))
        response = client.post("/api/v1/books/book/graph/view", json={
            "content_version_id": "v1",
            "view_mode": "full_book",
        })
        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["view_mode"], "full_book")
        self.assertIsNone(payload["boundary"])
        self.assertIn("Tess", {node["name"] for node in payload["nodes"]})
        self.assertEqual(payload["edges"][0]["evidence_count"], 2)
        self.assertIn("Mara", {node["name"] for node in payload["nodes"]})
        self.assertTrue(all("profile_summary" in node for node in payload["nodes"]))

    def test_till_now_graph_api_rejects_missing_boundary(self):
        response = TestClient(create_app(self.service)).post(
            "/api/v1/books/book/graph/view",
            json={"content_version_id": "v1", "view_mode": "till_now"},
        )
        self.assertEqual(response.status_code, 422)

    def test_staged_graph_rebuild_keeps_old_graph_after_validation_failure_and_replaces_idempotently(self):
        original = self.service.store.load_booknlp_graph("book", "v1")
        assert original is not None
        self.service.store.schedule_booknlp_rebuild(
            book_id="book", content_version_id="v1",
            total_chunks=len(self.service.store.get_book("book")[1]),
            extraction_version="fixture-booknlp-v2",
        )
        invalid = deepcopy(original)
        invalid["content_version_id"] = "v1"
        if invalid["edges"]:
            invalid["edges"][0]["evidence"][0]["source_chunk_id"] = "missing-chunk"
        with self.assertRaisesRegex(Exception, "unknown canonical chunk"):
            self.service.store.commit_booknlp_rebuild(
                book_id="book", content_version_id="v1", bundle=invalid,
                total_chunks=len(self.service.store.get_book("book")[1]),
            )
        self.assertEqual(self.service.store.load_booknlp_graph("book", "v1"), original)

        self.service.store._connection.execute(
            "CREATE TRIGGER reject_graph_rebuild BEFORE INSERT ON booknlp_edges "
            "BEGIN SELECT RAISE(ABORT, 'injected graph write failure'); END"
        )
        with self.assertRaisesRegex(Exception, "injected graph write failure"):
            self.service.store.commit_booknlp_rebuild(
                book_id="book", content_version_id="v1", bundle=original,
                total_chunks=len(self.service.store.get_book("book")[1]),
            )
        self.service.store._connection.execute("DROP TRIGGER reject_graph_rebuild")
        self.assertEqual(self.service.store.load_booknlp_graph("book", "v1"), original)

        for _ in range(2):
            self.service.store.commit_booknlp_rebuild(
                book_id="book", content_version_id="v1", bundle=original,
                total_chunks=len(self.service.store.get_book("book")[1]),
            )
        replaced = self.service.store.load_booknlp_graph("book", "v1")
        self.assertEqual(replaced, original)
        self.assertEqual(len(replaced["edges"]), len(original["edges"]))

    def test_graph_rebuild_api_schedules_only_explicitly(self):
        client = TestClient(create_app(self.service))
        response = client.post(
            "/api/v1/books/book/graph/rebuild?content_version_id=v1"
        )
        # This fixture service intentionally has no real BookNLP runner.
        self.assertEqual(response.status_code, 503)

    def test_graph_rebuild_api_returns_while_local_analysis_runs(self):
        class WaitingRunner:
            name = "fake-booknlp"
            version = "blocking-fixture"

            def __init__(self):
                self.started = Event()
                self.release = Event()

            def process(self, _text, _book_id):
                self.started.set()
                self.release.wait(2)
                raise RuntimeError("fixture stops before writing a replacement")

        runner = WaitingRunner()
        self.service.booknlp_runner = runner
        self.addCleanup(runner.release.set)
        client = TestClient(create_app(self.service))
        response = client.post(
            "/api/v1/books/book/graph/rebuild?content_version_id=v1"
        )
        self.assertEqual(response.status_code, 202, response.text)
        self.assertTrue(runner.started.wait(1))
        status = client.get(
            "/api/v1/books/book/graph/rebuild/status?content_version_id=v1"
        )
        self.assertEqual(status.status_code, 200)
        self.assertIn(status.json()["status"], {"queued", "processing"})

    def test_empty_rebuild_result_preserves_existing_graph_and_reports_failure(self):
        class EmptyRunner:
            name = "fake-booknlp"
            version = "empty-fixture"

            def process(self, _text, _book_id):
                return {"book": {"characters": []}, "entities": [], "tokens": []}

        self.service.booknlp_runner = EmptyRunner()
        original = self.service.store.load_booknlp_graph("book", "v1")
        self.service.store.schedule_booknlp_rebuild(
            book_id="book", content_version_id="v1",
            total_chunks=len(self.service.store.get_book("book")[1]),
            extraction_version="empty-fixture",
        )
        self.service._run_booknlp_rebuild(
            {"book_id": "book", "content_version_id": "v1"}
        )
        self.assertEqual(self.service.store.load_booknlp_graph("book", "v1"), original)
        status = self.service.booknlp_rebuild_status(book_id="book", content_version_id="v1")
        self.assertEqual(status["status"], "failed")
        self.assertIn("no usable source-grounded", status["error"])


if __name__ == "__main__":
    unittest.main()
