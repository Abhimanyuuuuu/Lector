"""Bounded graph-assisted retrieval and API integration tests."""

import unittest
from typing import Sequence

from fastapi.testclient import TestClient

from lector_rag_api.chunking import BookChunk
from lector_rag_api.graph_retrieval import GraphRetrievalOptions
from lector_rag_api.hybrid import HybridConfig
from lector_rag_api.main import create_app
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService, InvalidReadingBoundaryError
from lector_rag_api.story_graph import StoryExtraction


def source(
    unit_id: str, section: int, text: str, unit_index: int = 0
) -> SourceUnitInput:
    return SourceUnitInput(
        source_unit_id=unit_id,
        section_index=section,
        unit_index=unit_index,
        kind="paragraph",
        text=text,
        href=f"chapter-{section}.xhtml",
        cfi=f"epubcfi(/6/{section + 2}!/4/2/1:0,/1:{len(text)})",
    )


class FixtureExtractor:
    name = "fixture"
    version = "1"

    def extract(self, chunks: Sequence[BookChunk]) -> dict[str, object]:
        by_section = {chunk.section_index: chunk for chunk in chunks}
        if len(by_section) == 1:
            # Boundary-specific extraction supplies only eligible material.
            chunk = next(iter(by_section.values()))
            return {
                "entities": [
                    {
                        "ref": "mara",
                        "name": "Mara",
                        "kind": "character",
                        "identity_status": "resolved",
                        "assertedness": "explicit",
                        "evidence": [
                            {
                                "source_chunk_id": chunk.id,
                                "source_unit_ids": list(chunk.source_unit_ids),
                            }
                        ],
                    }
                ]
            }
        first, second = by_section[0], by_section[1]
        ev0 = {"source_chunk_id": first.id, "source_unit_ids": list(first.source_unit_ids)}
        ev1 = {"source_chunk_id": second.id, "source_unit_ids": list(second.source_unit_ids)}
        return {
            "entities": [
                {
                    "ref": "mara",
                    "name": "Mara",
                    "kind": "character",
                    "identity_status": "resolved",
                    "assertedness": "explicit",
                    "evidence": [ev0],
                },
                {
                    "ref": "jonas",
                    "name": "Jonas",
                    "kind": "character",
                    "assertedness": "explicit",
                    "evidence": [ev1],
                },
            ],
            "events": [
                {
                    "ref": "meeting",
                    "name": "The harbor meeting",
                    "participant_refs": ["mara", "jonas"],
                    "assertedness": "explicit",
                    "evidence": [ev1],
                }
            ],
            "relationships": [
                {
                    "subject_ref": "mara",
                    "object_ref": "jonas",
                    "relationship_type": "trusts",
                    "assertedness": "inferred",
                    "evidence": [ev1],
                }
            ],
        }


class CrossingBoundaryExtractor:
    name = "crossing-fixture"
    version = "1"

    def extract(self, chunks: Sequence[BookChunk]) -> dict[str, object]:
        by_section = {chunk.section_index: chunk for chunk in chunks}
        early, crossing = by_section[0], by_section[1]
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
                        {"source_chunk_id": crossing.id, "source_unit_ids": ["u1"]}
                    ],
                },
            ],
            "relationships": [
                {
                    "subject_ref": "mara",
                    "object_ref": "jonas",
                    "relationship_type": "recognizes",
                    "assertedness": "explicit",
                    "evidence": [
                        {"source_chunk_id": crossing.id, "source_unit_ids": ["u1"]}
                    ],
                }
            ],
        }


def configured_service() -> InMemoryRagService:
    service = InMemoryRagService(story_extractor=FixtureExtractor())
    service.ingest_book(
        "book",
        "v1",
        [
            source("u0", 0, "Mara arrives at the station."),
            source("u1", 1, "Jonas waits near the harbor."),
        ],
    )
    service.extract_story_graph(
        book_id="book", reading_boundary=ReadingBoundary("book", 1, 0, "u1")
    )
    return service


class GraphRetrievalTests(unittest.TestCase):
    def test_character_seed_expands_to_event_and_relationship_evidence(self) -> None:
        service = configured_service()
        results = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 1, 0, "u1"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )

        self.assertEqual(results[0].chunk.section_index, 0)
        expanded = next(result for result in results if result.chunk.section_index == 1)
        self.assertEqual(expanded.retrieval_method, "graph")
        self.assertEqual(expanded.graph_depth, 1)
        self.assertTrue(
            {"event", "relationship"}.issubset(
                {reason.fact_kind for reason in expanded.graph_reasons}
            )
        )
        self.assertEqual(expanded.graph_reasons[0].evidence[0].source_chunk_id, expanded.chunk.id)
        self.assertTrue(any(reason.assertedness == "inferred" for reason in expanded.graph_reasons))
        self.assertEqual(expanded.chunk.source_unit_ids, ("u1",))
        repeated = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 1, 0, "u1"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )
        self.assertEqual([item.chunk.id for item in results], [item.chunk.id for item in repeated])
        self.assertEqual(len({item.chunk.id for item in results}), len(results))

    def test_graph_mode_respects_confirmed_boundary_before_traversal(self) -> None:
        service = configured_service()
        results = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 0, 0, "u0"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )

        self.assertTrue(results)
        self.assertTrue(all(result.chunk.section_index == 0 for result in results))
        self.assertFalse(any(result.graph_depth for result in results))

    def test_ineligible_relationship_cannot_be_used_as_intermediate_path(self) -> None:
        service = configured_service()
        graph = service.story_graph.get_book("book")
        # The relationship/event evidence in section 1 is withheld at section 0.
        entities, events, relationships = graph.eligible_facts(
            ReadingBoundary("book", 0, 0, "u0")
        )
        self.assertEqual(events, ())
        self.assertEqual(relationships, ())
        self.assertEqual(len(entities), 1)

    def test_fact_is_rejected_when_its_canonical_chunk_crosses_boundary(self) -> None:
        service = InMemoryRagService(story_extractor=CrossingBoundaryExtractor())
        chunks = service.ingest_book(
            "book",
            "v1",
            [
                source("u0", 0, "Mara waits."),
                source("u1", 1, "Jonas arrives."),
                source("u2", 1, "A later clue appears.", unit_index=1),
            ],
        )
        crossing = next(chunk for chunk in chunks if chunk.section_index == 1)
        self.assertEqual(crossing.source_unit_ids, ("u1", "u2"))
        service.extract_story_graph(
            book_id="book", reading_boundary=ReadingBoundary("book", 1, 1, "u2")
        )

        results = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 1, 0, "u1"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )
        self.assertTrue(results)
        self.assertTrue(all(result.chunk.section_index == 0 for result in results))

    def test_limits_are_validated_and_constrain_expansion(self) -> None:
        with self.assertRaises(ValueError):
            GraphRetrievalOptions(max_depth=5)
        service = configured_service()
        results = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 1, 0, "u1"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
            graph_options=GraphRetrievalOptions(max_additional_chunks=0),
        )
        self.assertTrue(all(result.retrieval_method == "hybrid" for result in results))

        depth_limited = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 1, 0, "u1"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
            graph_options=GraphRetrievalOptions(max_depth=0),
        )
        self.assertTrue(all(result.retrieval_method == "hybrid" for result in depth_limited))

        fact_limited = service.search_book(
            book_id="book",
            query="Mara",
            retrieval_mode="graph",
            reading_boundary=ReadingBoundary("book", 1, 0, "u1"),
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
            graph_options=GraphRetrievalOptions(max_facts=1),
        )
        self.assertTrue(all(result.retrieval_method == "hybrid" for result in fact_limited))

    def test_all_books_requires_one_boundary_per_book(self) -> None:
        service = configured_service()
        service.ingest_book("other", "v1", [source("other-unit", 0, "A different story.")])
        boundaries = {
            "book": ReadingBoundary("book", 1, 0, "u1"),
            "other": ReadingBoundary("other", 0, 0, "other-unit"),
        }
        results = service.search_all_books(
            query="Mara",
            retrieval_mode="graph",
            boundaries=boundaries,
            hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )
        self.assertTrue(results)
        self.assertEqual({item.chunk.book_id for item in results}, {"book"})
        with self.assertRaises(InvalidReadingBoundaryError):
            service.search_all_books(
                query="Mara",
                retrieval_mode="graph",
                boundaries={"book": boundaries["book"]},
                hybrid_config=HybridConfig(keyword_weight=1, semantic_weight=0),
            )

    def test_api_returns_graph_explanations_and_validates_options(self) -> None:
        client = TestClient(create_app(configured_service()))
        response = client.post(
            "/api/v1/books/book/query",
            json={
                "query": "Mara",
                "retrieval_mode": "graph",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 1,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
                "hybrid_options": {"keyword_weight": 1, "semantic_weight": 0},
            },
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["retrieval_mode"], "graph")
        self.assertTrue(any(result["graph_reasons"] for result in body["results"]))
        expanded = next(
            result
            for result in body["results"]
            if result["retrieval_method"] == "graph"
        )
        evidence = expanded["graph_reasons"][0]["evidence"][0]
        self.assertEqual(evidence["source_chunk_id"], expanded["chunk"]["id"])
        self.assertEqual(evidence["source_positions"][0]["cfi"].startswith("epubcfi("), True)

        invalid = client.post(
            "/api/v1/books/book/query",
            json={
                "query": "Mara",
                "retrieval_mode": "graph",
                "reading_boundary": {
                    "confirmed": True,
                    "section_index": 1,
                    "unit_index": 0,
                    "source_unit_id": "u1",
                },
                "graph_options": {"max_depth": 5},
            },
        )
        self.assertEqual(invalid.status_code, 422)


if __name__ == "__main__":
    unittest.main()
