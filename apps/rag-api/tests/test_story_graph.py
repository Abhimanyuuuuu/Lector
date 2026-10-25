"""Deterministic evidence and boundary tests for the story graph foundation."""

import unittest
from typing import Callable, Sequence

from lector_rag_api.chunking import BookChunk
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import (
    InMemoryRagService,
    StoryExtractionUnavailableError,
)
from lector_rag_api.story_graph import (
    InvalidGraphEvidenceError,
    StoryExtraction,
)


class FakeStoryExtractor:
    name = "fixture-extractor"
    version = "test-1"

    def __init__(
        self,
        build: Callable[[Sequence[BookChunk]], StoryExtraction | dict[str, object]],
    ) -> None:
        self.build = build
        self.inputs: list[tuple[BookChunk, ...]] = []
        self.failure: Exception | None = None

    def extract(
        self, chunks: Sequence[BookChunk]
    ) -> StoryExtraction | dict[str, object]:
        self.inputs.append(tuple(chunks))
        if self.failure is not None:
            raise self.failure
        return self.build(chunks)


def unit(
    source_id: str,
    section: int,
    text: str,
    *,
    unit_index: int = 0,
) -> SourceUnitInput:
    return SourceUnitInput(
        source_unit_id=source_id,
        section_index=section,
        unit_index=unit_index,
        kind="paragraph",
        text=text,
        href=f"text/section-{section}.xhtml",
        cfi=f"epubcfi(/6/{section + 2}!/4/2/{unit_index + 1}:0,/1:5)",
    )


def boundary(section: int, source_id: str) -> ReadingBoundary:
    return ReadingBoundary("book", section, 0, source_id)


def evidence(chunk: BookChunk, source_ids: list[str] | None = None) -> dict[str, object]:
    return {
        "source_chunk_id": chunk.id,
        "source_unit_ids": source_ids or list(chunk.source_unit_ids),
    }


def entity_batch(chunks: Sequence[BookChunk], *, name: str = "Mara") -> dict[str, object]:
    chunk = chunks[0]
    return {
        "entities": [
            {
                "ref": "entity-ref-1",
                "name": name,
                "kind": "character",
                "aliases": ["M."],
                "assertedness": "explicit",
                "evidence": [evidence(chunk)],
            }
        ]
    }


class StoryGraphTests(unittest.TestCase):
    def make_service(
        self,
        extractor: FakeStoryExtractor,
        source_units: list[SourceUnitInput] | None = None,
    ) -> InMemoryRagService:
        service = InMemoryRagService(story_extractor=extractor)
        service.ingest_book(
            "book",
            "v1",
            source_units or [unit("u1", 0, "Mara met Jonas in the garden.")],
        )
        return service

    def test_entities_events_relationships_and_complete_evidence_are_built(self) -> None:
        def build(chunks: Sequence[BookChunk]) -> dict[str, object]:
            chunk = chunks[0]
            ev = evidence(chunk)
            return {
                "entities": [
                    {
                        "ref": "mara",
                        "name": "Mara Vale",
                        "kind": "character",
                        "aliases": ["Mara", "M. Vale"],
                        "description": "A traveler",
                        "assertedness": "explicit",
                        "evidence": [ev],
                    },
                    {
                        "ref": "garden",
                        "name": "Old Garden",
                        "kind": "location",
                        "aliases": ["the garden"],
                        "assertedness": "explicit",
                        "evidence": [ev],
                    },
                ],
                "events": [
                    {
                        "ref": "meeting",
                        "name": "Mara meets Jonas",
                        "description": "They meet at the garden.",
                        "participant_refs": ["mara"],
                        "assertedness": "explicit",
                        "evidence": [ev],
                    }
                ],
                "relationships": [
                    {
                        "subject_ref": "mara",
                        "object_ref": "garden",
                        "relationship_type": "visited",
                        "description": "Mara visits the garden.",
                        "temporal_metadata": {"when": "that morning"},
                        "assertedness": "inferred",
                        "evidence": [ev],
                    }
                ],
            }

        service = self.make_service(FakeStoryExtractor(build))
        graph = service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))

        self.assertEqual(len(graph.entities), 2)
        self.assertEqual(len(graph.events), 1)
        self.assertEqual(len(graph.relationships), 1)
        self.assertEqual(graph.relationships[0].assertedness, "inferred")
        self.assertEqual(graph.entities[0].evidence[0].book_id, "book")
        evidence_item = graph.entities[0].evidence[0]
        self.assertEqual(evidence_item.content_version_id, "v1")
        self.assertEqual(evidence_item.source_chunk_id, graph.entities[0].evidence[0].source_chunk_id)
        self.assertEqual(evidence_item.source_positions[0].href, "text/section-0.xhtml")
        self.assertTrue(evidence_item.source_positions[0].cfi.startswith("epubcfi("))
        self.assertEqual(evidence_item.extractor_name, "fixture-extractor")

    def test_ids_are_stable_repeat_extraction_is_idempotent_and_aliases_persist(self) -> None:
        extractor = FakeStoryExtractor(entity_batch)
        service = self.make_service(extractor)
        first = service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))
        first_id = first.entities[0].id
        second = service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))

        self.assertEqual(second.entities[0].id, first_id)
        self.assertEqual(len(second.entities), 1)
        self.assertEqual(len(second.entities[0].evidence), 1)
        self.assertEqual(second.entities[0].alias_names, ("M.",))
        self.assertEqual(second.entities[0].aliases[0].evidence[0].source_unit_ids, ("u1",))

    def test_exact_repeated_names_link_but_similar_names_remain_distinct(self) -> None:
        def build(chunks: Sequence[BookChunk]) -> dict[str, object]:
            return {
                "entities": [
                    {
                        "ref": "m1",
                        "name": "Mara",
                        "kind": "character",
                        "identity_status": "resolved",
                        "assertedness": "explicit",
                        "evidence": [evidence(chunks[0])],
                    },
                    {
                        "ref": "m2",
                        "name": "MARA",
                        "kind": "character",
                        "identity_status": "resolved",
                        "aliases": ["M."],
                        "assertedness": "explicit",
                        "evidence": [evidence(chunks[1])],
                    },
                    {
                        "ref": "m3",
                        "name": "Marauder",
                        "kind": "character",
                        "identity_status": "resolved",
                        "assertedness": "explicit",
                        "evidence": [evidence(chunks[1])],
                    },
                    {
                        "ref": "m4",
                        "name": "Mara",
                        "kind": "character",
                        "identity_status": "uncertain",
                        "assertedness": "inferred",
                        "evidence": [evidence(chunks[1])],
                    },
                ]
            }

        extractor = FakeStoryExtractor(build)
        service = self.make_service(
            extractor,
            [unit("u1", 0, "Mara arrives."), unit("u2", 1, "Mara returns.")],
        )
        graph = service.extract_story_graph(book_id="book", reading_boundary=boundary(1, "u2"))
        mara = [
            item
            for item in graph.entities
            if item.name.casefold() == "mara" and item.identity_status == "resolved"
        ]
        uncertain_mara = [
            item
            for item in graph.entities
            if item.name.casefold() == "mara" and item.identity_status == "uncertain"
        ]
        marauder = [item for item in graph.entities if item.name == "Marauder"]
        self.assertEqual(len(mara), 1)
        self.assertEqual(len(mara[0].evidence), 2)
        self.assertEqual(len(uncertain_mara), 1)
        self.assertNotEqual(uncertain_mara[0].id, mara[0].id)
        self.assertEqual(len(marauder), 1)
        self.assertNotEqual(mara[0].id, marauder[0].id)

    def test_invalid_chunk_or_unit_references_fail_without_mutating_graph(self) -> None:
        extractor = FakeStoryExtractor(entity_batch)
        service = self.make_service(extractor)
        committed = service.extract_story_graph(
            book_id="book", reading_boundary=boundary(0, "u1")
        )

        extractor.build = lambda chunks: {
            "entities": [
                {
                    "ref": "x",
                    "name": "Unknown source",
                    "kind": "other",
                    "assertedness": "explicit",
                    "evidence": [{"source_chunk_id": "missing", "source_unit_ids": ["u1"]}],
                }
            ]
        }
        with self.assertRaises(InvalidGraphEvidenceError):
            service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))
        self.assertIs(service.story_graph.get_book("book"), committed)

        extractor.build = lambda chunks: {
            "entities": [
                {
                    "ref": "x",
                    "name": "Unknown unit",
                    "kind": "other",
                    "assertedness": "explicit",
                    "evidence": [
                        {"source_chunk_id": chunks[0].id, "source_unit_ids": ["not-u1"]}
                    ],
                }
            ]
        }
        with self.assertRaises(InvalidGraphEvidenceError):
            service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))
        self.assertIs(service.story_graph.get_book("book"), committed)

    def test_schema_rejects_missing_evidence_and_unknown_entity_links(self) -> None:
        with self.assertRaises(ValueError):
            StoryExtraction.model_validate(
                {"entities": [{"ref": "x", "name": "Mara", "kind": "character"}]}
            )
        with self.assertRaises(ValueError):
            StoryExtraction.model_validate(
                {
                    "relationships": [
                        {
                            "subject_ref": "missing-a",
                            "object_ref": "missing-b",
                            "relationship_type": "knows",
                            "assertedness": "explicit",
                            "evidence": [
                                {"source_chunk_id": "c", "source_unit_ids": ["u"]}
                            ],
                        }
                    ]
                }
            )

    def test_confirmed_boundary_filters_chunks_before_extractor_call(self) -> None:
        extractor = FakeStoryExtractor(entity_batch)
        service = self.make_service(
            extractor,
            [unit("u1", 0, "Mara arrives."), unit("u2", 1, "Mara returns.")],
        )
        graph = service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))

        self.assertEqual(len(extractor.inputs), 1)
        self.assertEqual([chunk.section_index for chunk in extractor.inputs[0]], [0])
        self.assertEqual(graph.entities[0].evidence[0].source_unit_ids, ("u1",))

    def test_incremental_evidence_is_withheld_if_any_support_is_future(self) -> None:
        def build(chunks: Sequence[BookChunk]) -> dict[str, object]:
            refs = [evidence(chunk) for chunk in chunks]
            return {
                "entities": [
                    {
                        "ref": "mara",
                        "name": "Mara",
                        "kind": "character",
                        "assertedness": "explicit",
                        "evidence": refs,
                    },
                    {
                        "ref": "jonas",
                        "name": "Jonas",
                        "kind": "character",
                        "assertedness": "explicit",
                        "evidence": refs,
                    },
                ],
                "relationships": [
                    {
                        "subject_ref": "mara",
                        "object_ref": "jonas",
                        "relationship_type": "trusts",
                        "assertedness": "inferred",
                        "evidence": refs,
                    }
                ],
            }

        service = self.make_service(
            FakeStoryExtractor(build),
            [unit("u1", 0, "Mara trusts Jonas."), unit("u2", 1, "Mara trusts Jonas again.")],
        )
        earlier = service.extract_story_graph(
            book_id="book", reading_boundary=boundary(0, "u1")
        )
        self.assertEqual(len(earlier.eligible_facts(boundary(0, "u1"))[2]), 1)

        later = service.extract_story_graph(
            book_id="book", reading_boundary=boundary(1, "u2")
        )
        self.assertEqual(len(later.relationships), 1)
        self.assertEqual(len(later.relationships[0].evidence), 2)
        earlier_entities, _, earlier_relationships = later.eligible_facts(boundary(0, "u1"))
        self.assertEqual(earlier_entities, ())
        self.assertEqual(earlier_relationships, ())
        eligible_later = later.eligible_facts(boundary(1, "u2"))
        self.assertEqual(len(eligible_later[0]), 2)
        self.assertEqual(len(eligible_later[2]), 1)
        self.assertEqual(later.relationships[0].assertedness, "inferred")
        with self.assertRaises(ValueError):
            service.eligible_story_facts(
                book_id="book",
                reading_boundary=ReadingBoundary("book", 90, 0, "not-a-source"),
            )

        with self.assertRaises(ValueError):
            service.eligible_story_facts(
                book_id="book",
                reading_boundary=ReadingBoundary("book", 90, 0, "not-a-source"),
            )

    def test_same_version_reingestion_preserves_deduped_graph_and_new_version_clears_it(self) -> None:
        extractor = FakeStoryExtractor(entity_batch)
        service = self.make_service(extractor)
        original = service.extract_story_graph(
            book_id="book", reading_boundary=boundary(0, "u1")
        )
        service.ingest_book("book", "v1", [unit("u1", 0, "Mara met Jonas in the garden.")])
        retained = service.story_graph.get_book("book")
        self.assertIs(retained, original)

        service.ingest_book("book", "v2", [unit("u2", 0, "A different book version.")])
        replaced = service.story_graph.get_book("book")
        self.assertEqual(replaced.content_version_id, "v2")
        self.assertEqual(replaced.entities, ())
        self.assertEqual(replaced.relationships, ())

    def test_extractor_failure_is_transactional_and_missing_provider_is_clear(self) -> None:
        extractor = FakeStoryExtractor(entity_batch)
        service = self.make_service(extractor)
        committed = service.extract_story_graph(
            book_id="book", reading_boundary=boundary(0, "u1")
        )
        extractor.failure = RuntimeError("extractor failed")
        with self.assertRaisesRegex(RuntimeError, "extractor failed"):
            service.extract_story_graph(book_id="book", reading_boundary=boundary(0, "u1"))
        self.assertIs(service.story_graph.get_book("book"), committed)

        unconfigured = self.make_service(FakeStoryExtractor(entity_batch))
        unconfigured.story_extractor = None
        with self.assertRaises(StoryExtractionUnavailableError):
            unconfigured.extract_story_graph(
                book_id="book", reading_boundary=boundary(0, "u1")
            )


if __name__ == "__main__":
    unittest.main()
