"""Behavioral tests for Python's port of the TypeScript Phase 2 chunker."""

import copy
import unittest

from pydantic import ValidationError

from lector_rag_api.chunking import chunk_source_units, utf16_code_units
from lector_rag_api.schemas import SourceUnitInput


def unit(
    source_id: str,
    index: int,
    text: str,
    *,
    section: int = 0,
    kind: str = "paragraph",
    href: str | None = "text/chapter.xhtml",
    cfi: str | None = None,
    heading: str | None = None,
) -> SourceUnitInput:
    return SourceUnitInput(
        source_unit_id=source_id,
        section_index=section,
        unit_index=index,
        kind=kind,  # type: ignore[arg-type]
        text=text,
        href=href,
        cfi=cfi,
        heading=heading,
    )


class ChunkingTests(unittest.TestCase):
    def test_empty_input_and_empty_text_produce_no_chunks(self) -> None:
        self.assertEqual(chunk_source_units("book", "v1", []), [])
        self.assertEqual(
            chunk_source_units("book", "v1", [unit("empty", 0, "")]), []
        )

    def test_combines_units_and_counts_separators_toward_budget(self) -> None:
        units = [unit("a", 0, "ab"), unit("b", 1, "cd")]

        combined = chunk_source_units("book", "v1", units, max_chunk_size=6, overlap=0)
        split = chunk_source_units("book", "v1", units, max_chunk_size=5, overlap=0)

        self.assertEqual([chunk.text for chunk in combined], ["ab\n\ncd"])
        self.assertEqual([chunk.text for chunk in split], ["ab", "cd"])

    def test_chunk_ids_are_deterministic_and_version_scoped(self) -> None:
        units = [unit("one", 0, "same")]
        first = chunk_source_units("book", "v1", units)[0]
        repeated = chunk_source_units("book", "v1", units)[0]
        other_book = chunk_source_units("other", "v1", units)[0]
        other_version = chunk_source_units("book", "v2", units)[0]

        self.assertEqual(first.id, repeated.id)
        self.assertNotEqual(first.id, other_book.id)
        self.assertNotEqual(first.id, other_version.id)

    def test_section_isolation_ordering_and_local_chunk_indexes(self) -> None:
        chunks = chunk_source_units(
            "book",
            "v1",
            [
                unit("later", 0, "later", section=3),
                unit("earlier", 2, "second", section=1),
                unit("first", 0, "first", section=1),
            ],
            max_chunk_size=5,
            overlap=0,
        )

        self.assertEqual([chunk.section_index for chunk in chunks], [1, 1, 3])
        self.assertEqual([chunk.chunk_index for chunk in chunks], [0, 1, 0])
        self.assertEqual([chunk.text for chunk in chunks], ["first", "second", "later"])
        self.assertEqual(len({chunk.section_index for chunk in chunks[:2]}), 1)

    def test_overlap_copies_whole_units_and_can_be_disabled(self) -> None:
        units = [
            unit("a", 0, "alpha"),
            unit("b", 1, "beta"),
            unit("c", 2, "gamma"),
        ]
        with_overlap = chunk_source_units(
            "book", "v1", units, max_chunk_size=11, overlap=5
        )
        without_overlap = chunk_source_units(
            "book", "v1", units, max_chunk_size=11, overlap=0
        )

        self.assertEqual([chunk.source_unit_ids for chunk in with_overlap], [("a", "b"), ("b", "c")])
        self.assertEqual([chunk.source_unit_ids for chunk in without_overlap], [("a", "b"), ("c",)])

    def test_oversized_source_unit_stays_intact_in_its_own_chunk(self) -> None:
        chunks = chunk_source_units(
            "book",
            "v1",
            [unit("large", 0, "oversized"), unit("small", 1, "x")],
            max_chunk_size=4,
            overlap=2,
        )

        self.assertEqual([chunk.text for chunk in chunks], ["oversized", "x"])
        self.assertEqual([chunk.source_unit_ids for chunk in chunks], [("large",), ("small",)])

    def test_headings_stay_with_following_content_and_lists_are_preserved(self) -> None:
        chunks = chunk_source_units(
            "book",
            "v1",
            [
                unit("h1", 0, "Chapter One", kind="heading"),
                unit("p1", 1, "Opening", heading="Chapter One"),
                unit("li", 2, "A clue", kind="list-item"),
                unit("h2", 3, "Chapter Two", kind="heading"),
                unit("p2", 4, "Later", heading="Chapter Two"),
            ],
            max_chunk_size=100,
            overlap=0,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].text, "Chapter One\n\nOpening\n\nA clue")
        self.assertEqual(chunks[0].heading, "Chapter One")
        self.assertEqual(chunks[0].source_positions[2].kind, "list-item")
        self.assertEqual(chunks[1].heading, "Chapter Two")

    def test_source_metadata_order_and_cfi_status_are_preserved(self) -> None:
        units = [
            unit("first", 0, "A", cfi="epubcfi(/6/2!/4/2:0,/1:1)"),
            unit("second", 1, "B", href="other.xhtml", cfi=None),
        ]
        chunk = chunk_source_units("book", "v1", units, max_chunk_size=20, overlap=0)[0]

        self.assertEqual(chunk.source_unit_ids, ("first", "second"))
        self.assertEqual([position.unit_index for position in chunk.source_positions], [0, 1])
        self.assertEqual(chunk.source_positions[1].href, "other.xhtml")
        self.assertEqual(chunk.source_positions[0].cfi, "epubcfi(/6/2!/4/2:0,/1:1)")
        self.assertEqual(chunk.start_cfi, "epubcfi(/6/2!/4/2:0,/1:1)")
        self.assertIsNone(chunk.end_cfi)
        self.assertEqual(chunk.cfi_status, "partial")
        self.assertEqual((chunk.start_unit_index, chunk.end_unit_index), (0, 1))

    def test_unavailable_and_unreliable_cfi_are_explicit(self) -> None:
        no_cfi = chunk_source_units("book", "v1", [unit("a", 0, "A")])[0]
        bad_cfi = chunk_source_units(
            "book", "v1", [unit("b", 0, "B", cfi="not-a-cfi")]
        )[0]

        self.assertEqual(no_cfi.cfi_status, "unavailable")
        self.assertIsNone(no_cfi.start_cfi)
        self.assertEqual(bad_cfi.cfi_status, "unreliable")
        self.assertEqual(bad_cfi.source_positions[0].cfi, "not-a-cfi")
        self.assertIsNone(bad_cfi.start_cfi)

    def test_utf16_budget_matches_typescript_for_non_bmp_characters(self) -> None:
        self.assertEqual(utf16_code_units("😀"), 2)
        chunks = chunk_source_units(
            "book",
            "v1",
            [unit("emoji", 0, "😀"), unit("letter", 1, "a")],
            max_chunk_size=4,
            overlap=0,
        )

        # Python code-point counting would fit these in four units; JavaScript
        # compatibility counts the emoji as two and also counts the separator.
        self.assertEqual([chunk.text for chunk in chunks], ["😀", "a"])

    def test_rejects_invalid_options_and_malformed_units(self) -> None:
        with self.assertRaises(ValueError):
            chunk_source_units("book", "v1", [], max_chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_source_units(
                "book", "v1", [], max_chunk_size=9_007_199_254_740_992
            )
        with self.assertRaises(ValueError):
            chunk_source_units("book", "v1", [], max_chunk_size=4, overlap=4)
        with self.assertRaises(TypeError):
            chunk_source_units("book", "v1", [], max_chunk_size=True)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            chunk_source_units("book", "v1", [object()])  # type: ignore[list-item]
        with self.assertRaises(ValueError):
            chunk_source_units(
                "book", "v1", [unit("a", 0, "A"), unit("a", 1, "B")]
            )
        with self.assertRaises(ValidationError):
            SourceUnitInput(
                source_unit_id="",
                section_index=-1,
                unit_index=0,
                kind="paragraph",
                text="bad",
            )

    def test_does_not_mutate_input_units_or_nested_metadata(self) -> None:
        units = [unit("a", 0, "A", heading="Heading", cfi="epubcfi(/6/2!/4/2:0,/1:1)")]
        before = copy.deepcopy(units)

        chunk_source_units("book", "v1", units)

        self.assertEqual(units, before)
        self.assertEqual(units[0].model_dump(), before[0].model_dump())


if __name__ == "__main__":
    unittest.main()
