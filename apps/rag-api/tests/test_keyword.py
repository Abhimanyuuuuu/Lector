"""Tests for tokenizer behavior and explainable BM25 retrieval."""

import unittest

from lector_rag_api.chunking import chunk_source_units
from lector_rag_api.keyword import BM25Index, tokenize
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput


def chunks_for(
    book_id: str,
    version: str,
    texts: list[str],
    *,
    section: int = 0,
):
    units = [
        SourceUnitInput(
            source_unit_id=f"{book_id}-u{index}",
            section_index=section,
            unit_index=index,
            kind="paragraph",
            text=text,
        )
        for index, text in enumerate(texts)
    ]
    return chunk_source_units(
        book_id, version, units, max_chunk_size=1, overlap=0
    )


def boundary(book_id: str, unit_index: int) -> ReadingBoundary:
    return ReadingBoundary(
        book_id=book_id,
        section_index=0,
        unit_index=unit_index,
        source_unit_id=f"{book_id}-u{unit_index}",
    )


class KeywordRetrievalTests(unittest.TestCase):
    def test_tokenization_handles_unicode_case_punctuation_and_apostrophes(self) -> None:
        self.assertEqual(
            tokenize("CAFÉ café, O’Neill O'NEILL state-of-the-art foo_bar"),
            [
                "café",
                "café",
                "o’neill",
                "o'neill",
                "state",
                "of",
                "the",
                "art",
                "foo",
                "bar",
            ],
        )

    def test_bm25_ranks_relevant_text_and_exposes_matching_terms(self) -> None:
        index = BM25Index()
        index.index_book(
            "book",
            "v1",
            chunks_for(
                "book",
                "v1",
                [
                    "common rare rare",
                    "common common common common",
                    "common filler filler filler",
                ],
            ),
        )

        results = index.search(
            "RARE common",
            book_id="book",
            boundaries={"book": boundary("book", 2)},
        )

        self.assertEqual(results[0].chunk.source_unit_ids, ("book-u0",))
        self.assertEqual(results[0].matched_terms, ("rare", "common"))
        self.assertGreater(results[0].score, results[1].score)
        self.assertEqual(results[0].retrieval_method, "keyword")

    def test_duplicate_query_terms_do_not_change_the_score(self) -> None:
        index = BM25Index()
        index.index_book("book", "v1", chunks_for("book", "v1", ["one rare", "two common"]))
        scope = {"book": boundary("book", 1)}

        once = index.search("rare common", book_id="book", boundaries=scope)
        repeated = index.search("rare rare common common", book_id="book", boundaries=scope)

        self.assertEqual(
            [(item.chunk.id, item.score) for item in once],
            [(item.chunk.id, item.score) for item in repeated],
        )

    def test_stable_tie_breaking_uses_section_chunk_and_identity(self) -> None:
        chunks = chunks_for("book", "v1", ["same", "same", "same"])
        index = BM25Index()
        index.index_book("book", "v1", chunks)
        scope = {"book": boundary("book", 2)}

        first = index.search("same", book_id="book", boundaries=scope)
        second = index.search("same", book_id="book", boundaries=scope)

        self.assertEqual([item.chunk.id for item in first], [item.chunk.id for item in second])
        self.assertEqual(
            [item.chunk.chunk_index for item in first], [0, 1, 2]
        )

    def test_empty_index_empty_query_and_non_token_content(self) -> None:
        empty = BM25Index()
        scope = {"book": boundary("book", 0)}
        self.assertEqual(empty.search("word", book_id="book", boundaries=scope), [])
        self.assertEqual(empty.search("", book_id="book", boundaries=scope), [])
        self.assertEqual(empty.search("... ---", book_id="book", boundaries=scope), [])
        self.assertEqual(empty.search("word", book_id="book", boundaries=scope), [])

        index = BM25Index()
        index.index_book("book", "v1", chunks_for("book", "v1", ["   ", "---", "word"]))
        self.assertEqual(
            [result.chunk.text for result in index.search(
                "word", book_id="book", boundaries={"book": boundary("book", 2)}
            )],
            ["word"],
        )

    def test_book_isolation_and_explicit_all_books_scope(self) -> None:
        index = BM25Index()
        index.index_book("a", "v1", chunks_for("a", "v1", ["shared apple"]))
        index.index_book("b", "v1", chunks_for("b", "v1", ["shared banana"]))

        only_a = index.search(
            "shared", book_id="a", boundaries={"a": boundary("a", 0)}
        )
        all_books = index.search(
            "shared",
            all_books=True,
            boundaries={"a": boundary("a", 0), "b": boundary("b", 0)},
        )

        self.assertEqual({result.chunk.book_id for result in only_a}, {"a"})
        self.assertEqual({result.chunk.book_id for result in all_books}, {"a", "b"})
        with self.assertRaises(ValueError):
            index.search("shared", all_books=True, boundaries={"a": boundary("a", 0)})
        with self.assertRaises(ValueError):
            index.search("shared", book_id="a", all_books=True, boundaries={})

    def test_reindexing_replaces_version_terms_and_statistics(self) -> None:
        index = BM25Index()
        index.index_book("book", "v1", chunks_for("book", "v1", ["oldword oldword"]))
        index.index_book("book", "v2", chunks_for("book", "v2", ["newword"]))
        scope = {"book": boundary("book", 0)}

        self.assertEqual(index.search("oldword", book_id="book", boundaries=scope), [])
        result = index.search(
            "newword", book_id="book", content_version_id="v2", boundaries=scope
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].chunk.content_version_id, "v2")
        self.assertEqual(
            index.search("newword", book_id="book", content_version_id="v1", boundaries=scope),
            [],
        )

    def test_boundary_filters_candidates_before_bm25_statistics(self) -> None:
        eligible_chunks = chunks_for("book", "v1", ["alpha common", "common"])
        full_chunks = chunks_for(
            "book", "v1", ["alpha common", "common", "common futureword"]
        )
        eligible_index = BM25Index()
        eligible_index.index_book("book", "v1", eligible_chunks)
        full_index = BM25Index()
        full_index.index_book("book", "v1", full_chunks)
        reading_limit = {"book": boundary("book", 1)}

        expected = eligible_index.search(
            "alpha common", book_id="book", boundaries=reading_limit
        )
        actual = full_index.search(
            "alpha common", book_id="book", boundaries=reading_limit
        )

        self.assertEqual(
            [(item.chunk.id, item.score) for item in actual],
            [(item.chunk.id, item.score) for item in expected],
        )

    def test_result_preserves_chunk_provenance(self) -> None:
        chunk = chunks_for("book", "v1", ["Mara found the key"])[0]
        index = BM25Index()
        index.index_book("book", "v1", [chunk])

        result = index.search(
            "mara", book_id="book", boundaries={"book": boundary("book", 0)}
        )[0]

        self.assertIs(result.chunk, chunk)
        self.assertEqual(result.chunk.source_unit_ids, ("book-u0",))
        self.assertEqual(result.chunk.source_positions[0].unit_index, 0)
        self.assertEqual(result.chunk.book_id, "book")
        self.assertEqual(result.chunk.content_version_id, "v1")

    def test_rejects_malformed_chunks_and_bm25_parameters(self) -> None:
        index = BM25Index()
        with self.assertRaises(TypeError):
            index.index_book("book", "v1", [object()])  # type: ignore[list-item]
        duplicate = chunks_for("book", "v1", ["same"])[0]
        with self.assertRaises(ValueError):
            index.index_book("book", "v1", [duplicate, duplicate])
        with self.assertRaises(ValueError):
            BM25Index(k1=0)
        with self.assertRaises(ValueError):
            BM25Index(b=1.1)


if __name__ == "__main__":
    unittest.main()
