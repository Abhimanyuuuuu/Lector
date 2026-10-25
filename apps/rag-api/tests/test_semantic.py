"""Tests for local cosine retrieval and the injected embedding boundary."""

import math
import unittest
from collections.abc import Sequence

from lector_rag_api.chunking import chunk_source_units
from lector_rag_api.retrieval_types import ReadingBoundary
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.semantic import (
    SemanticIndex,
    SemanticUnavailableError,
    cosine_similarity,
)


class FakeEmbeddingProvider:
    dimensions = 2

    def __init__(self, vectors: dict[str, Sequence[float]]) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        self.calls.append(tuple(texts))
        return [self.vectors[text] for text in texts]


def chunks_for(book_id: str, version: str, texts: list[str]):
    units = [
        SourceUnitInput(
            source_unit_id=f"{book_id}-u{index}",
            section_index=0,
            unit_index=index,
            kind="paragraph",
            text=text,
        )
        for index, text in enumerate(texts)
    ]
    return chunk_source_units(book_id, version, units, max_chunk_size=1, overlap=0)


def boundary(book_id: str, unit_index: int) -> ReadingBoundary:
    return ReadingBoundary(
        book_id=book_id,
        section_index=0,
        unit_index=unit_index,
        source_unit_id=f"{book_id}-u{unit_index}",
    )


class SemanticRetrievalTests(unittest.TestCase):
    def test_cosine_similarity_handles_identical_orthogonal_opposite_and_zero(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [0, 0]), 0.0)
        self.assertAlmostEqual(cosine_similarity([1e308, 0], [1e308, 0]), 1.0)

    def test_cosine_rejects_dimension_mismatch_and_non_finite_values(self) -> None:
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [1])
        with self.assertRaises(ValueError):
            cosine_similarity([math.nan, 0], [1, 0])
        with self.assertRaises(ValueError):
            cosine_similarity([1, 0], [math.inf, 0])

    def test_provider_embeddings_rank_by_cosine_and_return_provenance(self) -> None:
        chunks = chunks_for("book", "v1", ["near", "far"])
        provider = FakeEmbeddingProvider({"near": [1, 0], "far": [0, 1], "query": [1, 0]})
        index = SemanticIndex(provider)
        index.index_book("book", "v1", chunks)

        results = index.search(
            "query", book_id="book", boundaries={"book": boundary("book", 1)}
        )

        self.assertEqual([result.chunk.text for result in results], ["near", "far"])
        self.assertEqual(results[0].score, 1.0)
        self.assertEqual(results[0].retrieval_method, "semantic")
        self.assertEqual(results[0].matched_terms, ())
        self.assertEqual(results[0].chunk.source_unit_ids, ("book-u0",))
        self.assertEqual(results[0].chunk.source_positions[0].unit_index, 0)
        self.assertEqual(provider.calls, [("near", "far"), ("query",)])

    def test_precomputed_vectors_do_not_call_provider(self) -> None:
        chunks = chunks_for("book", "v1", ["indexed"])
        provider = FakeEmbeddingProvider({"indexed": [1, 0], "query": [1, 0]})
        index = SemanticIndex(provider)
        index.index_book_with_embeddings("book", "v1", chunks, [[1, 0]])
        calls_after_index = list(provider.calls)

        result = index.search_vector(
            [1, 0], book_id="book", boundaries={"book": boundary("book", 0)}
        )

        self.assertEqual(provider.calls, calls_after_index)
        self.assertEqual(len(result), 1)

    def test_empty_query_and_empty_index_do_not_call_provider(self) -> None:
        provider = FakeEmbeddingProvider({})
        index = SemanticIndex(provider)
        scope = {"book": boundary("book", 0)}

        self.assertEqual(index.search("", book_id="book", boundaries=scope), [])
        self.assertEqual(index.search("meaning", book_id="book", boundaries=scope), [])
        self.assertEqual(provider.calls, [])

    def test_unconfigured_provider_fails_clearly_for_text_query(self) -> None:
        index = SemanticIndex()
        index.index_book("book", "v1", chunks_for("book", "v1", ["text"]))

        with self.assertRaises(SemanticUnavailableError):
            index.search(
                "query", book_id="book", boundaries={"book": boundary("book", 0)}
            )

    def test_rejects_dimension_mismatch_non_finite_duplicate_and_missing_vectors(self) -> None:
        chunks = chunks_for("book", "v1", ["one", "two"])
        index = SemanticIndex()
        with self.assertRaises(ValueError):
            index.index_book_with_embeddings("book", "v1", chunks, [[1, 0], [1]])
        with self.assertRaises(ValueError):
            index.index_book_with_embeddings(
                "book", "v1", chunks, [[math.nan, 0], [1, 0]]
            )
        with self.assertRaises(ValueError):
            index.index_book_with_embeddings("book", "v1", [chunks[0], chunks[0]], [[1, 0], [1, 0]])

        index.index_book_with_embeddings("book", "v1", chunks, [[1, 0], None])
        results = index.search_vector(
            [1, 0], book_id="book", boundaries={"book": boundary("book", 1)}
        )
        self.assertEqual([result.chunk.id for result in results], [chunks[0].id])

    def test_missing_or_malformed_vectors_reject_bad_input(self) -> None:
        chunks = chunks_for("book", "v1", ["one"])
        index = SemanticIndex()
        with self.assertRaises(ValueError):
            index.index_book_with_embeddings("book", "v1", chunks, [])
        with self.assertRaises(TypeError):
            index.index_book_with_embeddings(
                "book", "v1", chunks, [[True, 0]]  # type: ignore[list-item]
            )
        with self.assertRaises(ValueError):
            index.search_vector(
                [math.inf, 0],
                book_id="book",
                boundaries={"book": boundary("book", 0)},
            )

    def test_book_and_content_version_replacement_removes_stale_vectors(self) -> None:
        index = SemanticIndex()
        index.index_book_with_embeddings(
            "book", "v1", chunks_for("book", "v1", ["old"]), [[1, 0]]
        )
        index.index_book_with_embeddings(
            "book", "v2", chunks_for("book", "v2", ["new"]), [[0, 1]]
        )
        results = index.search_vector(
            [0, 1], book_id="book", boundaries={"book": boundary("book", 0)}
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].chunk.content_version_id, "v2")
        self.assertEqual(results[0].chunk.text, "new")

    def test_book_scope_and_all_books_require_per_book_boundaries(self) -> None:
        index = SemanticIndex()
        index.index_book_with_embeddings(
            "a", "v1", chunks_for("a", "v1", ["apple"]), [[1, 0]]
        )
        index.index_book_with_embeddings(
            "b", "v1", chunks_for("b", "v1", ["banana"]), [[0, 1]]
        )

        only_a = index.search_vector(
            [1, 0], book_id="a", boundaries={"a": boundary("a", 0)}
        )
        all_books = index.search_vector(
            [1, 0],
            all_books=True,
            boundaries={"a": boundary("a", 0), "b": boundary("b", 0)},
        )

        self.assertEqual({result.chunk.book_id for result in only_a}, {"a"})
        self.assertEqual({result.chunk.book_id for result in all_books}, {"a", "b"})
        with self.assertRaises(ValueError):
            index.search_vector(
                [1, 0], all_books=True, boundaries={"a": boundary("a", 0)}
            )

    def test_provider_output_dimensions_and_batch_size_are_validated(self) -> None:
        chunks = chunks_for("book", "v1", ["one", "two"])

        class BadDimensionProvider:
            dimensions = 2

            def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
                return [[1.0] for _ in texts]

        class BadCountProvider:
            dimensions = 2

            def embed_texts(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
                return [[1.0, 0.0]]

        with self.assertRaises(ValueError):
            SemanticIndex(BadDimensionProvider()).index_book("book", "v1", chunks)
        with self.assertRaises(ValueError):
            SemanticIndex(BadCountProvider()).index_book("book", "v1", chunks)

    def test_boundary_filters_future_vectors_before_ranking(self) -> None:
        chunks = chunks_for("book", "v1", ["eligible", "future"])
        index = SemanticIndex()
        index.index_book_with_embeddings("book", "v1", chunks, [[1, 0], [0, 1]])

        result = index.search_vector(
            [0, 1], book_id="book", boundaries={"book": boundary("book", 0)}
        )

        self.assertEqual([item.chunk.text for item in result], ["eligible"])
        self.assertEqual(result[0].score, 0.0)

    def test_scope_requires_explicit_boundary_and_consistent_identity(self) -> None:
        index = SemanticIndex()
        index.index_book_with_embeddings(
            "book", "v1", chunks_for("book", "v1", ["one"]), [[1, 0]]
        )
        with self.assertRaises(ValueError):
            index.search_vector([1, 0], book_id="book")
        with self.assertRaises(ValueError):
            index.search_vector(
                [1, 0],
                book_id="book",
                boundaries={"book": ReadingBoundary("wrong", 0, 0, "wrong-u0")},
            )


if __name__ == "__main__":
    unittest.main()
