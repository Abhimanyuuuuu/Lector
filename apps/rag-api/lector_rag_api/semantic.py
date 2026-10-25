"""Provider-neutral embedding and local cosine-similarity retrieval."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Mapping, Protocol, Sequence, runtime_checkable

from .chunking import BookChunk
from .keyword import validate_chunks
from .retrieval_types import ReadingBoundary, RetrievalResult, is_chunk_eligible


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Injectable embedder; dimension is fixed for its lifetime.

    ``embed_texts`` accepts a batch (including a one-element query batch) and
    returns one finite numeric vector of exactly ``dimensions`` values per
    input. Implementations are responsible for local processing unless the
    caller explicitly configures an approved remote provider.
    """

    @property
    def dimensions(self) -> int: ...

    def embed_texts(
        self, texts: Sequence[str]
    ) -> Sequence[Sequence[float]]: ...


class SemanticUnavailableError(RuntimeError):
    """Raised when semantic search has no configured provider or vectors."""


@dataclass(frozen=True, slots=True)
class _VectorDocument:
    chunk: BookChunk
    vector: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class PreparedSemanticBook:
    book_id: str
    content_version_id: str
    documents: tuple[_VectorDocument, ...]
    source_positions: frozenset[tuple[int, int, str]]
    dimensions: int | None


class SemanticIndex:
    """Semantic index storing vectors in memory and ranking by cosine."""

    def __init__(self, provider: EmbeddingProvider | None = None) -> None:
        self.provider = provider
        self.dimensions = _provider_dimensions(provider) if provider else None
        self._books: dict[str, PreparedSemanticBook] = {}

    def prepare_book(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
    ) -> PreparedSemanticBook:
        validated = validate_chunks(book_id, content_version_id, chunks)
        source_positions = _source_positions(validated)
        indexable = tuple(chunk for chunk in validated if chunk.text.strip())
        if not indexable:
            return PreparedSemanticBook(
                book_id, content_version_id, (), source_positions, self.dimensions
            )
        if self.provider is None:
            return PreparedSemanticBook(
                book_id, content_version_id, (), source_positions, self.dimensions
            )
        vectors = self.provider.embed_texts([chunk.text for chunk in indexable])
        return self._prepare_vectors(
            book_id,
            content_version_id,
            indexable,
            vectors,
            self.dimensions,
            source_positions,
        )

    def prepare_book_with_embeddings(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
        vectors: Sequence[Sequence[float] | None],
    ) -> PreparedSemanticBook:
        validated = validate_chunks(book_id, content_version_id, chunks)
        if not isinstance(vectors, Sequence) or len(vectors) != len(validated):
            raise ValueError("one vector or None is required per input chunk")
        dimensions = self.dimensions
        documents: list[_VectorDocument] = []
        for chunk, vector in zip(validated, vectors, strict=True):
            if not chunk.text.strip() or vector is None:
                continue
            valid_vector = validate_vector(
                vector,
                dimensions,
                name=f"embedding for chunk {chunk.id}",
            )
            if dimensions is None:
                dimensions = len(valid_vector)
            documents.append(_VectorDocument(chunk, valid_vector))
        return PreparedSemanticBook(
            book_id,
            content_version_id,
            tuple(sorted(documents, key=_document_order)),
            _source_positions(validated),
            dimensions,
        )

    def commit_book(self, prepared: PreparedSemanticBook) -> None:
        self.validate_prepared(prepared)
        self._books[prepared.book_id] = prepared
        if self.dimensions is None and prepared.dimensions is not None:
            self.dimensions = prepared.dimensions

    def validate_prepared(self, prepared: PreparedSemanticBook) -> None:
        if (
            self.dimensions is not None
            and prepared.dimensions is not None
            and prepared.dimensions != self.dimensions
        ):
            raise ValueError("embedding dimension does not match semantic index")

    def index_book(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
    ) -> None:
        self.commit_book(self.prepare_book(book_id, content_version_id, chunks))

    def index_book_with_embeddings(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
        vectors: Sequence[Sequence[float] | None],
    ) -> None:
        self.commit_book(
            self.prepare_book_with_embeddings(
                book_id, content_version_id, chunks, vectors
            )
        )

    def remove_book(self, book_id: str) -> None:
        self._books.pop(book_id, None)

    def search(
        self,
        query: str,
        *,
        book_id: str | None = None,
        content_version_id: str | None = None,
        all_books: bool = False,
        boundaries: Mapping[str, ReadingBoundary] | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        _validate_scope(book_id, content_version_id, all_books, boundaries, limit)
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if not query.strip() or limit == 0:
            return []
        if self.provider is None:
            raise SemanticUnavailableError(
                "semantic retrieval requires an explicitly configured embedding provider"
            )
        candidates = self._eligible_documents(
            book_id, content_version_id, all_books, boundaries
        )
        if not candidates:
            return []
        query_vectors = self.provider.embed_texts([query])
        if not isinstance(query_vectors, Sequence) or len(query_vectors) != 1:
            raise ValueError("embedding provider must return one query vector")
        query_vector = validate_vector(
            query_vectors[0], self.dimensions, name="query embedding"
        )
        return self._rank(query_vector, candidates, limit)

    def search_vector(
        self,
        query_vector: Sequence[float],
        *,
        book_id: str | None = None,
        content_version_id: str | None = None,
        all_books: bool = False,
        boundaries: Mapping[str, ReadingBoundary] | None = None,
        limit: int = 10,
    ) -> list[RetrievalResult]:
        _validate_scope(book_id, content_version_id, all_books, boundaries, limit)
        vector = validate_vector(
            query_vector, self.dimensions, name="query vector"
        )
        if limit == 0:
            return []
        candidates = self._eligible_documents(
            book_id, content_version_id, all_books, boundaries
        )
        if not candidates:
            return []
        if self.dimensions is None:
            raise SemanticUnavailableError(
                "semantic index contains no embeddings"
            )
        return self._rank(vector, candidates, limit)

    def _eligible_documents(
        self,
        book_id: str | None,
        content_version_id: str | None,
        all_books: bool,
        boundaries: Mapping[str, ReadingBoundary] | None,
    ) -> list[_VectorDocument]:
        if book_id is not None:
            book = self._books.get(book_id)
            books = [book] if book is not None else []
        elif all_books:
            if boundaries is None or set(boundaries) != set(self._books):
                raise ValueError(
                    "all-books search requires a confirmed boundary for every indexed book"
                )
            books = list(self._books.values())
        else:
            books = []
        selected: list[_VectorDocument] = []
        for book in books:
            if book is None:
                continue
            if content_version_id and book.content_version_id != content_version_id:
                continue
            boundary = boundaries.get(book.book_id) if boundaries else None
            if boundary is None:
                raise ValueError(
                    f"book {book.book_id!r} has no explicit reading boundary"
                )
            if not _contains_boundary_source(book, boundary):
                raise ValueError(
                    f"reading boundary source is not in indexed book {book.book_id!r}"
                )
            selected.extend(
                document
                for document in book.documents
                if is_chunk_eligible(document.chunk, boundary)
            )
        return sorted(selected, key=_document_order)

    @staticmethod
    def _rank(
        query_vector: tuple[float, ...],
        candidates: Sequence[_VectorDocument],
        limit: int,
    ) -> list[RetrievalResult]:
        results = [
            RetrievalResult(
                chunk=document.chunk,
                score=cosine_similarity(query_vector, document.vector),
                retrieval_method="semantic",
            )
            for document in candidates
        ]
        results.sort(key=_result_order)
        return results[:limit]

    def _prepare_vectors(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
        vectors: Sequence[Sequence[float]],
        dimensions: int | None,
        source_positions: frozenset[tuple[int, int, str]],
    ) -> PreparedSemanticBook:
        if not isinstance(vectors, Sequence) or len(vectors) != len(chunks):
            raise ValueError("embedding provider must return one vector per chunk")
        prepared_vectors: list[_VectorDocument] = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            valid = validate_vector(vector, dimensions, name=f"embedding for {chunk.id}")
            if dimensions is None:
                dimensions = len(valid)
            prepared_vectors.append(_VectorDocument(chunk, valid))
        return PreparedSemanticBook(
            book_id,
            content_version_id,
            tuple(sorted(prepared_vectors, key=_document_order)),
            source_positions,
            dimensions,
        )


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_vector = validate_vector(left, None, name="left vector")
    right_vector = validate_vector(right, len(left_vector), name="right vector")
    left_norm = math.hypot(*left_vector)
    right_norm = math.hypot(*right_vector)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    similarity = math.fsum(
        (a / left_norm) * (b / right_norm)
        for a, b in zip(left_vector, right_vector, strict=True)
    )
    return max(-1.0, min(1.0, similarity))


def validate_vector(
    vector: Sequence[float], dimensions: int | None, *, name: str
) -> tuple[float, ...]:
    if not isinstance(vector, Sequence) or isinstance(vector, (str, bytes)):
        raise TypeError(f"{name} must be a numeric vector")
    if dimensions is not None and len(vector) != dimensions:
        raise ValueError(f"{name} must have exactly {dimensions} dimensions")
    if len(vector) == 0:
        raise ValueError(f"{name} must not be empty")
    values: list[float] = []
    for value in vector:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must contain only finite real numbers")
        try:
            numeric = float(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(f"{name} must contain only finite real numbers") from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must contain only finite real numbers")
        values.append(numeric)
    return tuple(values)


def _provider_dimensions(provider: EmbeddingProvider) -> int:
    dimensions = provider.dimensions
    if (
        isinstance(dimensions, bool)
        or not isinstance(dimensions, int)
        or dimensions <= 0
    ):
        raise ValueError("embedding provider dimensions must be a positive integer")
    return dimensions


def _source_positions(
    chunks: Sequence[BookChunk],
) -> frozenset[tuple[int, int, str]]:
    return frozenset(
        (chunk.section_index, position.unit_index, position.source_unit_id)
        for chunk in chunks
        for position in chunk.source_positions
    )


def _contains_boundary_source(
    book: PreparedSemanticBook, boundary: ReadingBoundary
) -> bool:
    return (
        boundary.section_index,
        boundary.unit_index,
        boundary.source_unit_id,
    ) in book.source_positions


def _validate_scope(
    book_id: str | None,
    content_version_id: str | None,
    all_books: bool,
    boundaries: Mapping[str, ReadingBoundary] | None,
    limit: int,
) -> None:
    if not isinstance(all_books, bool):
        raise TypeError("all_books must be a boolean")
    if book_id is not None and (not isinstance(book_id, str) or not book_id.strip()):
        raise ValueError("book_id must be a non-empty string when supplied")
    if all_books == (book_id is not None):
        raise ValueError("provide either book_id or all_books=True")
    if content_version_id is not None and (
        book_id is None
        or not isinstance(content_version_id, str)
        or not content_version_id.strip()
    ):
        raise ValueError("content_version_id requires a non-empty book_id")
    if boundaries is None or not isinstance(boundaries, Mapping):
        raise ValueError("explicit reading boundaries are required")
    for boundary_book_id, boundary in boundaries.items():
        if not isinstance(boundary, ReadingBoundary):
            raise TypeError("each reading boundary must be a ReadingBoundary")
        if (
            boundary.book_id != boundary_book_id
            or not boundary.book_id.strip()
            or boundary.section_index < 0
            or boundary.unit_index < 0
            or not boundary.source_unit_id.strip()
        ):
            raise ValueError("reading boundary metadata is invalid")
    if book_id is not None and book_id not in boundaries:
        raise ValueError("the scoped book requires an explicit reading boundary")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")


def _document_order(document: _VectorDocument) -> tuple[int, int, str, str, str]:
    chunk = document.chunk
    return (
        chunk.section_index,
        chunk.chunk_index,
        chunk.book_id,
        chunk.content_version_id,
        chunk.id,
    )


def _result_order(result: RetrievalResult) -> tuple[float, int, int, str, str, str]:
    chunk = result.chunk
    return (
        -result.score,
        chunk.section_index,
        chunk.chunk_index,
        chunk.book_id,
        chunk.content_version_id,
        chunk.id,
    )
