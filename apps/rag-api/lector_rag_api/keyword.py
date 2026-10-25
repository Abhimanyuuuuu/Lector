"""Dependency-free Unicode tokenization and BM25 keyword retrieval."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence

from .chunking import BookChunk
from .retrieval_types import ReadingBoundary, RetrievalResult, is_chunk_eligible

_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)*", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """NFKC-normalize, casefold, and keep Unicode words/apostrophes together.

    Underscores, hyphens, and other punctuation separate tokens. Both straight
    and typographic apostrophes are retained when internal to a word.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


@dataclass(frozen=True, slots=True)
class _KeywordDocument:
    chunk: BookChunk
    frequencies: Mapping[str, int]
    length: int


@dataclass(frozen=True, slots=True)
class PreparedKeywordBook:
    book_id: str
    content_version_id: str
    documents: tuple[_KeywordDocument, ...]
    source_positions: frozenset[tuple[int, int, str]]


def validate_chunks(
    book_id: str,
    content_version_id: str,
    chunks: Sequence[BookChunk],
) -> tuple[BookChunk, ...]:
    if not isinstance(book_id, str) or not book_id.strip():
        raise ValueError("book_id must be a non-empty string")
    if not isinstance(content_version_id, str) or not content_version_id.strip():
        raise ValueError("content_version_id must be a non-empty string")
    if not isinstance(chunks, (list, tuple)):
        raise TypeError("chunks must be a list or tuple")
    seen_ids: set[str] = set()
    prepared: list[BookChunk] = []
    for chunk in chunks:
        if not isinstance(chunk, BookChunk):
            raise TypeError("each chunk must be a BookChunk")
        if (
            chunk.book_id != book_id
            or chunk.content_version_id != content_version_id
        ):
            raise ValueError("chunk book/content-version identity does not match")
        if not isinstance(chunk.id, str) or not chunk.id:
            raise ValueError("chunk ID must be a non-empty string")
        if chunk.id in seen_ids:
            raise ValueError(f"duplicate chunk ID: {chunk.id}")
        seen_ids.add(chunk.id)
        if not isinstance(chunk.text, str):
            raise TypeError("chunk text must be a string")
        prepared.append(chunk)
    return tuple(prepared)


class BM25Index:
    """Local BM25 index with explicit scope and pre-ranking boundary filtering.

    Scores use Robertson/Sparck Jones IDF `ln(1 + (N-df+0.5)/(df+0.5))` and
    standard BM25 term saturation/length normalization. Defaults are `k1=1.5`
    and `b=0.75`. Duplicate query terms count once.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        if isinstance(k1, bool) or not isinstance(k1, (int, float)):
            raise TypeError("k1 must be a finite positive number")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError("k1 must be a finite positive number")
        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError("b must be a finite number from 0 to 1")
        if not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("b must be a finite number from 0 to 1")
        self.k1 = float(k1)
        self.b = float(b)
        self._books: dict[str, PreparedKeywordBook] = {}

    def prepare_book(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
    ) -> PreparedKeywordBook:
        validated = validate_chunks(book_id, content_version_id, chunks)
        documents: list[_KeywordDocument] = []
        for chunk in validated:
            terms = tokenize(chunk.text)
            if not terms:
                continue
            documents.append(
                _KeywordDocument(
                    chunk=chunk,
                    frequencies=dict(Counter(terms)),
                    length=len(terms),
                )
            )
        documents.sort(key=_document_order)
        positions = frozenset(
            (chunk.section_index, position.unit_index, position.source_unit_id)
            for chunk in validated
            for position in chunk.source_positions
        )
        return PreparedKeywordBook(
            book_id, content_version_id, tuple(documents), positions
        )

    def commit_book(self, prepared: PreparedKeywordBook) -> None:
        self._books[prepared.book_id] = prepared

    def index_book(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
    ) -> None:
        self.commit_book(self.prepare_book(book_id, content_version_id, chunks))

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
        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms or limit == 0:
            return []

        candidates = self._eligible_documents(
            book_id, content_version_id, all_books, boundaries
        )
        # IDF and average length are computed only after the eligibility filter.
        if not candidates:
            return []
        document_frequency: Counter[str] = Counter()
        for document in candidates:
            document_frequency.update(document.frequencies.keys())
        document_count = len(candidates)
        average_length = sum(doc.length for doc in candidates) / document_count

        results: list[RetrievalResult] = []
        for document in candidates:
            score = 0.0
            matched: list[str] = []
            for term in query_terms:
                term_frequency = document.frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                matched.append(term)
                df = document_frequency[term]
                inverse_frequency = math.log(
                    1 + (document_count - df + 0.5) / (df + 0.5)
                )
                normalization = self.k1 * (
                    1 - self.b + self.b * document.length / average_length
                )
                score += (
                    inverse_frequency
                    * term_frequency
                    * (self.k1 + 1)
                    / (term_frequency + normalization)
                )
            if matched:
                results.append(
                    RetrievalResult(
                        chunk=document.chunk,
                        score=score,
                        retrieval_method="keyword",
                        matched_terms=tuple(matched),
                    )
                )
        results.sort(key=_result_order)
        return results[:limit]

    def _eligible_documents(
        self,
        book_id: str | None,
        content_version_id: str | None,
        all_books: bool,
        boundaries: Mapping[str, ReadingBoundary] | None,
    ) -> list[_KeywordDocument]:
        selected: list[_KeywordDocument] = []
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
            if not contains_source_position(book, boundary):
                raise ValueError(
                    f"reading boundary source is not in indexed book {book.book_id!r}"
                )
            for document in book.documents:
                if is_chunk_eligible(document.chunk, boundary):
                    selected.append(document)
        selected.sort(key=lambda document: _document_order(document))
        return selected


def _validate_scope(
    book_id: str | None,
    content_version_id: str | None,
    all_books: bool,
    boundaries: Mapping[str, ReadingBoundary] | None,
    limit: int,
) -> None:
    if not isinstance(all_books, bool):
        raise TypeError("all_books must be a boolean")
    if book_id is not None and (
        not isinstance(book_id, str) or not book_id.strip()
    ):
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


def contains_source_position(
    book: PreparedKeywordBook, boundary: ReadingBoundary
) -> bool:
    return (
        boundary.section_index,
        boundary.unit_index,
        boundary.source_unit_id,
    ) in book.source_positions


def _document_order(document: _KeywordDocument) -> tuple[int, int, str, str, str]:
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
