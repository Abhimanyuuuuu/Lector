"""Small process-local chunk store used until a persistence phase is approved."""

from collections.abc import Sequence

from .chunking import BookChunk


class InMemoryChunkStore:
    """Atomically replace each book's indexed content within this process."""

    def __init__(self) -> None:
        self._books: dict[str, tuple[str, tuple[BookChunk, ...]]] = {}

    def replace_book(
        self,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
    ) -> None:
        self._books[book_id] = (content_version_id, tuple(chunks))

    def get_book(self, book_id: str) -> tuple[str, tuple[BookChunk, ...]] | None:
        return self._books.get(book_id)

    @property
    def book_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._books))
