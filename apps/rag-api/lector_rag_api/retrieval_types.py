"""Shared in-memory retrieval types and ordering constraints."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from .chunking import BookChunk

if TYPE_CHECKING:
    from .story_graph import GraphEvidence


@dataclass(frozen=True, slots=True)
class GraphRetrievalReason:
    """Evidence-backed explanation for graph-assisted chunk retrieval."""

    fact_id: str
    fact_kind: Literal["entity", "event", "relationship"]
    label: str
    assertedness: Literal["explicit", "inferred"]
    depth: int
    connection: str
    evidence: tuple["GraphEvidence", ...]


@dataclass(frozen=True, slots=True)
class ReadingBoundary:
    book_id: str
    section_index: int
    unit_index: int
    source_unit_id: str


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk: BookChunk
    score: float
    retrieval_method: Literal["keyword", "semantic", "hybrid", "graph"]
    matched_terms: tuple[str, ...] = ()
    keyword_rank: int | None = None
    keyword_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    graph_score: float | None = None
    graph_depth: int | None = None
    graph_reasons: tuple[GraphRetrievalReason, ...] = ()


def is_chunk_eligible(chunk: BookChunk, boundary: ReadingBoundary) -> bool:
    """Only expose a chunk when all its source units precede the boundary."""
    if chunk.book_id != boundary.book_id or not chunk.source_positions:
        return False
    if chunk.section_index < boundary.section_index:
        return True
    if chunk.section_index > boundary.section_index:
        return False
    return all(
        position.unit_index <= boundary.unit_index
        for position in chunk.source_positions
    )
