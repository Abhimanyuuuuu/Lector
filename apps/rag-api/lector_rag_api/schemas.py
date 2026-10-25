"""Versioned HTTP schemas; these are independent of internal RAG models."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


UnitKind = Literal[
    "paragraph",
    "heading",
    "list-item",
    "blockquote",
    "preformatted",
    "other",
]


class SourceUnitInput(BaseModel):
    """A readable unit emitted from Flow's already-loaded EPUB DOM."""

    model_config = ConfigDict(extra="forbid")

    source_unit_id: str = Field(min_length=1)
    section_index: int = Field(ge=0)
    unit_index: int = Field(ge=0)
    kind: UnitKind
    text: str
    href: str | None = None
    cfi: str | None = None
    heading: str | None = None


class IndexBookRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_version_id: str = Field(min_length=1)
    source_units: list[SourceUnitInput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_source_identity_and_order(self) -> "IndexBookRequest":
        ids = [unit.source_unit_id for unit in self.source_units]
        positions = [
            (unit.section_index, unit.unit_index) for unit in self.source_units
        ]
        if len(ids) != len(set(ids)):
            raise ValueError("source_unit_id values must be unique")
        if len(positions) != len(set(positions)):
            raise ValueError("section_index/unit_index positions must be unique")
        return self


class ConfirmedReadingBoundary(BaseModel):
    """Inclusive source position explicitly confirmed by the reader."""

    model_config = ConfigDict(extra="forbid")

    confirmed: Literal[True]
    section_index: int = Field(ge=0)
    unit_index: int = Field(ge=0)
    source_unit_id: str = Field(min_length=1)
    cfi: str | None = None


class GraphExtractionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    book_id: str
    content_version_id: str
    boundary_section: int | None
    boundary_unit: int | None
    boundary_source_unit_id: str | None
    total_chunks: int
    processed_chunks: int
    status: Literal["queued", "processing", "persisting", "completed", "failed", "cancelled", "unavailable"]
    attempts: int
    error: str | None = None
    updated_at: str
    provider: str = "booknlp"
    provider_version: str = ""
    job_scope: Literal["confirmed_boundary", "full_book"] = "confirmed_boundary"


class GraphExtractionRequest(ConfirmedReadingBoundary):
    """Legacy confirmed-boundary request; graph extraction is local-only."""


class CharacterGraphViewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content_version_id: str = Field(min_length=1)
    view_mode: Literal["full_book", "till_now"] = "till_now"
    confirmed: Literal[True] | None = None
    section_index: int | None = Field(default=None, ge=0)
    unit_index: int | None = Field(default=None, ge=0)
    source_unit_id: str | None = Field(default=None, min_length=1)
    cfi: str | None = None

    @model_validator(mode="after")
    def validate_view_boundary(self) -> "CharacterGraphViewRequest":
        fields = (self.section_index, self.unit_index, self.source_unit_id)
        if self.view_mode == "till_now":
            if self.confirmed is not True or any(value is None for value in fields):
                raise ValueError("till_now graph view requires a confirmed reading boundary")
        elif any(value is not None for value in fields) or self.confirmed is not None:
            raise ValueError("full_book graph view must not include a reading boundary")
        return self


class CharacterGraphViewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    book_id: str
    content_version_id: str
    view_mode: Literal["full_book", "till_now"] = "till_now"
    boundary: dict[str, object] | None
    nodes: list[dict[str, object]]
    edges: list[dict[str, object]]


class HybridOptions(BaseModel):
    """Request-level RRF controls; ignored for non-hybrid retrieval modes."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(default=60, ge=1, strict=True)
    keyword_weight: float = Field(default=1.0, ge=0, strict=True)
    semantic_weight: float = Field(default=1.0, ge=0, strict=True)

    @model_validator(mode="after")
    def validate_weights(self) -> "HybridOptions":
        if not math.isfinite(self.keyword_weight) or not math.isfinite(
            self.semantic_weight
        ):
            raise ValueError("hybrid weights must be finite numbers")
        if self.keyword_weight == 0 and self.semantic_weight == 0:
            raise ValueError("at least one hybrid weight must be positive")
        return self


class GraphOptions(BaseModel):
    """Conservative bounds for graph-assisted retrieval expansion."""

    model_config = ConfigDict(extra="forbid")

    max_depth: int = Field(default=2, ge=0, le=4, strict=True)
    max_facts: int = Field(default=12, ge=0, le=100, strict=True)
    max_additional_chunks: int = Field(default=8, ge=0, le=100, strict=True)
    max_facts_per_entity: int = Field(default=3, ge=0, le=10, strict=True)


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    retrieval_mode: Literal["keyword", "semantic", "hybrid", "hybrid_graph", "graph"] = "keyword"
    content_version_id: str | None = None
    reading_boundary: ConfirmedReadingBoundary
    limit: int = Field(default=10, ge=0, le=100)
    hybrid_options: HybridOptions = Field(default_factory=HybridOptions)
    graph_options: GraphOptions = Field(default_factory=GraphOptions)

    @model_validator(mode="after")
    def validate_query_text(self) -> "QueryRequest":
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        if self.content_version_id is not None and not self.content_version_id.strip():
            raise ValueError("content_version_id must not be empty")
        return self


class BookReadingBoundary(ConfirmedReadingBoundary):
    book_id: str = Field(min_length=1)


class AllBooksQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    retrieval_mode: Literal["keyword", "semantic", "hybrid", "hybrid_graph", "graph"] = "keyword"
    all_books: Literal[True]
    reading_boundaries: list[BookReadingBoundary]
    limit: int = Field(default=10, ge=0, le=100)
    hybrid_options: HybridOptions = Field(default_factory=HybridOptions)
    graph_options: GraphOptions = Field(default_factory=GraphOptions)

    @model_validator(mode="after")
    def validate_query_and_boundaries(self) -> "AllBooksQueryRequest":
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        ids = [boundary.book_id for boundary in self.reading_boundaries]
        if len(ids) != len(set(ids)):
            raise ValueError("reading_boundaries must contain one boundary per book")
        return self


class AnswerContextOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_chars: int = Field(default=12_000, ge=1, le=100_000, strict=True)
    max_passages: int = Field(default=8, ge=1, le=100, strict=True)
    max_graph_facts: int = Field(default=8, ge=0, le=100, strict=True)


class AnswerRequest(BaseModel):
    """Question request with exactly one book or explicit all-books scope."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1)
    book_id: str | None = Field(default=None, min_length=1)
    all_books: Literal[True] | None = None
    content_version_id: str | None = Field(default=None, min_length=1)
    reading_boundary: ConfirmedReadingBoundary | None = None
    reading_boundaries: list[BookReadingBoundary] = Field(default_factory=list)
    retrieval_mode: Literal["keyword", "semantic", "hybrid", "hybrid_graph", "graph"] = "keyword"
    limit: int = Field(default=10, ge=0, le=100)
    hybrid_options: HybridOptions = Field(default_factory=HybridOptions)
    graph_options: GraphOptions = Field(default_factory=GraphOptions)
    context_options: AnswerContextOptions = Field(default_factory=AnswerContextOptions)

    @model_validator(mode="after")
    def validate_scope_and_query(self) -> "AnswerRequest":
        if not self.query.strip():
            raise ValueError("query must contain non-whitespace text")
        if self.book_id is not None:
            if (
                not self.book_id.strip()
                or self.all_books is not None
                or self.reading_boundary is None
                or self.reading_boundaries
            ):
                raise ValueError(
                    "book scope requires one reading_boundary and no all-books fields"
                )
            if self.content_version_id is not None and not self.content_version_id.strip():
                raise ValueError("content_version_id must not be empty")
        elif self.all_books is True:
            if (
                self.reading_boundary is not None
                or self.content_version_id is not None
                or not self.reading_boundaries
            ):
                raise ValueError(
                    "all-books scope requires a boundary for each book and no single-book fields"
                )
            ids = [item.book_id for item in self.reading_boundaries]
            if len(ids) != len(set(ids)):
                raise ValueError("reading_boundaries must contain one boundary per book")
        else:
            raise ValueError("provide book_id or explicit all_books=true scope")
        return self


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_unit_id: str
    section_index: int = Field(ge=0)
    unit_index: int = Field(ge=0)
    href: str | None = None
    cfi: str | None = None
    excerpt: str


class IndexAcceptedResponse(BaseModel):
    book_id: str
    content_version_id: str
    accepted_source_units: int
    chunk_count: int


class SavedBookResponse(BaseModel):
    book_id: str
    title: str
    original_filename: str
    content_version_id: str
    file_size: int
    indexing_status: Literal["uploaded", "indexed", "stale"]
    preparation_indexing_status: Literal["pending", "queued", "processing", "complete", "failed"] = "pending"
    indexing_progress: float = 0
    indexing_completed: int = 0
    indexing_total: int = 0
    indexing_error: str | None = None
    embedding_status: Literal["pending", "processing", "complete", "failed", "unavailable", "not_applicable"] = "pending"
    embedding_progress: float = 0
    embedding_completed: int = 0
    embedding_total: int = 0
    embedding_error: str | None = None
    graph_rag_enabled: bool = False
    graph_status: Literal["not_enabled", "pending", "queued", "processing", "complete", "failed", "unavailable"] = "not_enabled"
    graph_progress: float = 0
    graph_completed: int = 0
    graph_total: int = 0
    graph_error: str | None = None
    file_available: bool
    updated_at: str


class SavedBookListResponse(BaseModel):
    books: list[SavedBookResponse]


class SavedEpubResponse(BaseModel):
    book_id: str
    content_version_id: str
    file_size: int
    indexing_status: Literal["uploaded", "indexed", "stale"]


class BookPreparationRequest(BaseModel):
    content_version_id: str = Field(min_length=1, max_length=256)
    source_units: list[SourceUnitInput] = Field(min_length=1, max_length=100000)
    graph_rag_enabled: bool | None = None

    @model_validator(mode="after")
    def validate_source_identity_and_order(self) -> "BookPreparationRequest":
        ids = [unit.source_unit_id for unit in self.source_units]
        positions = [(unit.section_index, unit.unit_index) for unit in self.source_units]
        if len(ids) != len(set(ids)):
            raise ValueError("source_unit_id values must be unique")
        if len(positions) != len(set(positions)):
            raise ValueError("section_index/unit_index positions must be unique")
        return self


class BookPreparationResponse(BaseModel):
    book_id: str
    content_version_id: str
    indexing_status: Literal["queued", "processing", "complete", "failed"]
    embedding_status: Literal["pending", "processing", "complete", "failed", "unavailable", "not_applicable"]
    indexing_completed: int = 0
    indexing_total: int = 0
    embedding_completed: int = 0
    embedding_total: int = 0
    indexing_error: str | None = None
    embedding_error: str | None = None
    graph_rag_enabled: bool = False
    graph_status: Literal["not_enabled", "pending", "queued", "processing", "complete", "failed", "unavailable"] = "not_enabled"
    graph_completed: int = 0
    graph_total: int = 0
    graph_error: str | None = None
    updated_at: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation]


class SourcePositionResponse(BaseModel):
    source_unit_id: str
    unit_index: int
    kind: UnitKind
    href: str | None = None
    cfi: str | None = None


class ChunkResponse(BaseModel):
    id: str
    book_id: str
    content_version_id: str
    section_index: int
    chunk_index: int
    text: str
    source_unit_ids: list[str]
    source_positions: list[SourcePositionResponse]
    start_unit_index: int
    end_unit_index: int
    href: str | None = None
    heading: str | None = None
    start_cfi: str | None = None
    end_cfi: str | None = None
    cfi_status: Literal["complete", "partial", "unavailable", "unreliable"]


class RetrievalResultResponse(BaseModel):
    chunk: ChunkResponse
    score: float
    retrieval_method: Literal["keyword", "semantic", "hybrid", "graph"]
    matched_terms: list[str]
    keyword_rank: int | None = None
    keyword_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    graph_score: float | None = None
    graph_depth: int | None = None
    graph_reasons: list["GraphReasonResponse"] = Field(default_factory=list)


class GraphSourcePositionResponse(BaseModel):
    source_unit_id: str
    unit_index: int
    kind: UnitKind
    href: str | None = None
    cfi: str | None = None


class GraphEvidenceResponse(BaseModel):
    book_id: str
    content_version_id: str
    source_chunk_id: str
    section_index: int
    chunk_index: int
    source_unit_ids: list[str]
    source_positions: list[GraphSourcePositionResponse]
    assertedness: Literal["explicit", "inferred"]
    extractor_name: str
    extractor_version: str


class GraphReasonResponse(BaseModel):
    fact_id: str
    fact_kind: Literal["entity", "event", "relationship"]
    label: str
    assertedness: Literal["explicit", "inferred"]
    depth: int = Field(ge=0)
    connection: str
    evidence: list[GraphEvidenceResponse]


class RetrievalQueryResponse(BaseModel):
    retrieval_mode: Literal["keyword", "semantic", "hybrid", "hybrid_graph", "graph"]
    results: list[RetrievalResultResponse]


class AnswerCitationResponse(BaseModel):
    passage_id: str
    book_id: str
    content_version_id: str
    section_index: int
    chunk_index: int
    source_unit_ids: list[str]
    source_positions: list[SourcePositionResponse]
    href: str | None = None
    start_cfi: str | None = None
    end_cfi: str | None = None
    cfi_status: Literal["complete", "partial", "unavailable", "unreliable"]
    excerpt: str


class AnswerSourceResponse(BaseModel):
    passage_id: str
    score: float
    retrieval_method: Literal["keyword", "semantic", "hybrid", "graph"]
    keyword_rank: int | None = None
    keyword_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    graph_score: float | None = None
    graph_depth: int | None = None


class GroundedAnswerResponse(BaseModel):
    status: Literal["answered", "insufficient_evidence", "abstained"]
    answer: str
    uncertainty: str | None = None
    retrieval_mode: Literal["keyword", "semantic", "hybrid", "hybrid_graph", "graph"]
    citations: list[AnswerCitationResponse]
    sources: list[AnswerSourceResponse]
    graph_fact_ids: list[str]
    selected_passage_ids: list[str]


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: list[dict[str, object]] | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
