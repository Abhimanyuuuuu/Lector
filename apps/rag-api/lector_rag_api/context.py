"""Deterministic assembly of boundary-eligible answer evidence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence

from .chunking import BookChunk
from .retrieval_types import (
    GraphRetrievalReason,
    ReadingBoundary,
    RetrievalResult,
    is_chunk_eligible,
)
from .story_graph import (
    GraphEvidence,
    StoryEntity,
    StoryEvent,
    StoryGraph,
    StoryRelationship,
)


@dataclass(frozen=True, slots=True)
class ContextOptions:
    max_chars: int = 12_000
    max_passages: int = 8
    max_graph_facts: int = 8

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("max_chars", self.max_chars, 1, 100_000),
            ("max_passages", self.max_passages, 1, 100),
            ("max_graph_facts", self.max_graph_facts, 0, 100),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")


DEFAULT_CONTEXT_OPTIONS = ContextOptions()


@dataclass(frozen=True, slots=True)
class ContextPassage:
    passage_id: str
    chunk: BookChunk
    score: float
    retrieval_method: str
    keyword_rank: int | None
    keyword_score: float | None
    semantic_rank: int | None
    semantic_score: float | None
    graph_score: float | None
    graph_depth: int | None


@dataclass(frozen=True, slots=True)
class ContextGraphFact:
    fact_id: str
    fact_kind: str
    label: str
    assertedness: str
    depth: int
    evidence_passage_ids: tuple[str, ...]
    source_unit_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AssembledContext:
    passages: tuple[ContextPassage, ...]
    graph_facts: tuple[ContextGraphFact, ...]
    prompt_json: str
    character_count: int
    sufficient: bool


def assemble_context(
    *,
    results: Sequence[RetrievalResult],
    chunks_by_book: Mapping[str, Sequence[BookChunk]],
    graphs: Mapping[str, StoryGraph],
    boundaries: Mapping[str, ReadingBoundary],
    options: ContextOptions = DEFAULT_CONTEXT_OPTIONS,
) -> AssembledContext:
    """Select only canonical, complete, boundary-eligible retrieval evidence.

    The character budget is measured on the compact JSON actually sent to the
    provider (Python Unicode code points). Chunks are indivisible: an oversized
    candidate is skipped rather than excerpted or assigned a false location.
    """
    canonical = {
        book_id: {chunk.id: chunk for chunk in chunks}
        for book_id, chunks in chunks_by_book.items()
    }
    _validate_boundaries(boundaries, canonical)
    required_books = {result.chunk.book_id for result in results}
    missing_boundaries = required_books - set(boundaries)
    if missing_boundaries:
        raise ValueError(
            "a confirmed reading boundary is required for every result book"
        )
    passages: list[ContextPassage] = []
    seen_chunks: set[tuple[str, str]] = set()
    payload: dict[str, object] = {"passages": [], "graph_facts": []}

    for result in results:
        chunk = result.chunk
        key = (chunk.book_id, chunk.id)
        if key in seen_chunks:
            continue
        boundary = boundaries.get(chunk.book_id)
        stored_chunk = canonical.get(chunk.book_id, {}).get(chunk.id)
        if (
            boundary is None
            or stored_chunk is None
            or stored_chunk != chunk
            or stored_chunk.content_version_id != chunk.content_version_id
            or not is_chunk_eligible(stored_chunk, boundary)
            or not chunk.text.strip()
        ):
            continue
        if len(passages) >= options.max_passages:
            break
        passage = ContextPassage(
            passage_id=chunk.id,
            chunk=stored_chunk,
            score=result.score,
            retrieval_method=result.retrieval_method,
            keyword_rank=result.keyword_rank,
            keyword_score=result.keyword_score,
            semantic_rank=result.semantic_rank,
            semantic_score=result.semantic_score,
            graph_score=result.graph_score,
            graph_depth=result.graph_depth,
        )
        candidate_passages = [*passages, passage]
        candidate_payload = _prompt_payload(candidate_passages, ())
        serialized = _serialize(candidate_payload)
        if len(serialized) > options.max_chars:
            continue
        passages.append(passage)
        seen_chunks.add(key)
        payload = candidate_payload

    passage_ids = {passage.passage_id for passage in passages}
    passage_by_id = {passage.passage_id: passage for passage in passages}
    eligible_fact_maps = _eligible_fact_maps(graphs, boundaries, canonical)
    graph_facts: list[ContextGraphFact] = []
    seen_facts: set[str] = set()

    for result in results:
        for reason in result.graph_reasons:
            if reason.fact_id in seen_facts or len(graph_facts) >= options.max_graph_facts:
                continue
            book_id = result.chunk.book_id
            fact = eligible_fact_maps.get(book_id, {}).get(reason.fact_id)
            if fact is None:
                continue
            evidence = _fact_evidence(fact)
            # The complete supporting evidence must be in the final context.
            evidence_ids = tuple(item.source_chunk_id for item in evidence)
            if not evidence_ids or not set(evidence_ids).issubset(passage_ids):
                continue
            if not _same_evidence(evidence, reason.evidence):
                continue
            source_ids: list[str] = []
            for item in evidence:
                boundary = boundaries.get(item.book_id)
                chunk = canonical.get(item.book_id, {}).get(item.source_chunk_id)
                if (
                    boundary is None
                    or chunk is None
                    or not item.is_eligible(boundary)
                    or not is_chunk_eligible(chunk, boundary)
                    or chunk.id not in passage_by_id
                    or chunk.content_version_id != item.content_version_id
                    or not set(item.source_unit_ids).issubset(chunk.source_unit_ids)
                ):
                    source_ids = []
                    break
                source_ids.extend(item.source_unit_ids)
            if not source_ids:
                continue
            candidate = ContextGraphFact(
                fact_id=reason.fact_id,
                fact_kind=reason.fact_kind,
                label=reason.label,
                assertedness=reason.assertedness,
                depth=reason.depth,
                evidence_passage_ids=tuple(dict.fromkeys(evidence_ids)),
                source_unit_ids=tuple(dict.fromkeys(source_ids)),
            )
            candidate_facts = [*graph_facts, candidate]
            candidate_payload = _prompt_payload(passages, candidate_facts)
            serialized = _serialize(candidate_payload)
            if len(serialized) > options.max_chars:
                continue
            graph_facts.append(candidate)
            seen_facts.add(reason.fact_id)
            payload = candidate_payload

    prompt_json = _serialize(payload)
    return AssembledContext(
        passages=tuple(passages),
        graph_facts=tuple(graph_facts),
        prompt_json=prompt_json,
        character_count=len(prompt_json),
        sufficient=bool(passages),
    )


def _prompt_payload(
    passages: Sequence[ContextPassage], graph_facts: Sequence[ContextGraphFact]
) -> dict[str, object]:
    return {
        "passages": [
            {
                "passage_id": passage.passage_id,
                "source": {
                    "section_index": passage.chunk.section_index,
                    "chunk_index": passage.chunk.chunk_index,
                    "href": passage.chunk.href,
                },
                "text": passage.chunk.text,
            }
            for passage in passages
        ],
        "graph_facts": [
            {
                "fact_id": fact.fact_id,
                "kind": fact.fact_kind,
                "label": fact.label,
                "assertedness": fact.assertedness,
                "evidence_passage_ids": list(fact.evidence_passage_ids),
            }
            for fact in graph_facts
        ],
    }


def _serialize(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _validate_boundaries(
    boundaries: Mapping[str, ReadingBoundary],
    canonical: Mapping[str, Mapping[str, BookChunk]],
) -> None:
    for book_id, boundary in boundaries.items():
        if boundary.book_id != book_id:
            raise ValueError("reading boundary book_id does not match its scope")
        positions = {
            (chunk.section_index, position.unit_index, position.source_unit_id)
            for chunk in canonical.get(book_id, {}).values()
            for position in chunk.source_positions
        }
        if (
            boundary.section_index < 0
            or boundary.unit_index < 0
            or not boundary.source_unit_id
            or (
                boundary.section_index,
                boundary.unit_index,
                boundary.source_unit_id,
            )
            not in positions
        ):
            raise ValueError("confirmed reading boundary is invalid for indexed content")


def _eligible_fact_maps(
    graphs: Mapping[str, StoryGraph],
    boundaries: Mapping[str, ReadingBoundary],
    canonical: Mapping[str, Mapping[str, BookChunk]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for book_id, graph in graphs.items():
        boundary = boundaries.get(book_id)
        if boundary is None:
            continue
        entities, events, relationships = graph.eligible_facts(boundary)
        facts: dict[str, object] = {}
        for fact in (*entities, *events, *relationships):
            evidence = _fact_evidence(fact)
            if evidence and all(
                item.is_eligible(boundary)
                and _evidence_resolves(item, canonical.get(book_id, {}), boundary)
                for item in evidence
            ):
                facts[fact.id] = fact
        result[book_id] = facts
    return result


def _fact_evidence(fact: object) -> tuple[GraphEvidence, ...]:
    return getattr(fact, "evidence", ())


def _evidence_resolves(
    evidence: GraphEvidence,
    chunks: Mapping[str, BookChunk],
    boundary: ReadingBoundary,
) -> bool:
    chunk = chunks.get(evidence.source_chunk_id)
    return bool(
        chunk
        and chunk.book_id == evidence.book_id
        and chunk.content_version_id == evidence.content_version_id
        and is_chunk_eligible(chunk, boundary)
        and evidence.source_unit_ids
        and set(evidence.source_unit_ids).issubset(chunk.source_unit_ids)
    )


def _same_evidence(
    expected: Sequence[GraphEvidence], supplied: Sequence[GraphEvidence]
) -> bool:
    return tuple(_evidence_key(item) for item in expected) == tuple(
        _evidence_key(item) for item in supplied
    )


def _evidence_key(item: GraphEvidence) -> tuple[object, ...]:
    return (
        item.book_id,
        item.content_version_id,
        item.source_chunk_id,
        item.source_unit_ids,
        item.section_index,
        item.chunk_index,
        item.assertedness,
        item.extractor_name,
        item.extractor_version,
    )
