"""Bounded, evidence-backed expansion over an eligible story graph.

Text-retrieval seeds stay first in their component order. Additional passages
are ordered by shallowest eligible graph distance, then source position and
chunk ID. The graph score is a reciprocal-depth signal, not narrative weight.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from .chunking import BookChunk
from .keyword import tokenize
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
class GraphRetrievalOptions:
    max_depth: int = 2
    max_facts: int = 12
    max_additional_chunks: int = 8
    max_facts_per_entity: int = 3

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_depth", self.max_depth, 4),
            ("max_facts", self.max_facts, 100),
            ("max_additional_chunks", self.max_additional_chunks, 100),
            ("max_facts_per_entity", self.max_facts_per_entity, 10),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0 or value > maximum:
                raise ValueError(f"{name} must be between 0 and {maximum}")


DEFAULT_GRAPH_OPTIONS = GraphRetrievalOptions()


@dataclass(frozen=True, slots=True)
class _Fact:
    id: str
    kind: Literal["entity", "event", "relationship"]
    label: str
    assertedness: Literal["explicit", "inferred"]
    evidence: tuple[GraphEvidence, ...]
    entity_ids: tuple[str, ...]


def expand_hybrid_results(
    *,
    query: str,
    seed_results: Sequence[RetrievalResult],
    graphs: Mapping[str, StoryGraph],
    chunks_by_book: Mapping[str, Sequence[BookChunk]],
    boundaries: Mapping[str, ReadingBoundary],
    limit: int,
    options: GraphRetrievalOptions = DEFAULT_GRAPH_OPTIONS,
) -> list[RetrievalResult]:
    """Expand hybrid seeds using only fully eligible facts and canonical chunks."""
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    if limit == 0:
        return []

    seeds: dict[str, RetrievalResult] = {}
    for result in seed_results:
        if result.retrieval_method not in ("keyword", "semantic", "hybrid"):
            raise ValueError("graph expansion requires text-retrieval seed results")
        chunk = result.chunk
        boundary = boundaries.get(chunk.book_id)
        if boundary is None or not is_chunk_eligible(chunk, boundary):
            continue
        previous = seeds.get(chunk.id)
        if previous is not None and previous.chunk != chunk:
            raise ValueError(f"conflicting provenance for chunk {chunk.id!r}")
        seeds.setdefault(chunk.id, result)

    if not seeds:
        return []

    canonical: dict[str, dict[str, BookChunk]] = {}
    for book_id, chunks in chunks_by_book.items():
        canonical[book_id] = {chunk.id: chunk for chunk in chunks}

    query_terms = set(tokenize(query))
    reasons_by_chunk: dict[str, dict[str, GraphRetrievalReason]] = defaultdict(dict)
    best_depth: dict[str, int] = {}
    seed_ids_by_book: dict[str, set[str]] = defaultdict(set)
    for result in seeds.values():
        seed_ids_by_book[result.chunk.book_id].add(result.chunk.id)

    remaining_fact_budget = options.max_facts
    for book_id in sorted(seed_ids_by_book):
        if remaining_fact_budget <= 0:
            break
        graph = graphs.get(book_id)
        boundary = boundaries.get(book_id)
        if graph is None or boundary is None:
            continue
        # This API filters complete fact evidence and endpoints before traversal.
        entities, events, relationships = graph.eligible_facts(boundary)
        facts = _make_facts(entities, events, relationships)
        chunks = canonical.get(book_id, {})
        valid_facts = [
            fact
            for fact in facts
            if fact.evidence
            and all(
                evidence.is_eligible(boundary)
                and _evidence_resolves(evidence, chunks, boundary)
                for evidence in fact.evidence
            )
        ]
        valid_facts.sort(key=_fact_order)
        adjacency: dict[str, list[_Fact]] = defaultdict(list)
        for fact in valid_facts:
            for entity_id in fact.entity_ids:
                adjacency[entity_id].append(fact)
        for entity_id in adjacency:
            adjacency[entity_id].sort(key=_fact_order)

        seed_chunk_ids = seed_ids_by_book[book_id]
        start_entities: set[str] = set()
        for entity in entities:
            evidence_ids = {item.source_chunk_id for item in entity.evidence}
            eligible_aliases = tuple(
                alias.name
                for alias in entity.aliases
                if alias.evidence
                and all(
                    evidence.is_eligible(boundary)
                    and _evidence_resolves(evidence, chunks, boundary)
                    for evidence in alias.evidence
                )
            )
            if evidence_ids & seed_chunk_ids or _text_matches(
                query_terms, (entity.name, *eligible_aliases)
            ):
                start_entities.add(entity.id)
        # Facts directly supported by a retrieved passage can start expansion,
        # including events and relationships whose evidence is the seed itself.
        direct_fact_ids = {
            fact.id
            for fact in valid_facts
            if {e.source_chunk_id for e in fact.evidence} & seed_chunk_ids
            or _text_matches(query_terms, (fact.label,))
        }
        direct_fact_ids = set(
            fact.id
            for fact in sorted(
                (fact for fact in valid_facts if fact.id in direct_fact_ids),
                key=_fact_order,
            )[:remaining_fact_budget]
        )
        for fact in valid_facts:
            if fact.id in direct_fact_ids:
                start_entities.update(fact.entity_ids)
                _record_fact(
                    fact,
                    0,
                    "seed passage or query match",
                    chunks,
                    boundary,
                    reasons_by_chunk,
                    best_depth,
                )

        queue = deque((entity_id, 0) for entity_id in sorted(start_entities))
        visited_entities = set(start_entities)
        visited_facts = set(direct_fact_ids)
        facts_expanded = len(direct_fact_ids)
        while queue and facts_expanded < remaining_fact_budget:
            entity_id, depth = queue.popleft()
            if depth >= options.max_depth:
                continue
            used_for_entity = 0
            for fact in adjacency.get(entity_id, ()):
                if fact.id in visited_facts:
                    continue
                if used_for_entity >= options.max_facts_per_entity:
                    break
                if facts_expanded >= remaining_fact_budget:
                    break
                visited_facts.add(fact.id)
                facts_expanded += 1
                used_for_entity += 1
                fact_depth = depth + 1
                _record_fact(
                    fact,
                    fact_depth,
                    f"connected through entity {entity_id}",
                    chunks,
                    boundary,
                    reasons_by_chunk,
                    best_depth,
                )
                for neighbor in fact.entity_ids:
                    if neighbor not in visited_entities:
                        visited_entities.add(neighbor)
                        queue.append((neighbor, fact_depth))
        remaining_fact_budget -= facts_expanded

    # Preserve all hybrid seeds, with graph explanations attached when relevant.
    output: list[RetrievalResult] = []
    for seed in seeds.values():
        reasons = tuple(
            sorted(reasons_by_chunk.get(seed.chunk.id, {}).values(), key=_reason_order)
        )
        output.append(
            RetrievalResult(
                chunk=seed.chunk,
                score=seed.score,
                retrieval_method=seed.retrieval_method,
                matched_terms=seed.matched_terms,
                keyword_rank=seed.keyword_rank,
                keyword_score=seed.keyword_score,
                semantic_rank=seed.semantic_rank,
                semantic_score=seed.semantic_score,
                graph_score=(
                    1.0 / (1 + best_depth[seed.chunk.id])
                    if seed.chunk.id in best_depth
                    else None
                ),
                graph_depth=best_depth.get(seed.chunk.id),
                graph_reasons=reasons,
            )
        )
    output.sort(key=lambda item: _seed_order(item, seeds))

    additions: list[RetrievalResult] = []
    for book_id in sorted(canonical):
        boundary = boundaries.get(book_id)
        if boundary is None:
            continue
        for chunk_id, depth in best_depth.items():
            chunk = canonical[book_id].get(chunk_id)
            if chunk is None or chunk_id in seeds or not is_chunk_eligible(chunk, boundary):
                continue
            reasons = tuple(
                sorted(reasons_by_chunk[chunk_id].values(), key=_reason_order)
            )
            additions.append(
                RetrievalResult(
                    chunk=chunk,
                    score=0.0,
                    retrieval_method="graph",
                    graph_score=1.0 / (1 + depth),
                    graph_depth=depth,
                    graph_reasons=reasons,
                )
            )
    additions.sort(key=_expanded_order)
    additions = additions[: options.max_additional_chunks]
    # `limit` caps the initial hybrid candidate set. Graph additions have their
    # own explicit cap so they can supplement a full hybrid result list.
    return output + additions


def _make_facts(
    entities: Sequence[StoryEntity],
    events: Sequence[StoryEvent],
    relationships: Sequence[StoryRelationship],
) -> list[_Fact]:
    facts = [
        _Fact(item.id, "entity", item.name, item.assertedness, item.evidence, (item.id,))
        for item in entities
    ]
    facts.extend(
        _Fact(
            item.id,
            "event",
            item.name,
            item.assertedness,
            item.evidence,
            item.participant_entity_ids,
        )
        for item in events
    )
    facts.extend(
        _Fact(
            item.id,
            "relationship",
            item.relationship_type,
            item.assertedness,
            item.evidence,
            (item.subject_entity_id, item.object_entity_id),
        )
        for item in relationships
    )
    return facts


def _evidence_resolves(
    evidence: GraphEvidence,
    chunks: Mapping[str, BookChunk],
    boundary: ReadingBoundary,
) -> bool:
    chunk = chunks.get(evidence.source_chunk_id)
    if (
        chunk is None
        or chunk.book_id != evidence.book_id
        or chunk.content_version_id != evidence.content_version_id
    ):
        return False
    positions = {position.source_unit_id for position in chunk.source_positions}
    return (
        is_chunk_eligible(chunk, boundary)
        and bool(evidence.source_unit_ids)
        and set(evidence.source_unit_ids).issubset(positions)
    )


def _text_matches(query_terms: set[str], candidates: Sequence[str]) -> bool:
    return bool(query_terms & {term for text in candidates for term in tokenize(text)})


def _record_fact(
    fact: _Fact,
    depth: int,
    connection: str,
    chunks: Mapping[str, BookChunk],
    boundary: ReadingBoundary,
    reasons_by_chunk: dict[str, dict[str, GraphRetrievalReason]],
    best_depth: dict[str, int],
) -> None:
    reason = GraphRetrievalReason(
        fact_id=fact.id,
        fact_kind=fact.kind,
        label=fact.label,
        assertedness=fact.assertedness,
        depth=depth,
        connection=connection,
        evidence=fact.evidence,
    )
    for evidence in fact.evidence:
        chunk = chunks.get(evidence.source_chunk_id)
        if (
            chunk is None
            or not evidence.is_eligible(boundary)
            or not is_chunk_eligible(chunk, boundary)
        ):
            continue
        reasons_by_chunk[chunk.id][fact.id] = reason
        best_depth[chunk.id] = min(depth, best_depth.get(chunk.id, depth))


def _fact_order(fact: _Fact) -> tuple[object, ...]:
    evidence = min(
        fact.evidence,
        key=lambda item: (item.section_index, item.chunk_index, item.source_chunk_id),
    )
    return evidence.section_index, evidence.chunk_index, fact.kind, fact.id


def _reason_order(reason: GraphRetrievalReason) -> tuple[object, ...]:
    return reason.depth, reason.fact_kind, reason.label.casefold(), reason.fact_id


def _seed_order(
    result: RetrievalResult, seeds: Mapping[str, RetrievalResult]
) -> tuple[object, ...]:
    original = list(seeds).index(result.chunk.id)
    return original, result.chunk.section_index, result.chunk.chunk_index, result.chunk.id


def _expanded_order(result: RetrievalResult) -> tuple[object, ...]:
    chunk = result.chunk
    return (
        result.graph_depth if result.graph_depth is not None else 999,
        chunk.section_index,
        chunk.chunk_index,
        chunk.book_id,
        chunk.content_version_id,
        chunk.id,
    )
