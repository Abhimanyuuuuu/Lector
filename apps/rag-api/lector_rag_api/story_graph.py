"""Evidence-backed, process-local story graph foundations.

Entity resolution is deliberately conservative: only exact NFKC/casefolded
canonical names with the same entity type are joined. Aliases are retained as
evidence-backed labels, but are never used to merge nodes automatically.
Every fact requires source evidence, and a fact is boundary-eligible only when
all of its supporting evidence is eligible.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Callable, Literal, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .chunking import BookChunk, ChunkSourcePosition
from .retrieval_types import ReadingBoundary

EntityKind = Literal["character", "location", "organization", "object", "other"]
AssertionKind = Literal["explicit", "inferred"]
IdentityKind = Literal["resolved", "uncertain"]


class EvidenceCandidate(BaseModel):
    """Untrusted extractor reference to source material supplied in its input."""

    model_config = ConfigDict(extra="forbid")

    source_chunk_id: str = Field(min_length=1)
    source_unit_ids: list[str] = Field(min_length=1)

    @field_validator("source_chunk_id", mode="before")
    @classmethod
    def strip_chunk_id(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("source_unit_ids")
    @classmethod
    def validate_unique_source_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() for value in values):
            raise ValueError("source_unit_ids must contain non-empty IDs")
        if len(values) != len(set(values)):
            raise ValueError("source_unit_ids must be unique")
        return values


class EntityCandidate(BaseModel):
    """A candidate entity. `ref` is only a batch-local link, not a graph ID."""

    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=512)
    kind: EntityKind
    identity_status: IdentityKind = "uncertain"
    aliases: list[str] = Field(default_factory=list)
    description: str | None = None
    assertedness: AssertionKind
    evidence: list[EvidenceCandidate] = Field(min_length=1)

    @field_validator("ref", "name", mode="before")
    @classmethod
    def strip_identity_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("aliases")
    @classmethod
    def validate_aliases(cls, aliases: list[str]) -> list[str]:
        if any(not alias.strip() for alias in aliases):
            raise ValueError("aliases must contain non-empty names")
        return aliases


class EventCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ref: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=512)
    description: str | None = None
    participant_refs: list[str] = Field(default_factory=list)
    assertedness: AssertionKind
    evidence: list[EvidenceCandidate] = Field(min_length=1)

    @field_validator("ref", "name", mode="before")
    @classmethod
    def strip_identity_text(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value


class RelationshipCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject_ref: str = Field(min_length=1)
    object_ref: str = Field(min_length=1)
    relationship_type: str = Field(min_length=1, max_length=128)
    description: str | None = None
    temporal_metadata: dict[str, str] = Field(default_factory=dict)
    assertedness: AssertionKind
    evidence: list[EvidenceCandidate] = Field(min_length=1)


class StoryExtraction(BaseModel):
    """Validated schema returned by a story extractor implementation."""

    model_config = ConfigDict(extra="forbid")

    entities: list[EntityCandidate] = Field(default_factory=list)
    events: list[EventCandidate] = Field(default_factory=list)
    relationships: list[RelationshipCandidate] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_batch_refs(self) -> "StoryExtraction":
        entity_refs = [entity.ref for entity in self.entities]
        event_refs = [event.ref for event in self.events]
        if len(entity_refs) != len(set(entity_refs)):
            raise ValueError("entity refs must be unique within an extraction")
        if len(event_refs) != len(set(event_refs)):
            raise ValueError("event refs must be unique within an extraction")
        known = set(entity_refs)
        for event in self.events:
            if len(event.participant_refs) != len(set(event.participant_refs)):
                raise ValueError("event participant refs must be unique")
            unknown = set(event.participant_refs) - known
            if unknown:
                raise ValueError(f"event refers to unknown entity refs: {sorted(unknown)}")
        for relation in self.relationships:
            if relation.subject_ref not in known or relation.object_ref not in known:
                raise ValueError("relationship refers to an unknown entity ref")
            if not relation.relationship_type.strip():
                raise ValueError("relationship_type must be non-empty")
        return self


class StoryExtractor(Protocol):
    """Injectable extraction boundary; implementations receive eligible chunks only."""

    name: str
    version: str

    def extract(self, chunks: Sequence[BookChunk]) -> StoryExtraction | dict[str, object]:
        """Return structured candidates grounded in the supplied chunks."""


@dataclass(frozen=True, slots=True)
class GraphSourcePosition:
    source_unit_id: str
    unit_index: int
    kind: str
    href: str | None
    cfi: str | None


@dataclass(frozen=True, slots=True)
class GraphEvidence:
    book_id: str
    content_version_id: str
    source_chunk_id: str
    section_index: int
    chunk_index: int
    source_unit_ids: tuple[str, ...]
    source_positions: tuple[GraphSourcePosition, ...]
    assertedness: AssertionKind
    extractor_name: str
    extractor_version: str

    def is_eligible(self, boundary: ReadingBoundary) -> bool:
        if self.book_id != boundary.book_id or not self.source_positions:
            return False
        if self.section_index < boundary.section_index:
            return True
        if self.section_index > boundary.section_index:
            return False
        return all(
            position.unit_index <= boundary.unit_index
            for position in self.source_positions
        )


@dataclass(frozen=True, slots=True)
class StoryEntity:
    id: str
    book_id: str
    content_version_id: str
    name: str
    kind: EntityKind
    identity_status: IdentityKind
    aliases: tuple[StoryAlias, ...]
    description: str | None
    evidence: tuple[GraphEvidence, ...]

    @property
    def assertedness(self) -> AssertionKind:
        return _combined_assertedness(self.evidence)

    @property
    def alias_names(self) -> tuple[str, ...]:
        return tuple(alias.name for alias in self.aliases)


@dataclass(frozen=True, slots=True)
class StoryAlias:
    """Alternative entity label with its own supporting source evidence."""

    name: str
    evidence: tuple[GraphEvidence, ...]

    @property
    def assertedness(self) -> AssertionKind:
        return _combined_assertedness(self.evidence)


@dataclass(frozen=True, slots=True)
class StoryEvent:
    id: str
    book_id: str
    content_version_id: str
    name: str
    description: str | None
    participant_entity_ids: tuple[str, ...]
    evidence: tuple[GraphEvidence, ...]

    @property
    def assertedness(self) -> AssertionKind:
        return _combined_assertedness(self.evidence)


@dataclass(frozen=True, slots=True)
class StoryRelationship:
    id: str
    book_id: str
    content_version_id: str
    subject_entity_id: str
    object_entity_id: str
    relationship_type: str
    description: str | None
    temporal_metadata: tuple[tuple[str, str], ...]
    evidence: tuple[GraphEvidence, ...]

    @property
    def assertedness(self) -> AssertionKind:
        return _combined_assertedness(self.evidence)


@dataclass(frozen=True, slots=True)
class StoryGraph:
    book_id: str
    content_version_id: str
    entities: tuple[StoryEntity, ...] = ()
    events: tuple[StoryEvent, ...] = ()
    relationships: tuple[StoryRelationship, ...] = ()

    def eligible_facts(
        self, boundary: ReadingBoundary
    ) -> tuple[tuple[StoryEntity, ...], tuple[StoryEvent, ...], tuple[StoryRelationship, ...]]:
        """Return facts only when all supporting evidence is within the boundary.

        This is intentionally conservative: a fact with even one supporting
        passage beyond the boundary is withheld. Event participants and
        relationship endpoints must themselves be eligible as well.
        """
        if boundary.book_id != self.book_id:
            raise ValueError("reading boundary does not match graph book")
        eligible_entities = tuple(
            entity for entity in self.entities if _all_evidence_eligible(entity.evidence, boundary)
        )
        eligible_entity_ids = {entity.id for entity in eligible_entities}
        eligible_events = tuple(
            event
            for event in self.events
            if _all_evidence_eligible(event.evidence, boundary)
            and set(event.participant_entity_ids).issubset(eligible_entity_ids)
        )
        eligible_relationships = tuple(
            relation
            for relation in self.relationships
            if _all_evidence_eligible(relation.evidence, boundary)
            and relation.subject_entity_id in eligible_entity_ids
            and relation.object_entity_id in eligible_entity_ids
        )
        return eligible_entities, eligible_events, eligible_relationships


class InvalidGraphEvidenceError(ValueError):
    pass


class GraphStore:
    """Process-local graph snapshots keyed by book, replacing on version change."""

    def __init__(self) -> None:
        self._books: dict[str, StoryGraph] = {}

    def get_book(self, book_id: str) -> StoryGraph | None:
        return self._books.get(book_id)

    def remove_book(self, book_id: str) -> None:
        self._books.pop(book_id, None)

    def replace_book(self, book_id: str, content_version_id: str) -> None:
        current = self._books.get(book_id)
        if current is None or current.content_version_id != content_version_id:
            self._books[book_id] = StoryGraph(book_id, content_version_id)

    def commit(self, graph: StoryGraph) -> None:
        self._books[graph.book_id] = graph

    def prepare_update(
        self,
        *,
        book_id: str,
        content_version_id: str,
        chunks: Sequence[BookChunk],
        extraction: StoryExtraction | dict[str, object],
        extractor_name: str,
        extractor_version: str,
    ) -> StoryGraph:
        """Validate all references and build a new snapshot without mutation."""
        if not isinstance(extractor_name, str) or not extractor_name.strip():
            raise ValueError("extractor name must be non-empty")
        if not isinstance(extractor_version, str) or not extractor_version.strip():
            raise ValueError("extractor version must be non-empty")
        batch = StoryExtraction.model_validate(extraction)
        chunk_by_id = {chunk.id: chunk for chunk in chunks}
        if len(chunk_by_id) != len(chunks):
            raise ValueError("input chunks contain duplicate IDs")
        if any(
            chunk.book_id != book_id or chunk.content_version_id != content_version_id
            for chunk in chunks
        ):
            raise ValueError("input chunks do not match the graph book/version")
        old = self._books.get(book_id)
        if old is not None and old.content_version_id == content_version_id:
            entities = {item.id: item for item in old.entities}
            events = {item.id: item for item in old.events}
            relationships = {item.id: item for item in old.relationships}
        else:
            entities, events, relationships = {}, {}, {}

        evidence_cache: dict[
            tuple[tuple[tuple[str, tuple[str, ...]], ...], AssertionKind],
            tuple[GraphEvidence, ...],
        ] = {}

        def evidence_for(
            refs: Sequence[EvidenceCandidate], assertedness: AssertionKind
        ) -> tuple[GraphEvidence, ...]:
            cache_key = (
                tuple(
                    (ref.source_chunk_id, tuple(ref.source_unit_ids)) for ref in refs
                ),
                assertedness,
            )
            cached = evidence_cache.get(cache_key)
            if cached is not None:
                return cached
            items: list[GraphEvidence] = []
            for ref in refs:
                chunk = chunk_by_id.get(ref.source_chunk_id)
                if chunk is None:
                    raise InvalidGraphEvidenceError(
                        f"extractor referenced unknown chunk {ref.source_chunk_id!r}"
                    )
                positions_by_id = {
                    position.source_unit_id: position for position in chunk.source_positions
                }
                if any(source_id not in positions_by_id for source_id in ref.source_unit_ids):
                    raise InvalidGraphEvidenceError(
                        f"extractor referenced a source unit outside chunk {chunk.id!r}"
                    )
                selected_positions = tuple(
                    _graph_position(positions_by_id[source_id])
                    for source_id in ref.source_unit_ids
                )
                selected_positions = tuple(
                    sorted(selected_positions, key=lambda position: position.unit_index)
                )
                items.append(
                    GraphEvidence(
                        book_id=book_id,
                        content_version_id=content_version_id,
                        source_chunk_id=chunk.id,
                        section_index=chunk.section_index,
                        chunk_index=chunk.chunk_index,
                        source_unit_ids=tuple(
                            position.source_unit_id for position in selected_positions
                        ),
                        source_positions=selected_positions,
                        assertedness=assertedness,
                        extractor_name=extractor_name,
                        extractor_version=extractor_version,
                    )
                )
            result = _merge_evidence((), tuple(items))
            evidence_cache[cache_key] = result
            return result

        entity_ids: dict[str, str] = {}
        entity_candidates_by_id: dict[str, list[EntityCandidate]] = {}
        for candidate in batch.entities:
            normalized = _normalized(candidate.name)
            identity_parts = [candidate.kind, normalized]
            if candidate.identity_status == "uncertain":
                initial_evidence = evidence_for(
                    candidate.evidence, candidate.assertedness
                )
                first_evidence = initial_evidence[0]
                identity_parts.append(
                    json.dumps(
                        (
                            first_evidence.section_index,
                            first_evidence.chunk_index,
                            first_evidence.source_unit_ids,
                        ),
                        ensure_ascii=False,
                    )
                )
            entity_id = _stable_id(
                "entity", book_id, content_version_id, *identity_parts
            )
            entity_ids[candidate.ref] = entity_id
            entity_candidates_by_id.setdefault(entity_id, []).append(candidate)

        for entity_id, candidates in entity_candidates_by_id.items():
            first = candidates[0]
            previous = entities.get(entity_id)
            aliases = _merge_aliases(
                previous.aliases if previous else (),
                candidates,
                evidence_for,
            )
            support = _merge_evidence(
                previous.evidence if previous else (),
                *(
                    evidence_for(candidate.evidence, candidate.assertedness)
                    for candidate in candidates
                ),
            )
            entities[entity_id] = StoryEntity(
                id=entity_id,
                book_id=book_id,
                content_version_id=content_version_id,
                name=previous.name if previous else first.name,
                kind=first.kind,
                identity_status=first.identity_status,
                aliases=aliases,
                description=(
                    previous.description
                    if previous and previous.description
                    else next((candidate.description for candidate in candidates if candidate.description), None)
                ),
                evidence=support,
            )

        for candidate in batch.events:
            participant_ids = tuple(
                sorted(entity_ids[ref] for ref in candidate.participant_refs)
            )
            event_evidence = evidence_for(candidate.evidence, candidate.assertedness)
            evidence_origin = tuple(
                (item.section_index, item.chunk_index, item.source_unit_ids)
                for item in event_evidence
            )
            event_id = _stable_id(
                "event",
                book_id,
                content_version_id,
                _normalized(candidate.name),
                json.dumps(participant_ids),
                json.dumps(evidence_origin, ensure_ascii=False),
            )
            previous = events.get(event_id)
            events[event_id] = StoryEvent(
                id=event_id,
                book_id=book_id,
                content_version_id=content_version_id,
                name=previous.name if previous else candidate.name,
                description=(previous.description if previous and previous.description else candidate.description),
                participant_entity_ids=participant_ids,
                evidence=_merge_evidence(
                    previous.evidence if previous else (), event_evidence
                ),
            )

        for candidate in batch.relationships:
            subject_id = entity_ids[candidate.subject_ref]
            object_id = entity_ids[candidate.object_ref]
            normalized_type = _normalized(candidate.relationship_type)
            temporal = tuple(sorted(candidate.temporal_metadata.items()))
            relation_id = _stable_id(
                "relationship",
                book_id,
                content_version_id,
                subject_id,
                normalized_type,
                object_id,
                json.dumps(temporal, ensure_ascii=False),
            )
            previous = relationships.get(relation_id)
            relation_evidence = evidence_for(candidate.evidence, candidate.assertedness)
            relationships[relation_id] = StoryRelationship(
                id=relation_id,
                book_id=book_id,
                content_version_id=content_version_id,
                subject_entity_id=subject_id,
                object_entity_id=object_id,
                relationship_type=(
                    previous.relationship_type if previous else candidate.relationship_type
                ),
                description=(previous.description if previous and previous.description else candidate.description),
                temporal_metadata=previous.temporal_metadata if previous else temporal,
                evidence=_merge_evidence(
                    previous.evidence if previous else (), relation_evidence
                ),
            )

        return StoryGraph(
            book_id=book_id,
            content_version_id=content_version_id,
            entities=tuple(sorted(entities.values(), key=lambda item: item.id)),
            events=tuple(sorted(events.values(), key=lambda item: item.id)),
            relationships=tuple(sorted(relationships.values(), key=lambda item: item.id)),
        )


def _graph_position(position: ChunkSourcePosition) -> GraphSourcePosition:
    return GraphSourcePosition(
        source_unit_id=position.source_unit_id,
        unit_index=position.unit_index,
        kind=position.kind,
        href=position.href,
        cfi=position.cfi,
    )


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def _stable_id(kind: str, *parts: str) -> str:
    encoded = json.dumps([kind, *parts], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"{kind}:{digest}"


def _merge_aliases(
    existing: Sequence[StoryAlias],
    candidates: Sequence[EntityCandidate],
    evidence_for: Callable[
        [Sequence[EvidenceCandidate], AssertionKind], tuple[GraphEvidence, ...]
    ],
) -> tuple[StoryAlias, ...]:
    aliases: dict[str, StoryAlias] = {
        _normalized(alias.name): alias for alias in existing
    }
    for candidate in candidates:
        candidate_evidence = evidence_for(candidate.evidence, candidate.assertedness)
        for name in candidate.aliases:
            normalized = _normalized(name)
            if not normalized:
                continue
            previous = aliases.get(normalized)
            aliases[normalized] = StoryAlias(
                name=previous.name if previous else name.strip(),
                evidence=_merge_evidence(
                    previous.evidence if previous else (), candidate_evidence
                ),
            )
    return tuple(aliases[key] for key in sorted(aliases))


def _evidence_key(evidence: GraphEvidence) -> tuple[object, ...]:
    return (
        evidence.book_id,
        evidence.content_version_id,
        evidence.source_chunk_id,
        evidence.source_unit_ids,
        evidence.assertedness,
        evidence.extractor_name,
        evidence.extractor_version,
    )


def _merge_evidence(
    *groups: Sequence[GraphEvidence],
) -> tuple[GraphEvidence, ...]:
    merged = {_evidence_key(item): item for group in groups for item in group}
    return tuple(
        sorted(
            merged.values(),
            key=lambda item: (
                item.section_index,
                item.chunk_index,
                item.source_chunk_id,
                item.source_unit_ids,
                item.assertedness,
                item.extractor_name,
                item.extractor_version,
            ),
        )
    )


def _all_evidence_eligible(
    evidence: Sequence[GraphEvidence], boundary: ReadingBoundary
) -> bool:
    return bool(evidence) and all(item.is_eligible(boundary) for item in evidence)


def _combined_assertedness(evidence: Sequence[GraphEvidence]) -> AssertionKind:
    return "inferred" if any(item.assertedness == "inferred" for item in evidence) else "explicit"
