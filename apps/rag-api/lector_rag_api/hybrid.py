"""Reciprocal Rank Fusion for keyword and semantic retrieval candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .retrieval_types import RetrievalResult

DEFAULT_RRF_K = 60


class ConflictingChunkProvenanceError(ValueError):
    """A stable chunk ID refers to different source records across inputs."""


@dataclass(frozen=True, slots=True)
class HybridConfig:
    """RRF configuration. Weights scale each source's reciprocal-rank vote."""

    k: int = DEFAULT_RRF_K
    keyword_weight: float = 1.0
    semantic_weight: float = 1.0

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k <= 0:
            raise ValueError("RRF k must be a positive integer")
        for name, weight in (
            ("keyword_weight", self.keyword_weight),
            ("semantic_weight", self.semantic_weight),
        ):
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise TypeError(f"{name} must be a finite non-negative number")
            try:
                finite = math.isfinite(weight)
            except OverflowError:
                finite = False
            if not finite or weight < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.keyword_weight == 0 and self.semantic_weight == 0:
            raise ValueError("at least one retrieval source weight must be positive")


DEFAULT_HYBRID_CONFIG = HybridConfig()


def fuse_results(
    keyword_results: Sequence[RetrievalResult],
    semantic_results: Sequence[RetrievalResult],
    *,
    config: HybridConfig = DEFAULT_HYBRID_CONFIG,
    limit: int = 10,
) -> list[RetrievalResult]:
    """Fuse ranked lists without comparing their component scores.

    Input positions are one-based ranks. The default score is
    ``sum(weight / (60 + rank))`` across the two sources. Component rankings
    and scores are retained for inspection. Equal fusion scores sort by
    earliest component rank, then stable source location and chunk identity.
    Duplicate IDs within or across lists count once per source; differing
    chunk records for the same ID are rejected.
    """
    if not isinstance(config, HybridConfig):
        raise TypeError("config must be a HybridConfig")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("limit must be a non-negative integer")
    _validate_results(keyword_results, "keyword")
    _validate_results(semantic_results, "semantic")
    if limit == 0:
        return []

    records: dict[str, dict[str, object]] = {}
    _add_source(records, keyword_results, "keyword", config.keyword_weight)
    _add_source(records, semantic_results, "semantic", config.semantic_weight)

    fused: list[tuple[RetrievalResult, int, int]] = []
    for record in records.values():
        keyword_rank = record.get("keyword_rank")
        semantic_rank = record.get("semantic_rank")
        score = math.fsum(
            (
                config.keyword_weight / (config.k + keyword_rank)
                if keyword_rank is not None
                else 0.0,
                config.semantic_weight / (config.k + semantic_rank)
                if semantic_rank is not None
                else 0.0,
            )
        )
        chunk_result = record["result"]
        assert isinstance(chunk_result, RetrievalResult)
        result = RetrievalResult(
            chunk=chunk_result.chunk,
            score=score,
            retrieval_method="hybrid",
            matched_terms=tuple(record.get("matched_terms", ())),
            keyword_rank=keyword_rank,
            keyword_score=record.get("keyword_score"),
            semantic_rank=semantic_rank,
            semantic_score=record.get("semantic_score"),
        )
        first_rank = min(
            rank for rank in (keyword_rank, semantic_rank) if rank is not None
        )
        fused.append((result, first_rank, keyword_rank or math.inf))

    fused.sort(key=_fused_order)
    return [result for result, _, _ in fused[:limit]]


def _validate_results(results: Sequence[RetrievalResult], source: str) -> None:
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        raise TypeError(f"{source}_results must be a sequence")
    for result in results:
        if not isinstance(result, RetrievalResult):
            raise TypeError(f"{source}_results must contain RetrievalResult values")
        if result.retrieval_method != source:
            raise ValueError(f"{source}_results contain a non-{source} result")
        if not math.isfinite(result.score):
            raise ValueError("component scores must be finite")


def _add_source(
    records: dict[str, dict[str, object]],
    results: Sequence[RetrievalResult],
    source: str,
    weight: float,
) -> None:
    # A disabled source contributes neither ranks nor candidates.
    if weight == 0:
        return
    seen_in_source: set[str] = set()
    rank = 0
    for result in results:
        chunk = result.chunk
        existing = records.get(chunk.id)
        if existing is not None and existing["result"].chunk != chunk:
            raise ConflictingChunkProvenanceError(
                f"chunk ID {chunk.id!r} has conflicting source provenance"
            )
        if chunk.id in seen_in_source:
            # Identical duplicate rows do not consume a rank or vote twice.
            continue
        seen_in_source.add(chunk.id)
        rank += 1
        if existing is None:
            existing = {"result": result, "matched_terms": ()}
            records[chunk.id] = existing
        existing[f"{source}_rank"] = rank
        existing[f"{source}_score"] = result.score
        if source == "keyword":
            existing["matched_terms"] = result.matched_terms


def _fused_order(
    item: tuple[RetrievalResult, int, int | float],
) -> tuple[float, int, int | float, int, int, str, str, str]:
    result, first_rank, keyword_rank = item
    chunk = result.chunk
    return (
        -result.score,
        first_rank,
        keyword_rank,
        chunk.section_index,
        chunk.chunk_index,
        chunk.book_id,
        chunk.content_version_id,
        chunk.id,
    )
