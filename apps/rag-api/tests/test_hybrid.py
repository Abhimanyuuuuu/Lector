"""Unit tests for score-independent reciprocal-rank fusion."""

import unittest
from dataclasses import replace

from lector_rag_api.chunking import chunk_source_units
from lector_rag_api.hybrid import (
    ConflictingChunkProvenanceError,
    HybridConfig,
    fuse_results,
)
from lector_rag_api.retrieval_types import RetrievalResult
from lector_rag_api.schemas import SourceUnitInput


def result(
    chunk_id: str,
    method: str,
    score: float,
    *,
    text: str | None = None,
    source_id: str | None = None,
) -> RetrievalResult:
    source = source_id or chunk_id
    chunk = chunk_source_units(
        "book",
        "version",
        [
            SourceUnitInput(
                source_unit_id=source,
                section_index=0,
                unit_index=0,
                kind="paragraph",
                text=text or f"text for {source}",
            )
        ],
        overlap=0,
    )[0]
    if chunk.id != chunk_id:
        chunk = replace(chunk, id=chunk_id)
    return RetrievalResult(
        chunk=chunk,
        score=score,
        retrieval_method=method,
        matched_terms=("matched",) if method == "keyword" else (),
    )


class HybridRetrievalTests(unittest.TestCase):
    def test_hand_calculated_rrf_and_shared_result_contribution(self) -> None:
        shared_keyword = result("shared", "keyword", 8.5)
        keyword_only = result("keyword-only", "keyword", 2.0)
        shared_semantic = replace(
            shared_keyword, retrieval_method="semantic", score=0.75, matched_terms=()
        )
        semantic_only = result("semantic-only", "semantic", 0.5)

        fused = fuse_results(
            [shared_keyword, keyword_only],
            [shared_semantic, semantic_only],
        )

        self.assertEqual([item.chunk.id for item in fused], ["shared", "keyword-only", "semantic-only"])
        self.assertAlmostEqual(fused[0].score, 2 / 61)
        self.assertAlmostEqual(fused[1].score, 1 / 62)
        self.assertAlmostEqual(fused[2].score, 1 / 62)
        self.assertEqual((fused[0].keyword_rank, fused[0].semantic_rank), (1, 1))
        self.assertEqual((fused[0].keyword_score, fused[0].semantic_score), (8.5, 0.75))
        self.assertEqual(fused[0].matched_terms, ("matched",))

    def test_weights_and_rrf_constant_are_configurable(self) -> None:
        keyword = result("kw", "keyword", 100)
        semantic = result("sem", "semantic", 0.99)
        default = fuse_results([keyword], [semantic])
        weighted = fuse_results(
            [keyword],
            [semantic],
            config=HybridConfig(k=1, keyword_weight=0.25, semantic_weight=3),
        )

        self.assertAlmostEqual(default[0].score, 1 / 61)
        self.assertAlmostEqual(default[1].score, 1 / 61)
        self.assertAlmostEqual(weighted[0].score, 3 / 2)
        self.assertAlmostEqual(weighted[1].score, 0.25 / 2)
        keyword_only = fuse_results(
            [keyword],
            [semantic],
            config=HybridConfig(keyword_weight=1, semantic_weight=0),
        )
        self.assertEqual([item.chunk.id for item in keyword_only], ["kw"])

    def test_invalid_configuration_and_limits_are_rejected(self) -> None:
        for config in (
            {"k": 0},
            {"k": True},
            {"keyword_weight": -1},
            {"semantic_weight": float("inf")},
            {"keyword_weight": 10**1000},
            {"keyword_weight": 0, "semantic_weight": 0},
        ):
            with self.subTest(config=config), self.assertRaises((TypeError, ValueError)):
                HybridConfig(**config)
        with self.assertRaises(ValueError):
            fuse_results([], [], limit=-1)

    def test_ties_are_deterministic_and_source_order_breaks_them(self) -> None:
        keyword_first = result("keyword-first", "keyword", 1)
        semantic_first = result("semantic-first", "semantic", 1)
        first = fuse_results([keyword_first], [semantic_first])
        second = fuse_results([keyword_first], [semantic_first])

        self.assertEqual([item.chunk.id for item in first], ["keyword-first", "semantic-first"])
        self.assertEqual([item.chunk.id for item in first], [item.chunk.id for item in second])

    def test_empty_lists_and_zero_limit(self) -> None:
        semantic = result("semantic", "semantic", 0.2)
        self.assertEqual(fuse_results([], []), [])
        self.assertEqual(fuse_results([], [semantic])[0].semantic_rank, 1)
        self.assertEqual(fuse_results([result("kw", "keyword", 1)], []), [
            fuse_results([result("kw", "keyword", 1)], [])[0]
        ])
        self.assertEqual(fuse_results([], [semantic], limit=0), [])

    def test_matching_identity_deduplicates_but_conflicting_provenance_fails(self) -> None:
        keyword = result("shared-id", "keyword", 3, source_id="source-a")
        semantic = replace(keyword, retrieval_method="semantic", score=0.5)
        fused = fuse_results([keyword], [semantic])
        self.assertEqual(len(fused), 1)
        self.assertEqual(fused[0].chunk.source_unit_ids, keyword.chunk.source_unit_ids)

        conflicting = result("shared-id", "semantic", 0.5, source_id="source-b")
        with self.assertRaises(ConflictingChunkProvenanceError):
            fuse_results([keyword], [conflicting])

        repeated = fuse_results(
            [keyword, keyword, result("next", "keyword", 1)], []
        )
        self.assertEqual(repeated[0].keyword_rank, 1)
        self.assertEqual(repeated[1].keyword_rank, 2)

    def test_inputs_are_not_mutated(self) -> None:
        keywords = [result("kw", "keyword", 2)]
        semantics = [result("sem", "semantic", 0.5)]
        before = (keywords[:], semantics[:])
        fuse_results(keywords, semantics)
        self.assertEqual(keywords, before[0])
        self.assertEqual(semantics, before[1])


if __name__ == "__main__":
    unittest.main()
