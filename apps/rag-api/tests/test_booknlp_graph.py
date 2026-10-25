"""Offline BookNLP artifact mapping and deterministic candidate graph tests."""

import unittest
from pathlib import Path
import sys
import types
from unittest.mock import patch

from lector_rag_api.booknlp_graph import (
    BookNLPUnavailableError,
    LocalBookNLPRunner,
    _BOOKNLP_TEXT_READER_MODULES,
    _booknlp_utf8_text_reads,
    assemble_booknlp_input,
    graph_from_artifacts,
)
from lector_rag_api.chunking import chunk_source_units
from lector_rag_api.schemas import SourceUnitInput


def tokens_for(text: str) -> list[dict[str, object]]:
    import re

    result = []
    matches = list(re.finditer(r"[A-Za-z']+|[.!?]", text))
    sentence = 0
    for token_id, match in enumerate(matches):
        word = match.group(0)
        result.append({
            "paragraph_ID": "0", "sentence_ID": str(sentence),
            "token_ID_within_document": str(token_id), "word": word,
            "lemma": {"met": "meet", "helped": "help"}.get(
                word.casefold(), word.casefold().rstrip("s")
            ),
            # BookNLP 1.0.8 writes spaCy's Python character offsets despite
            # naming these TSV columns byte_onset/byte_offset.
            "byte_onset": str(len(text[:match.start()])),
            "byte_offset": str(len(text[:match.end()])),
            "POS_tag": "VERB" if word.casefold() in {"met", "helped"} else "NOUN",
        })
        if word in {".", "!", "?"}:
            sentence += 1
    return result


class BookNLPGraphTests(unittest.TestCase):
    def test_utf8_open_patch_is_restored_when_processing_raises(self):
        modules = {
            name: types.ModuleType(name) for name in _BOOKNLP_TEXT_READER_MODULES
        }
        original_opens = {name: module.__dict__.get("open") for name, module in modules.items()}

        with patch(
            "lector_rag_api.booknlp_graph.importlib.import_module",
            side_effect=modules.__getitem__,
        ):
            with self.assertRaisesRegex(RuntimeError, "simulated failure"):
                with _booknlp_utf8_text_reads():
                    self.assertTrue(all("open" in module.__dict__ for module in modules.values()))
                    raise RuntimeError("simulated failure")

            for name, module in modules.items():
                self.assertEqual(module.__dict__.get("open"), original_opens[name])

    def make_bundle(self):
        text = "Tess met John. Tess helped John."
        chunks = chunk_source_units("book", "v1", [SourceUnitInput(
            source_unit_id="unit-1", section_index=0, unit_index=0,
            kind="paragraph", text=text, href="chapter.xhtml", cfi=None,
        )])
        token_rows = tokens_for(text)
        # Token positions: Tess(0), met(1), John(2), .(3), Tess(4), helped(5), John(6), .(7)
        entities = [
            {"COREF": "1", "start_token": 0, "end_token": 0, "prop": "PROP", "cat": "PER", "text": "Tess"},
            {"COREF": "2", "start_token": 2, "end_token": 2, "prop": "PROP", "cat": "PER", "text": "John"},
            {"COREF": "1", "start_token": 4, "end_token": 4, "prop": "PROP", "cat": "PER", "text": "Tess"},
            {"COREF": "2", "start_token": 6, "end_token": 6, "prop": "PROP", "cat": "PER", "text": "John"},
        ]
        summaries = {"characters": [
            {"id": "1", "count": 2, "mentions": {"proper": [{"n": "Tess", "c": 2}], "common": [{"n": "Teresa", "c": 1}], "pronoun": [{"n": "she", "c": 3}]}, "agent": [{"n": "help", "c": 1}]},
            {"id": "2", "count": 2, "mentions": {"proper": [{"n": "John", "c": 2}]}, "patient": [{"n": "help", "c": 1}]},
            {"id": "synthetic", "count": 9, "mentions": {"proper": [{"n": "Character_756", "c": 9}]}},
        ]}
        return graph_from_artifacts(
            book_id="book", content_version_id="v1", chunks=chunks,
            book_data=summaries, entities=entities, tokens=token_rows,
        )

    def test_profiles_aliases_ids_and_sentence_edges_are_deterministic(self):
        first = self.make_bundle()
        second = self.make_bundle()
        self.assertEqual(first, second)
        self.assertEqual([profile.canonical_name for profile in first.profiles], ["John", "Tess"])
        tess = next(profile for profile in first.profiles if profile.canonical_name == "Tess")
        self.assertIn("Teresa", [item["term"] for item in tess.aliases])
        self.assertEqual(tess.agents[0]["term"], "help")
        self.assertEqual(len(first.edges), 1)
        edge = first.edges[0]
        self.assertEqual(edge.weight, 2)
        self.assertEqual(edge.candidate_category, "POSSIBLE_HELP")
        self.assertEqual({item.source_chunk_id for item in edge.evidence}, {first.edges[0].evidence[0].source_chunk_id})
        self.assertEqual(edge.evidence[0].source_unit_ids, ("unit-1",))

    def test_unicode_before_mentions_maps_using_booknlp_character_offsets(self):
        text = 'Élodie said “Tess”—John helped.'
        chunks = chunk_source_units("book", "v1", [SourceUnitInput(
            source_unit_id="unicode-unit", section_index=0, unit_index=0,
            kind="paragraph", text=text,
        )])
        tokens = tokens_for(text)
        tess = next(row for row in tokens if row["word"] == "Tess")
        john = next(row for row in tokens if row["word"] == "John")
        bundle = graph_from_artifacts(
            book_id="book", content_version_id="v1", chunks=chunks,
            book_data={"characters": [
                {"id": "t", "count": 1, "mentions": {"proper": [{"n": "Tess", "c": 1}]}},
                {"id": "j", "count": 1, "mentions": {"proper": [{"n": "John", "c": 1}]}},
            ]},
            entities=[
                {"COREF": "t", "start_token": tess["token_ID_within_document"],
                 "end_token": tess["token_ID_within_document"], "text": "Tess", "prop": "PROP", "cat": "PER"},
                {"COREF": "j", "start_token": john["token_ID_within_document"],
                 "end_token": john["token_ID_within_document"], "text": "John", "prop": "PROP", "cat": "PER"},
            ],
            tokens=tokens,
        )
        self.assertEqual(bundle.unmapped_mentions, 0)
        self.assertEqual(len(bundle.mentions), 2)
        self.assertTrue(all(mention.source_chunk_id == chunks[0].id for mention in bundle.mentions))
        self.assertEqual(len(bundle.edges), 1)
        self.assertEqual(bundle.edges[0].evidence[0].source_chunk_id, chunks[0].id)
        self.assertEqual(bundle.mentions[0].source_unit_id, "unicode-unit")
        self.assertEqual(bundle.mentions[0].start_character, text.index("Tess"))
        self.assertEqual(bundle.edges[0].evidence[0].source_unit_id, "unicode-unit")
        self.assertEqual(
            bundle.edges[0].evidence[0].sentence,
            text[text.index("lodie"):],
        )

    def test_unmappable_mentions_are_not_attached_to_a_false_chunk(self):
        text = "Tess met John."
        chunk = chunk_source_units("book", "v1", [SourceUnitInput(
            source_unit_id="unit-1", section_index=0, unit_index=0,
            kind="paragraph", text="Different source text.",
        )])[0]
        bundle = graph_from_artifacts(
            book_id="book", content_version_id="v1", chunks=[chunk],
            book_data={"characters": [{"id": "1", "count": 1, "mentions": {"proper": [{"n": "Tess", "c": 1}]}}]},
            entities=[{"COREF": "1", "start_token": 0, "end_token": 0, "text": "Tess", "prop": "PROP", "cat": "PER"}],
            tokens=tokens_for(text),
        )
        self.assertEqual(bundle.unmapped_mentions, 1)
        self.assertIsNone(bundle.mentions[0].source_chunk_id)
        self.assertEqual(bundle.edges, ())

    def test_relationship_sentence_crossing_source_units_is_not_fabricated(self):
        units = [
            SourceUnitInput(
                source_unit_id="first", section_index=0, unit_index=0,
                kind="paragraph", text="Tess met",
            ),
            SourceUnitInput(
                source_unit_id="second", section_index=0, unit_index=1,
                kind="paragraph", text="John.",
            ),
        ]
        chunks = chunk_source_units("book", "v1", units)
        text, _ = assemble_booknlp_input(chunks)
        tokens = tokens_for(text)
        bundle = graph_from_artifacts(
            book_id="book", content_version_id="v1", chunks=chunks,
            book_data={"characters": [
                {"id": "t", "count": 1, "mentions": {"proper": [{"n": "Tess", "c": 1}]}},
                {"id": "j", "count": 1, "mentions": {"proper": [{"n": "John", "c": 1}]}},
            ]},
            entities=[
                {"COREF": "t", "start_token": 0, "end_token": 0, "text": "Tess", "prop": "PROP"},
                {"COREF": "j", "start_token": 2, "end_token": 2, "text": "John", "prop": "PROP"},
            ],
            tokens=tokens,
        )
        self.assertEqual({mention.source_unit_id for mention in bundle.mentions}, {"first", "second"})
        self.assertEqual(bundle.edges, ())

    def test_similar_names_remain_separate_clusters(self):
        text = "Anne met Ann."
        tokens = tokens_for(text)
        chunks = chunk_source_units("book", "v1", [SourceUnitInput(
            source_unit_id="u", section_index=0, unit_index=0, kind="paragraph", text=text,
        )])
        bundle = graph_from_artifacts(
            book_id="book", content_version_id="v1", chunks=chunks,
            book_data={"characters": [
                {"id": "a", "count": 1, "mentions": {"proper": [{"n": "Anne", "c": 1}]}},
                {"id": "b", "count": 1, "mentions": {"proper": [{"n": "Ann", "c": 1}]}},
            ]},
            entities=[
                {"COREF": "a", "start_token": 0, "end_token": 0, "text": "Anne", "prop": "PROP", "cat": "PER"},
                {"COREF": "b", "start_token": 2, "end_token": 2, "text": "Ann", "prop": "PROP", "cat": "PER"},
            ], tokens=tokens,
        )
        self.assertEqual(len(bundle.profiles), 2)
        self.assertNotEqual(bundle.profiles[0].character_id, bundle.profiles[1].character_id)
        self.assertEqual(len(bundle.edges), 1)
        self.assertEqual(bundle.edges[0].candidate_category, "POSSIBLE_MOVEMENT")

    def test_overlapping_chunks_reconstruct_each_source_unit_once(self):
        units = [
            SourceUnitInput(source_unit_id="a", section_index=0, unit_index=0,
                             kind="paragraph", text="Tess met John."),
            SourceUnitInput(source_unit_id="b", section_index=0, unit_index=1,
                             kind="paragraph", text="Tess helped John."),
            SourceUnitInput(source_unit_id="c", section_index=0, unit_index=2,
                             kind="paragraph", text="Tess saw John."),
        ]
        chunks = chunk_source_units(
            "book", "v1", units, max_chunk_size=34, overlap=18
        )
        self.assertEqual([chunk.source_unit_ids for chunk in chunks], [
            ("a", "b"), ("b", "c")
        ])
        text, spans = assemble_booknlp_input(chunks)
        self.assertEqual(
            text,
            "Tess met John.\n\nTess helped John.\n\nTess saw John.",
        )
        self.assertEqual(len(spans), 3)
        # The overlapped unit is deterministically attributed to its first chunk.
        self.assertEqual(spans[1].chunk.id, chunks[0].id)

    def test_missing_local_model_files_fail_without_attempting_download(self):
        model_path = Path(__file__).parent / "missing-booknlp-model-assets"
        runner = LocalBookNLPRunner(model_path=str(model_path))
        self.assertTrue(runner.missing_model_files)
        with self.assertRaisesRegex(BookNLPUnavailableError, "never downloaded"):
            runner.process("A short test.", "book")


if __name__ == "__main__":
    unittest.main()
