"""Deterministic adaptation of BookNLP artifacts to Lector graph records.

BookNLP analyzes a whole English text. This module parses its ``.book``,
``.entities`` and ``.tokens`` artifacts and maps character offsets to exact
Flow source units. Canonical chunk IDs are retained as optional passage links;
graph eligibility is based on source-unit locations, not chunk membership.
"""

from __future__ import annotations

import csv
import builtins
import hashlib
import importlib
import io
import json
import re
import os
from contextlib import contextmanager
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .chunking import BookChunk

BOOKNLP_GRAPH_VERSION = "booknlp-small-entity-quote-coref-v1"
BOOKNLP_SMALL_MODEL_FILES = (
    "entities_google_bert_uncased_L-4_H-256_A-4-v1.0.model",
    "coref_google_bert_uncased_L-2_H-256_A-4-v1.0.model",
    "speaker_google_bert_uncased_L-8_H-256_A-4-v1.0.1.model",
)
BOOKNLP_HF_MODEL_ID = "google/bert_uncased_L-4_H-256_A-4"
# BookNLP 1.0.8 derives a different Hugging Face base model from each custom
# checkpoint path: entity, coreference, and speaker/quotation attribution.
BOOKNLP_HF_MODEL_IDS = (
    BOOKNLP_HF_MODEL_ID,
    "google/bert_uncased_L-2_H-256_A-4",
    "google/bert_uncased_L-8_H-256_A-4",
)
_BOOKNLP_TEXT_READER_MODULES = (
    "booknlp.english.english_booknlp",
    "booknlp.english.gender_inference_model_1",
    "booknlp.english.entity_tagger",
    "booknlp.english.name_coref",
    "booknlp.english.bert_coref_quote_pronouns",
    "booknlp.common.sequence_layered_reader",
)


def default_booknlp_model_dir() -> Path:
    """Return the repository-local BookNLP asset directory independent of cwd."""
    # booknlp_graph.py lives at <repo>/apps/rag-api/lector_rag_api.
    return Path(__file__).resolve().parents[3] / "models" / "booknlp"


def default_huggingface_home() -> Path:
    """Keep Hugging Face model/cache data in the repository's models folder."""
    return Path(__file__).resolve().parents[3] / "models" / "huggingface"

CONFLICT_VERBS = frozenset({"attack", "kill", "fight", "hit", "strike", "shoot", "threaten", "hate", "destroy", "stab", "murder", "beat", "wound", "hurt", "chase"})
HELP_VERBS = frozenset({"help", "save", "protect", "rescue", "support", "aid"})
TRANSFER_VERBS = frozenset({"give", "hand", "send", "offer", "take", "steal", "bring", "deliver", "receive", "return", "pass"})
DIALOGUE_VERBS = frozenset({"say", "tell", "ask", "reply", "speak", "shout", "whisper", "answer", "call", "talk", "mutter", "remark", "scream", "announce"})
MOVEMENT_VERBS = frozenset({"follow", "meet", "join", "leave", "enter", "approach", "arrive", "visit", "accompany", "escape", "travel"})
GENERIC_NAMES = frozenset({"", "unknown", "none", "null", "character", "someone", "somebody", "nobody", "everybody"})


@dataclass(frozen=True, slots=True)
class CharacterProfileRecord:
    character_id: str
    coref_id: str
    canonical_name: str
    mention_count: int
    aliases: tuple[dict[str, Any], ...]
    pronouns: tuple[dict[str, Any], ...]
    agents: tuple[dict[str, Any], ...]
    patients: tuple[dict[str, Any], ...]
    possessions: tuple[dict[str, Any], ...]
    modifiers: tuple[dict[str, Any], ...]
    gender: Any
    summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CharacterMentionRecord:
    mention_id: str
    character_id: str
    mention_text: str
    mention_kind: str
    category: str
    token_start: int
    token_end: int
    sentence_id: str | None
    paragraph_id: str | None
    source_chunk_id: str | None
    section_index: int | None
    chunk_index: int | None
    source_unit_ids: tuple[str, ...]
    source_unit_id: str | None = None
    unit_index: int | None = None
    start_character: int | None = None
    end_character: int | None = None
    href: str | None = None
    cfi: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateEdgeEvidence:
    evidence_id: str
    source_chunk_id: str
    section_index: int
    chunk_index: int
    source_unit_ids: tuple[str, ...]
    sentence_id: str
    paragraph_id: str | None
    sentence: str
    character_mentions: tuple[str, ...]
    verbs: tuple[str, ...]
    source_unit_id: str | None = None
    unit_index: int | None = None
    start_character: int | None = None
    end_character: int | None = None
    href: str | None = None
    source_cfi: str | None = None
    start_cfi: str | None = None
    end_cfi: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateCharacterEdge:
    edge_id: str
    source_character_id: str
    target_character_id: str
    candidate_category: str
    candidate_verbs: tuple[str, ...]
    evidence: tuple[CandidateEdgeEvidence, ...]
    evidence_truncated: bool = False

    @property
    def weight(self) -> int:
        return len(self.evidence)


@dataclass(frozen=True, slots=True)
class BookNLPGraphBundle:
    book_id: str
    content_version_id: str
    extraction_version: str
    profiles: tuple[CharacterProfileRecord, ...]
    mentions: tuple[CharacterMentionRecord, ...]
    edges: tuple[CandidateCharacterEdge, ...]
    unmapped_mentions: int = 0


@dataclass(frozen=True, slots=True)
class _SourceUnitSpan:
    start: int
    end: int
    chunk: BookChunk
    source_unit_id: str
    unit_index: int
    href: str | None
    cfi: str | None


class BookNLPRunner(Protocol):
    name: str
    version: str

    def process(self, text: str, book_id: str) -> Mapping[str, Any]:
        """Run the local whole-book pipeline and return parsed artifact data."""


def stable_id(*parts: str, prefix: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode("utf-8")).hexdigest()[:24]
    return f"{prefix}_{digest}"


def parse_tsv(value: str) -> list[dict[str, str]]:
    if not value.strip():
        return []
    return list(csv.DictReader(io.StringIO(value), delimiter="\t"))


def graph_from_artifacts(
    *, book_id: str, content_version_id: str, chunks: Sequence[BookChunk],
    book_data: Mapping[str, Any], entities: Sequence[Mapping[str, Any]],
    tokens: Sequence[Mapping[str, Any]], extraction_version: str = BOOKNLP_GRAPH_VERSION,
    max_evidence_per_edge: int = 100,
) -> BookNLPGraphBundle:
    """Build profiles, mention provenance, and heuristic co-mention edges.

    BookNLP 1.0.8 labels its output columns ``byte_onset`` and
    ``byte_offset``, but the installed spaCy pipeline writes ``tok.idx`` and
    ``len(token.text)``: Python string character offsets, not UTF-8 byte
    offsets. The input is canonical chunk text separated by two newlines.
    Sentence and mention offsets are accepted only when wholly contained in
    one source unit. Their original source text, offsets, and location metadata
    are retained independently of chunk boundaries.
    """
    if max_evidence_per_edge < 1:
        raise ValueError("max_evidence_per_edge must be positive")
    text, source_spans = assemble_booknlp_input(chunks)
    # BookNLP's ``byte_onset``/``byte_offset`` values are character offsets
    # into this Python string (see its Token/SpacyPipeline implementation).
    source_text = text

    summaries = book_data.get("characters", [])
    if not isinstance(summaries, list):
        raise ValueError("BookNLP .book characters must be a list")
    summary_by_coref: dict[str, Mapping[str, Any]] = {}
    for item in summaries:
        if isinstance(item, Mapping) and item.get("id") is not None:
            summary_by_coref[str(item["id"])] = item

    entity_rows = [dict(row) for row in entities]
    token_rows = [dict(row) for row in tokens]
    token_by_id: dict[int, dict[str, Any]] = {}
    sentence_tokens: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in token_rows:
        token_id = _int(_field(row, "token_ID_within_document", "token_id", "tokenId", "id"), -1)
        if token_id < 0:
            continue
        sentence_id = str(_field(row, "sentence_ID", "sentence_id", "sentenceId") or "")
        row["_token_id"] = token_id
        row["_sentence_id"] = sentence_id
        row["_word"] = str(_field(row, "word", "token", "originalWord", "text") or "").strip()
        row["_lemma"] = str(_field(row, "lemma", "lemmas") or row["_word"]).strip().casefold()
        row["_pos"] = str(_field(row, "POS_tag", "pos", "POS") or "").upper()
        row["_start"] = _int(_field(row, "byte_onset", "byte_offset_start", "start_byte"), -1)
        row["_end"] = _int(_field(row, "byte_offset", "byte_end", "end_byte"), -1)
        row["_paragraph_id"] = str(_field(row, "paragraph_ID", "paragraph_id", "paragraphId") or "") or None
        token_by_id[token_id] = row
        sentence_tokens[sentence_id].append(row)
    for rows in sentence_tokens.values():
        rows.sort(key=lambda row: row["_token_id"])

    # Stable cluster IDs use BookNLP's coreference assignment, names are never
    # fuzzy-merged. The .book summaries are retained as structured metadata.
    profile_data: dict[str, dict[str, Any]] = {}
    for coref, summary in summary_by_coref.items():
        mention_data = summary.get("mentions", {}) if isinstance(summary, Mapping) else {}
        if not isinstance(mention_data, Mapping):
            mention_data = {}
        proper = _terms(mention_data.get("proper", []))
        common = _terms(mention_data.get("common", []))
        canonical = next((term["term"] for term in proper), None) or next(
            (term["term"] for term in common), None
        ) or f"Character_{coref}"
        if _is_placeholder(canonical):
            continue
        aliases = _dedupe_terms((*proper, *common))
        profile_data[coref] = {
            "character_id": stable_id(book_id, content_version_id, coref, prefix="char"),
            "coref_id": coref,
            "canonical_name": canonical,
            "mention_count": _int(summary.get("count"), sum(x["count"] for x in aliases)),
            "aliases": tuple(aliases),
            "pronouns": tuple(_terms(mention_data.get("pronoun", []))),
            "agents": tuple(_terms(summary.get("agent", []))),
            "patients": tuple(_terms(summary.get("patient", []))),
            "possessions": tuple(_terms(summary.get("poss", []))),
            "modifiers": tuple(_terms(summary.get("mod", []))),
            "gender": summary.get("gender", {}),
            "summary": dict(summary),
        }

    mapped_mentions: list[CharacterMentionRecord] = []
    sentence_characters: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    sentence_paragraph: dict[str, str | None] = {}
    unmapped = 0
    for row in entity_rows:
        coref = str(_field(row, "COREF", "coref", "character_id") or "")
        profile = profile_data.get(coref)
        if profile is None:
            continue
        start_token = _int(_field(row, "start_token", "start"), -1)
        end_token = _int(_field(row, "end_token", "end"), start_token)
        if start_token < 0 or end_token < start_token:
            continue
        start_row, end_row = token_by_id.get(start_token), token_by_id.get(end_token)
        mention_text = str(_field(row, "text", "mention") or "").strip()
        sentence_ids = sorted({
            str(token_by_id[token_id]["_sentence_id"])
            for token_id in range(start_token, end_token + 1)
            if token_id in token_by_id and token_by_id[token_id]["_sentence_id"]
        })
        source_mention = _source_slice(source_text, start_row, end_row)
        span = _contained_unit(
            start_row.get("_start", -1) if start_row else -1,
            end_row.get("_end", -1) if end_row else -1,
            source_spans,
        ) if source_mention and _same_text(source_mention, mention_text) else None
        source_span = span
        chunk = source_span.chunk if source_span else None
        chunk_id = chunk.id if chunk else None
        if source_span is None:
            unmapped += 1
        mention_kind = str(_field(row, "prop", "mention_kind") or "")
        category = str(_field(row, "cat", "category") or "")
        sentence_id = sentence_ids[0] if len(sentence_ids) == 1 else None
        paragraph_id = start_row.get("_paragraph_id") if start_row else None
        mid = stable_id(book_id, content_version_id, coref, str(start_token), str(end_token), mention_text, prefix="mention")
        record = CharacterMentionRecord(
            mention_id=mid, character_id=profile["character_id"], mention_text=mention_text,
            mention_kind=mention_kind, category=category, token_start=start_token,
            token_end=end_token, sentence_id=sentence_id, paragraph_id=paragraph_id,
            source_chunk_id=chunk_id, section_index=chunk.section_index if chunk else None,
            chunk_index=chunk.chunk_index if chunk else None,
            source_unit_ids=(source_span.source_unit_id,) if source_span else (),
            source_unit_id=source_span.source_unit_id if source_span else None,
            unit_index=source_span.unit_index if source_span else None,
            start_character=start_row.get("_start") if source_span and start_row else None,
            end_character=end_row.get("_end") if source_span and end_row else None,
            href=source_span.href if source_span else None,
            cfi=source_span.cfi if source_span else None,
        )
        mapped_mentions.append(record)
        if source_span and sentence_id:
            sentence_characters[sentence_id][profile["character_id"]].append(mention_text)
            sentence_paragraph[sentence_id] = paragraph_id

    edge_sentences: dict[tuple[str, str], dict[str, CandidateEdgeEvidence]] = defaultdict(dict)
    for sentence_id, char_mentions in sorted(sentence_characters.items()):
        character_ids = sorted(char_mentions)
        if len(character_ids) < 2:
            continue
        token_list = sentence_tokens.get(sentence_id, [])
        if not token_list:
            continue
        sentence_span = _contained_unit(token_list[0]["_start"], token_list[-1]["_end"], source_spans)
        if sentence_span is None:
            continue
        chunk = sentence_span.chunk
        sentence = _source_slice(source_text, token_list[0], token_list[-1])
        if not sentence:
            continue
        verbs = tuple(dict.fromkeys(
            row["_lemma"] for row in token_list
            if row["_pos"] in {"VERB", "AUX", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
            and row["_lemma"]
        ))
        for index, left in enumerate(character_ids):
            for right in character_ids[index + 1:]:
                eid = stable_id(book_id, content_version_id, left, right, sentence_span.source_unit_id, sentence_id, prefix="edge-evidence")
                evidence = CandidateEdgeEvidence(
                    evidence_id=eid, source_chunk_id=chunk.id,
                    section_index=chunk.section_index, chunk_index=chunk.chunk_index,
                    source_unit_ids=(sentence_span.source_unit_id,), sentence_id=sentence_id,
                    paragraph_id=sentence_paragraph.get(sentence_id), sentence=sentence,
                    character_mentions=tuple(sorted({m for values in (char_mentions[left], char_mentions[right]) for m in values}, key=str.casefold)),
                    verbs=verbs,
                    source_unit_id=sentence_span.source_unit_id,
                    unit_index=sentence_span.unit_index,
                    start_character=token_list[0]["_start"],
                    end_character=token_list[-1]["_end"],
                    href=sentence_span.href,
                    source_cfi=sentence_span.cfi,
                )
                edge_sentences[(left, right)][eid] = evidence

    edges: list[CandidateCharacterEdge] = []
    for (left, right), unique_evidence in sorted(edge_sentences.items()):
        all_evidence = sorted(unique_evidence.values(), key=lambda item: (item.section_index, item.chunk_index, item.sentence_id, item.evidence_id))
        verbs = Counter(verb for evidence in all_evidence for verb in evidence.verbs)
        top_verbs = tuple(sorted(verbs, key=lambda verb: (-verbs[verb], verb))[:5])
        category = categorize_candidate_verbs(top_verbs)
        edge_id = stable_id(book_id, content_version_id, left, right, prefix="edge")
        edges.append(CandidateCharacterEdge(
            edge_id=edge_id, source_character_id=left, target_character_id=right,
            candidate_category=category, candidate_verbs=top_verbs,
            evidence=tuple(all_evidence[:max_evidence_per_edge]),
            evidence_truncated=len(all_evidence) > max_evidence_per_edge,
        ))

    profiles = tuple(
        CharacterProfileRecord(**values)
        for _, values in sorted(profile_data.items(), key=lambda item: (item[1]["canonical_name"].casefold(), item[0]))
    )
    return BookNLPGraphBundle(
        book_id=book_id, content_version_id=content_version_id,
        extraction_version=extraction_version, profiles=profiles,
        mentions=tuple(sorted(mapped_mentions, key=lambda m: (m.section_index if m.section_index is not None else 10**9, m.chunk_index if m.chunk_index is not None else 10**9, m.token_start, m.mention_id))),
        edges=tuple(edges), unmapped_mentions=unmapped,
    )


def categorize_candidate_verbs(verbs: Sequence[str]) -> str:
    terms = {term.casefold() for term in verbs}
    for name, vocabulary in (
        ("POSSIBLE_CONFLICT", CONFLICT_VERBS), ("POSSIBLE_HELP", HELP_VERBS),
        ("POSSIBLE_TRANSFER", TRANSFER_VERBS), ("POSSIBLE_DIALOGUE", DIALOGUE_VERBS),
        ("POSSIBLE_MOVEMENT", MOVEMENT_VERBS),
    ):
        if terms & vocabulary:
            return name
    return "ACTION_UNVERIFIED" if terms else "CO_MENTION"


def assemble_booknlp_input(
    chunks: Sequence[BookChunk],
) -> tuple[str, tuple[_SourceUnitSpan, ...]]:
    """Reconstruct unique ordered source text, removing chunk overlap.

    Chunk text joins its source units with blank lines. Since the chunker
    deliberately overlaps whole units, concatenating chunk text would count
    some paragraphs repeatedly. Every unit is assigned to its first ordered
    canonical chunk for stable evidence provenance.
    """
    units: dict[str, tuple[int, int, str, BookChunk, str | None, str | None, str]] = {}
    for chunk in sorted(
        chunks, key=lambda item: (item.section_index, item.chunk_index, item.id)
    ):
        positions = tuple(chunk.source_positions)
        parts = chunk.text.split("\n\n")
        if len(parts) != len(positions):
            raise ValueError(
                f"chunk {chunk.id!r} cannot be aligned to its source positions"
            )
        for position, part in zip(positions, parts, strict=True):
            previous = units.get(position.source_unit_id)
            candidate = (chunk.section_index, position.unit_index, part, chunk, position.href, position.cfi, position.source_unit_id)
            if previous is not None:
                if previous[:3] != candidate[:3]:
                    raise ValueError(
                        f"overlapping source unit {position.source_unit_id!r} has conflicting text or order"
                    )
                continue
            units[position.source_unit_id] = candidate

    ordered = sorted(
        units.items(), key=lambda item: (item[1][0], item[1][1], item[0])
    )
    full_text = "\n\n".join(value[2] for _, value in ordered)
    spans: list[_SourceUnitSpan] = []
    offset = 0
    for index, (source_unit_id, (section_index, unit_index, unit_text, chunk, href, cfi, _)) in enumerate(ordered):
        start = offset
        # BookNLP 1.0.8's spaCy token ``idx`` values are Python character
        # offsets, although its TSV header calls them byte offsets.
        offset += len(unit_text)
        spans.append(_SourceUnitSpan(start, offset, chunk, source_unit_id, unit_index, href, cfi))
        if index + 1 < len(ordered):
            offset += len("\n\n")
    return full_text, tuple(spans)


class BookNLPUnavailableError(RuntimeError):
    pass


@contextmanager
def _booknlp_utf8_text_reads():
    """Read BookNLP pipeline text as UTF-8 without changing binary I/O.

    The configured pipeline has bare text reads for the input EPUB text,
    gender hyperparameter TSV, generated token/entity text, and UTF-8/ASCII
    tagset, alias, and WordNet resources. BookNLP 1.0.8 omits ``encoding`` at
    these sites. Shadowing ``open`` only in the audited BookNLP modules gives
    those reads UTF-8 while leaving explicit encodings, writes, binary reads,
    Python builtins, and unrelated modules unchanged.
    """
    modules = [importlib.import_module(name) for name in _BOOKNLP_TEXT_READER_MODULES]
    missing = object()
    originals = [(module, module.__dict__.get("open", missing)) for module in modules]

    def make_utf8_open(original_open):
        def open_booknlp_text(file, *args, **kwargs):
            mode = kwargs.get("mode", args[0] if args else "r")
            # ``open`` positional arguments after ``file`` are mode,
            # buffering, encoding, errors, ...; preserve an explicit encoding.
            has_positional_encoding = len(args) >= 3
            if (
                "r" in mode
                and "b" not in mode
                and "encoding" not in kwargs
                and not has_positional_encoding
            ):
                kwargs = {**kwargs, "encoding": "utf-8"}
            if original_open is missing:
                return builtins.open(file, *args, **kwargs)
            return original_open(file, *args, **kwargs)

        return open_booknlp_text

    for module, original_open in originals:
        module.open = make_utf8_open(original_open)
    try:
        yield
    finally:
        for module, original_open in reversed(originals):
            if original_open is missing:
                del module.open
            else:
                module.open = original_open


class LocalBookNLPRunner:
    """Lazy local-only BookNLP adapter that requires preinstalled model files."""

    name = "booknlp"

    def __init__(self, model: str = "small", model_path: str | None = None) -> None:
        if model != "small":
            raise ValueError("Lector supports the BookNLP small model in this adapter")
        self.model = model
        self.model_path = str(
            Path(model_path or os.getenv("LECTOR_BOOKNLP_MODEL_DIR") or default_booknlp_model_dir())
            .expanduser()
            .resolve()
        )
        self.version = f"booknlp:{model}:{BOOKNLP_GRAPH_VERSION}"

    @property
    def missing_model_files(self) -> tuple[str, ...]:
        from pathlib import Path

        model_path = Path(self.model_path)
        return tuple(
            name for name in BOOKNLP_SMALL_MODEL_FILES
            if not (model_path / name).is_file() or (model_path / name).stat().st_size == 0
        )

    def process(self, text: str, book_id: str) -> Mapping[str, Any]:
        import os
        import tempfile
        from pathlib import Path

        model_path = Path(self.model_path)
        missing_models = self.missing_model_files
        if missing_models:
            raise BookNLPUnavailableError(
                "BookNLP small-model assets are missing from "
                f"{model_path}: {', '.join(missing_models)}. Run "
                "`python -m lector_rag_api.booknlp_models` from apps/rag-api "
                "to install them into Lector's models directory. Normal reading "
                "and keyword retrieval remain available without Graph RAG. "
                "Model assets are never downloaded during normal graph processing."
            )
        # Hugging Face reads HF_HOME while importing huggingface_hub. Set it,
        # together with offline mode, before BookNLP imports Transformers.
        from .booknlp_base_model import (
            BookNLPBaseModelSetupError,
            configure_huggingface_environment,
            prepare_booknlp_base_model,
        )

        # Hugging Face reads these variables during import. Keep model/cache
        # resolution on the repository drive and force inference to stay offline.
        configure_huggingface_environment(default_huggingface_home(), offline=True)

        try:
            prepare_booknlp_base_model(
                default_huggingface_home(), local_only=True
            )
        except BookNLPBaseModelSetupError as exc:
            raise BookNLPUnavailableError(
                "BookNLP's local Hugging Face tokenizer/base model is missing or "
                "incomplete. Run `python -m lector_rag_api.booknlp_base_model` "
                "from apps/rag-api once to prepare it; graph inference remains offline. "
                f"Details: {exc}"
            ) from exc
        try:
            from booknlp.booknlp import BookNLP  # type: ignore[import-not-found]
        except ImportError as exc:
            raise BookNLPUnavailableError(
                "BookNLP is not installed. Install the optional `booknlp` extra and its English model assets locally."
            ) from exc
        # The explicit asset preflight above prevents upstream's urllib download
        # path; Hugging Face is pinned to the repository-local cache and offline.
        safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", book_id)[:80] or "book"
        with tempfile.TemporaryDirectory(prefix="lector-booknlp-") as temporary:
            root = Path(temporary)
            input_path = root / f"{safe_id}.txt"
            output_dir = root / "output"
            output_dir.mkdir()
            input_path.write_text(text, encoding="utf-8")
            try:
                # BookNLP's built-in "small" mode downloads absent files itself.
                # Its documented "custom" mode accepts exact paths and avoids
                # that fallback, while still using the same three small weights.
                from .booknlp_compat import booknlp_state_dict_compatibility

                # Text assets are read during BookNLP construction as well as
                # processing (aliases, WordNet/tagsets, input and gender TSV).
                # Keep the compatibility patch scoped around both operations.
                with _booknlp_utf8_text_reads(), booknlp_state_dict_compatibility():
                    pipeline = BookNLP("en", {
                        "pipeline": "entity,quote,coref",
                        "model": "custom",
                        # BookNLP 1.0.8 splits this path on '/' to derive its HF
                        # base-model ID. Path.as_posix is required on Windows.
                        "entity_model_path": (model_path / BOOKNLP_SMALL_MODEL_FILES[0]).as_posix(),
                        "coref_model_path": (model_path / BOOKNLP_SMALL_MODEL_FILES[1]).as_posix(),
                        "quote_attribution_model_path": (model_path / BOOKNLP_SMALL_MODEL_FILES[2]).as_posix(),
                    })
                    pipeline.process(str(input_path), str(output_dir), safe_id)
            except Exception as exc:
                raise BookNLPUnavailableError(
                    "Local BookNLP could not initialize or process this English book. "
                    f"Verify BookNLP, PyTorch, spaCy, and the three model files in {model_path}; "
                    "rerun `python -m lector_rag_api.booknlp_models` if an asset is corrupt."
                ) from exc
            book_path = output_dir / f"{safe_id}.book"
            entities_path = output_dir / f"{safe_id}.entities"
            tokens_path = output_dir / f"{safe_id}.tokens"
            if not book_path.is_file() or not entities_path.is_file() or not tokens_path.is_file():
                raise BookNLPUnavailableError("BookNLP completed without its required .book, .entities, and .tokens artifacts.")
            return {
                "book": json.loads(book_path.read_text(encoding="utf-8")),
                "entities": parse_tsv(entities_path.read_text(encoding="utf-8")),
                "tokens": parse_tsv(tokens_path.read_text(encoding="utf-8")),
            }


def _contained_unit(start: int, end: int, spans: Sequence[_SourceUnitSpan]) -> _SourceUnitSpan | None:
    """Return a source unit only when the complete character range lies inside it."""
    if start < 0 or end <= start:
        return None
    return next((span for span in spans if span.start <= start and end <= span.end), None)


def _field(row: Mapping[str, Any], *names: str) -> Any:
    normalized = {str(key).strip().casefold(): value for key, value in row.items()}
    for name in names:
        if name.casefold() in normalized:
            return normalized[name.casefold()]
    return None


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError, OverflowError):
        return default


def _terms(values: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not isinstance(values, (list, tuple)):
        return result
    for item in values:
        if isinstance(item, Mapping):
            name = str(item.get("n", item.get("w", ""))).strip()
            count = _int(item.get("c", item.get("count", 0)))
        else:
            name, count = str(item).strip(), 0
        if name:
            result.append({"term": name, "count": count})
    result.sort(key=lambda item: (-item["count"], item["term"].casefold(), item["term"]))
    return result


def _dedupe_terms(values: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in values:
        key = item["term"].casefold()
        if key not in seen:
            result.append(item)
            seen.add(key)
    return result


def _is_placeholder(name: str) -> bool:
    clean = name.strip()
    return bool(
        re.fullmatch(r"Character[_\s-]*\d+", clean, re.IGNORECASE)
        or clean.casefold() in GENERIC_NAMES
        or not any(char.isalpha() for char in clean)
    )


def _detokenize(words: Sequence[str]) -> str:
    text = " ".join(words)
    text = re.sub(r"\s+([,.!?;:%)\]])", r"\1", text)
    text = re.sub(r"([\(\[])\s+", r"\1", text)
    for suffix in ("n't", "'s", "'re", "'ve", "'ll", "'d", "'m"):
        text = text.replace(" " + suffix, suffix)
    return text.strip()


def _source_slice(source: str, start_row: Mapping[str, Any] | None, end_row: Mapping[str, Any] | None) -> str:
    """Return a mention span only when its character offsets are valid."""
    if start_row is None or end_row is None:
        return ""
    start, end = start_row.get("_start", -1), end_row.get("_end", -1)
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start or end > len(source):
        return ""
    return source[start:end]


def _same_text(left: str, right: str) -> bool:
    """Compare source text conservatively while tolerating whitespace variation."""
    normalize = lambda value: " ".join(value.casefold().split())
    return bool(right.strip()) and normalize(left) == normalize(right)


def bundle_to_json(bundle: BookNLPGraphBundle) -> dict[str, Any]:
    return {
        "book_id": bundle.book_id,
        "content_version_id": bundle.content_version_id,
        "extraction_version": bundle.extraction_version,
        "unmapped_mentions": bundle.unmapped_mentions,
        "profiles": [asdict(record) for record in bundle.profiles],
        "mentions": [asdict(record) for record in bundle.mentions],
        "edges": [asdict(record) for record in bundle.edges],
    }
