"""Deterministic, section-local chunking of Flow-extracted source units."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

from .schemas import SourceUnitInput

DEFAULT_MAX_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
MAX_SAFE_INTEGER = 9_007_199_254_740_991


@dataclass(frozen=True, slots=True)
class ChunkSourcePosition:
    source_unit_id: str
    unit_index: int
    kind: str
    href: str | None
    cfi: str | None


@dataclass(frozen=True, slots=True)
class BookChunk:
    id: str
    book_id: str
    content_version_id: str
    section_index: int
    chunk_index: int
    text: str
    source_unit_ids: tuple[str, ...]
    source_positions: tuple[ChunkSourcePosition, ...]
    start_unit_index: int
    end_unit_index: int
    href: str | None
    heading: str | None
    start_cfi: str | None
    end_cfi: str | None
    cfi_status: str


def utf16_code_units(text: str) -> int:
    """Return JavaScript-compatible string length, including surrogate pairs."""
    return len(text.encode("utf-16-le", errors="surrogatepass")) // 2


def _validate_options(max_chunk_size: int, overlap: int | None) -> tuple[int, int]:
    if isinstance(max_chunk_size, bool) or not isinstance(max_chunk_size, int):
        raise TypeError("max_chunk_size must be an integer")
    if max_chunk_size <= 0 or max_chunk_size > MAX_SAFE_INTEGER:
        raise ValueError("max_chunk_size must be a positive safe integer")
    if overlap is None:
        overlap = min(DEFAULT_CHUNK_OVERLAP, max_chunk_size // 4)
    if isinstance(overlap, bool) or not isinstance(overlap, int):
        raise TypeError("overlap must be an integer")
    if overlap < 0 or overlap >= max_chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than max_chunk_size"
        )
    return max_chunk_size, overlap


def _serialized_text(units: list[SourceUnitInput]) -> str:
    return "\n\n".join(unit.text for unit in units)


def _is_cfi(value: str | None) -> bool:
    return value is not None and re.fullmatch(r"epubcfi\(.+\)", value) is not None


def _chunk_cfi(
    units: list[SourceUnitInput],
) -> tuple[str | None, str | None, str]:
    supplied = [unit.cfi for unit in units if unit.cfi is not None]
    if any(not _is_cfi(cfi) for cfi in supplied):
        status = "unreliable"
    elif not supplied:
        status = "unavailable"
    elif len(supplied) == len(units):
        status = "complete"
    else:
        status = "partial"
    first_cfi = units[0].cfi
    last_cfi = units[-1].cfi
    return (
        first_cfi if _is_cfi(first_cfi) else None,
        last_cfi if _is_cfi(last_cfi) else None,
        status,
    )


def _encoded_id_part(value: str) -> str:
    # Match JavaScript encodeURIComponent's unescaped character set.
    return quote(value, safe="-_.!~*'()")


def _create_chunk(
    units: list[SourceUnitInput],
    book_id: str,
    content_version_id: str,
    chunk_index: int,
    max_chunk_size: int,
    overlap: int,
) -> BookChunk:
    first = units[0]
    last = units[-1]
    source_ids = tuple(unit.source_unit_id for unit in units)
    heading = first.heading
    for unit in units:
        if unit.kind == "heading":
            heading = unit.text

    start_cfi, end_cfi, cfi_status = _chunk_cfi(units)
    id_parts = (
        book_id,
        content_version_id,
        str(first.section_index),
        str(chunk_index),
        str(max_chunk_size),
        str(overlap),
        *source_ids,
    )
    chunk_id = "chunk:" + ":".join(_encoded_id_part(part) for part in id_parts)
    return BookChunk(
        id=chunk_id,
        book_id=book_id,
        content_version_id=content_version_id,
        section_index=first.section_index,
        chunk_index=chunk_index,
        text=_serialized_text(units),
        source_unit_ids=source_ids,
        source_positions=tuple(
            ChunkSourcePosition(
                source_unit_id=unit.source_unit_id,
                unit_index=unit.unit_index,
                kind=unit.kind,
                href=unit.href,
                cfi=unit.cfi,
            )
            for unit in units
        ),
        start_unit_index=first.unit_index,
        end_unit_index=last.unit_index,
        href=first.href,
        heading=heading,
        start_cfi=start_cfi,
        end_cfi=end_cfi,
        cfi_status=cfi_status,
    )


def _make_overlap(
    units: list[SourceUnitInput], overlap: int, max_chunk_size: int
) -> list[SourceUnitInput]:
    if overlap == 0 or not units:
        return []
    suffix: list[SourceUnitInput] = []
    for unit in reversed(units):
        if utf16_code_units(unit.text) > max_chunk_size:
            break
        candidate = [unit, *suffix]
        if utf16_code_units(_serialized_text(candidate)) > overlap:
            break
        suffix.insert(0, unit)
    return suffix


def _unit_sort_key(unit: SourceUnitInput) -> tuple[int, str, str, str, str, str]:
    return (
        unit.unit_index,
        unit.source_unit_id,
        unit.text,
        unit.kind,
        unit.href or "",
        unit.cfi or "",
    )


def chunk_source_units(
    book_id: str,
    content_version_id: str,
    source_units: list[SourceUnitInput] | tuple[SourceUnitInput, ...],
    *,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
    overlap: int | None = None,
) -> list[BookChunk]:
    """Chunk source units without altering them or inventing location data.

    The character budget and overlap use UTF-16 code units, matching JavaScript
    ``String.length`` exactly. Separators (two newlines) count toward the
    budget. Units remain whole; oversized units form standalone chunks.
    Headings start a new chunk when possible and remain attached to following
    content when it fits. Overlap copies only whole trailing source units.
    """
    max_chunk_size, overlap = _validate_options(max_chunk_size, overlap)
    if not isinstance(book_id, str) or not book_id.strip():
        raise ValueError("book_id must be a non-empty string")
    if not isinstance(content_version_id, str) or not content_version_id.strip():
        raise ValueError("content_version_id must be a non-empty string")
    if not isinstance(source_units, (list, tuple)):
        raise TypeError("source_units must be a list or tuple")

    groups: dict[int, list[SourceUnitInput]] = {}
    seen_ids: set[str] = set()
    seen_positions: set[tuple[int, int]] = set()
    for unit in source_units:
        if not isinstance(unit, SourceUnitInput):
            raise TypeError("each source unit must be a validated SourceUnitInput")
        if not unit.source_unit_id.strip():
            raise ValueError("source_unit_id must be non-empty")
        if unit.source_unit_id in seen_ids:
            raise ValueError(f"duplicate source_unit_id: {unit.source_unit_id}")
        seen_ids.add(unit.source_unit_id)
        position = (unit.section_index, unit.unit_index)
        if position in seen_positions:
            raise ValueError("section_index/unit_index positions must be unique")
        seen_positions.add(position)
        if unit.text == "":
            continue
        groups.setdefault(unit.section_index, []).append(unit)

    chunks: list[BookChunk] = []
    for section_index in sorted(groups):
        units = sorted(groups[section_index], key=_unit_sort_key)
        cursor = 0
        chunk_index = 0
        carry: list[SourceUnitInput] = []

        while cursor < len(units):
            current = list(carry)
            carry = []
            fresh_count = 0
            oversized_emitted = False

            while cursor < len(units):
                next_unit = units[cursor]
                has_body = any(unit.kind != "heading" for unit in current)
                if next_unit.kind == "heading" and current and has_body:
                    break

                candidate = [*current, next_unit]
                if utf16_code_units(_serialized_text(candidate)) > max_chunk_size:
                    if fresh_count == 0 and current:
                        current = []
                        continue
                    if not current:
                        chunks.append(
                            _create_chunk(
                                [next_unit],
                                book_id,
                                content_version_id,
                                chunk_index,
                                max_chunk_size,
                                overlap,
                            )
                        )
                        chunk_index += 1
                        cursor += 1
                        oversized_emitted = True
                    break

                current.append(next_unit)
                cursor += 1
                fresh_count += 1

            if oversized_emitted:
                continue
            if fresh_count == 0:
                if current:
                    continue
                break

            chunks.append(
                _create_chunk(
                    current,
                    book_id,
                    content_version_id,
                    chunk_index,
                    max_chunk_size,
                    overlap,
                )
            )
            chunk_index += 1
            if cursor < len(units):
                carry = _make_overlap(current, overlap, max_chunk_size)

    return chunks
