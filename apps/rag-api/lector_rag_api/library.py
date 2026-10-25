"""Safe local storage for imported EPUB originals."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import uuid
import zipfile
from typing import AsyncIterable


class InvalidEpubUpload(ValueError):
    pass


def default_epub_directory() -> Path:
    configured = os.getenv("LECTOR_EPUB_STORAGE_DIR", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parent.parent / "data" / "epubs"
    )


async def save_epub_upload(
    body: AsyncIterable[bytes],
    *,
    book_id: str,
    content_version_id: str,
    directory: Path | None = None,
    max_bytes: int = 512 * 1024 * 1024,
) -> tuple[Path, str, int]:
    """Stream, hash, validate, and atomically place an EPUB under a generated name."""
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", content_version_id):
        raise InvalidEpubUpload("content_version_id must be a SHA-256 EPUB fingerprint")
    root = (directory or default_epub_directory()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    storage_key = hashlib.sha256(
        f"{book_id}\0{content_version_id}".encode("utf-8")
    ).hexdigest() + ".epub"
    destination = (root / storage_key).resolve()
    if destination.parent != root:
        raise InvalidEpubUpload("invalid local storage key")
    temporary = root / f".{uuid.uuid4().hex}.part"
    digest = hashlib.sha256()
    total = 0
    try:
        with temporary.open("xb") as stream:
            async for piece in body:
                if not isinstance(piece, bytes):
                    raise InvalidEpubUpload("EPUB upload stream contained invalid data")
                total += len(piece)
                if total > max_bytes:
                    raise InvalidEpubUpload("EPUB exceeds the configured upload limit")
                digest.update(piece)
                stream.write(piece)
            stream.flush()
            os.fsync(stream.fileno())
        if total < 4 or digest.hexdigest() != content_version_id.removeprefix("sha256:"):
            raise InvalidEpubUpload("uploaded EPUB does not match its content-version fingerprint")
        if not zipfile.is_zipfile(temporary):
            raise InvalidEpubUpload("uploaded file is not a valid EPUB ZIP archive")
        os.replace(temporary, destination)
        return destination, storage_key, total
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
