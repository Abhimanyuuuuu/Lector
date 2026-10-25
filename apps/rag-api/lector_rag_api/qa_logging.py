"""Local, per-request text audit logs for answer-generation attempts.

These files can contain copyrighted book excerpts. They stay on the local
backend filesystem and are never served by the API.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import threading
import uuid


class QAAuditLogError(OSError):
    """The audit record could not be durably written."""


_write_lock = threading.RLock()


def default_qa_log_directory() -> Path:
    configured = os.getenv("LECTOR_QA_LOG_DIR", "").strip()
    return (
        Path(configured).expanduser().resolve()
        if configured
        else Path(__file__).resolve().parent.parent / "data" / "qa-logs"
    )


@dataclass(slots=True)
class QARequestLog:
    request_id: str
    path: Path

    @classmethod
    def begin(
        cls,
        *,
        query: str,
        retrieval_mode: str,
        book_id: str | None,
        content_version_id: object,
        boundaries: object,
    ) -> "QARequestLog":
        request_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        directory = default_qa_log_directory() / timestamp[:10]
        path = directory / f"{request_id}.txt"
        boundary_text = _text(boundaries)
        entry = (
            "=" * 64 + "\n"
            f"REQUEST ID: {request_id}\n"
            f"TIMESTAMP: {timestamp}\n"
            f"BOOK ID: {book_id or '(all books)'}\n"
            "CONTENT VERSION(S): "
            + (_text(content_version_id) if content_version_id is not None else "(unspecified)")
            + "\n"
            f"CONFIRMED BOUNDARY: {boundary_text}\n"
            f"RETRIEVAL MODE: {retrieval_mode}\n\n"
            "RAW QUESTION:\n" + query + "\n\n"
            "EVIDENCE SENT TO GROQ:\n(not yet sent)\n\n"
            "GROQ HTTP STATUS: (not attempted)\n\n"
            "RAW GROQ RESPONSE:\n(no response body received)\n\n"
            "POST-PROCESSING RESULT:\n(pending)\n"
        )
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with _write_lock, path.open("x", encoding="utf-8", newline="") as stream:
                stream.write(entry)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise QAAuditLogError("The local QA audit log could not be created.") from exc
        return cls(request_id=request_id, path=path)

    def record_transport(
        self,
        *,
        question: str,
        evidence_json: str,
        status: int | None,
        raw_body: str | None,
        outcome: str,
    ) -> None:
        evidence = _text(evidence_json)
        body = raw_body if raw_body is not None else "(no response body received)"
        block = (
            "\nGROQ TRANSPORT RECORD\n"
            f"GROQ HTTP STATUS: {status if status is not None else '(no response)'}\n"
            f"REQUEST OUTCOME: {outcome}\n\n"
            "RAW QUESTION SENT:\n" + question + "\n\n"
            "EVIDENCE SENT TO GROQ:\n" + evidence + "\n\n"
            "RAW GROQ RESPONSE:\n" + body + "\n"
        )
        self._append(block)

    def record_citation_aliases(self, alias_to_original: dict[str, str]) -> None:
        """Record transport aliases without duplicating passage text."""
        mapping = "\n".join(
            f"{alias} -> {canonical_id}"
            for alias, canonical_id in alias_to_original.items()
        ) or "(no passages)"
        self._append("\nCITATION ALIASES SENT TO MODEL:\n" + mapping + "\n")

    def record_citation_validation(
        self,
        *,
        raw_aliases: object,
        resolutions: object,
        outcome: str,
        reason: str | None,
    ) -> None:
        """Record parsed citation handling, separate from the raw HTTP body."""
        block = (
            "\nPARSED CITATION VALIDATION\n"
            "RAW CITATION ALIASES:\n" + _text(raw_aliases) + "\n"
            "ALIAS RESOLUTION:\n" + _text(resolutions) + "\n"
            f"CITATION VALIDATION: {outcome}\n"
            "CITATION REJECTION REASON:\n"
            + (reason if reason else "(none)") + "\n"
        )
        self._append(block)

    def finish(self, result: str, error: str | None = None) -> None:
        self._append(
            "\nPOST-PROCESSING RESULT:\n" + _text(result) + "\n"
            + ("ERROR, IF ANY:\n" + _text(error) + "\n" if error else "")
            + "=" * 64 + "\n"
        )

    def _append(self, value: str) -> None:
        try:
            with _write_lock, self.path.open("a", encoding="utf-8", newline="") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as exc:
            raise QAAuditLogError("The local QA audit log could not be updated.") from exc


def _text(value: object) -> str:
    if isinstance(value, str):
        return value
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
