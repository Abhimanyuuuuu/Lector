"""Explicit setup for BookNLP's Hugging Face base tokenizer/model assets.

Normal BookNLP inference uses the repository-local cache in offline mode. This
module downloads only the three required files per component when the setup
command is run explicitly.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from .booknlp_graph import (
    BOOKNLP_HF_MODEL_ID,
    BOOKNLP_HF_MODEL_IDS,
    default_huggingface_home,
)

REQUIRED_BASE_MODEL_FILES = ("config.json", "pytorch_model.bin", "vocab.txt")


class SnapshotDownloader(Protocol):
    def __call__(
        self, *, repo_id: str, cache_dir: str, allow_patterns: list[str],
        local_files_only: bool,
    ) -> str: ...


class BookNLPBaseModelSetupError(RuntimeError):
    """Raised when the tokenizer/base model is missing or cannot be fetched."""


def configure_huggingface_environment(cache_home: Path | str, *, offline: bool) -> Path:
    """Configure this Python process before importing Hugging Face packages."""
    resolved = Path(cache_home).expanduser().resolve()
    os.environ["HF_HOME"] = str(resolved)
    os.environ["HF_HUB_CACHE"] = str(resolved / "hub")
    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"
    return resolved


def _snapshot_download(
    *, repo_id: str, cache_dir: str, allow_patterns: list[str],
    local_files_only: bool,
) -> str:
    # Import only after the caller has configured HF_HOME and offline mode.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise BookNLPBaseModelSetupError(
            "huggingface_hub is unavailable in this Python environment. "
            "Install the already-declared BookNLP runtime before preparing its base model."
        ) from exc
    return snapshot_download(
        repo_id=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        local_files_only=local_files_only,
    )


def validate_base_model_snapshot(snapshot_dir: Path | str) -> tuple[Path, ...]:
    snapshot = Path(snapshot_dir)
    files = tuple(snapshot / name for name in REQUIRED_BASE_MODEL_FILES)
    missing = [path.name for path in files if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise BookNLPBaseModelSetupError(
            f"Hugging Face snapshot is missing required non-empty files: {', '.join(missing)}"
        )
    return files


def prepare_booknlp_base_model(
    cache_home: Path | str | None = None,
    *,
    downloader: SnapshotDownloader | None = None,
    local_only: bool = False,
    progress: Callable[[str], None] | None = None,
) -> tuple[Path, ...]:
    """Validate or explicitly fetch the base assets for all three components.

    Hugging Face Hub owns the cache layout; using its snapshot API ensures the
    repository IDs derived by BookNLP resolve correctly without hand-building
    Hub cache symlinks or metadata. A complete local snapshot is checked first,
    including in explicit setup mode, so existing assets are never fetched again.
    """
    home = configure_huggingface_environment(
        cache_home or default_huggingface_home(), offline=local_only
    )
    snapshot_download = downloader or _snapshot_download
    prepared: list[Path] = []
    errors: list[str] = []
    for repo_id in BOOKNLP_HF_MODEL_IDS:
        files: tuple[Path, ...] | None = None
        local_error: Exception | None = None
        try:
            snapshot_dir = snapshot_download(
                repo_id=repo_id,
                cache_dir=str(home / "hub"),
                allow_patterns=list(REQUIRED_BASE_MODEL_FILES),
                local_files_only=True,
            )
            files = validate_base_model_snapshot(snapshot_dir)
        except Exception as exc:
            local_error = exc

        state = "Verified cached"
        if files is None:
            if local_only:
                detail = str(local_error) if local_error else "snapshot is incomplete"
                errors.append(f"{repo_id}: {detail}")
                continue
            try:
                snapshot_dir = snapshot_download(
                    repo_id=repo_id,
                    cache_dir=str(home / "hub"),
                    allow_patterns=list(REQUIRED_BASE_MODEL_FILES),
                    local_files_only=False,
                )
                files = validate_base_model_snapshot(snapshot_dir)
                state = "Prepared"
            except Exception as exc:
                errors.append(f"{repo_id}: {exc}")
                continue

        prepared.extend(files)
        if progress:
            progress(f"{state} {repo_id} in {home}")
            for path in files:
                progress(f"  {path.name} ({path.stat().st_size:,} bytes)")

    if errors:
        action = "local cache check" if local_only else "explicit setup"
        raise BookNLPBaseModelSetupError(
            f"BookNLP base assets are incomplete in {home} during {action}: "
            + "; ".join(errors)
        )
    return tuple(prepared)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Explicitly prepare BookNLP 1.0.8's entity, coreference, and "
            "speaker base tokenizer/models in Lector's repository-local cache."
        )
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="verify the local cache only; do not download missing files",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cache_home = default_huggingface_home()
    try:
        files = prepare_booknlp_base_model(
            cache_home, local_only=args.check, progress=print
        )
    except BookNLPBaseModelSetupError as exc:
        parser.exit(1, f"BookNLP base-model setup failed: {exc}\n")
    print(f"BookNLP component base tokenizer/model assets are ready in {cache_home.resolve()}")
    for path in files:
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
