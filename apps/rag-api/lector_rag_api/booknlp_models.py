"""Explicit, local-only setup for BookNLP's official small-model assets.

This command is never run by application startup. Downloads are staged beside
their final destination, validated as PyTorch state dictionaries, and renamed
atomically so interrupted transfers cannot be mistaken for installed models.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path

from .booknlp_graph import BOOKNLP_SMALL_MODEL_FILES, default_booknlp_model_dir

OFFICIAL_MODEL_BASE_URL = "http://people.ischool.berkeley.edu/~dbamman/booknlp_models/"
ProgressCallback = Callable[[str], None]


class BookNLPModelSetupError(RuntimeError):
    """Raised when an asset cannot be downloaded or validated safely."""


def _validate_pytorch_state_dict(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise BookNLPModelSetupError(f"Model asset is missing or empty: {path}")
    try:
        import torch

        # Never fall back to unrestricted pickle loading. These are official
        # tensor state dictionaries and must load with PyTorch's safe mode.
        value = torch.load(path, map_location="cpu", weights_only=True)
    except ImportError as exc:
        raise BookNLPModelSetupError(
            "PyTorch is not installed in this environment. Install the optional "
            "BookNLP runtime before preparing model assets."
        ) from exc
    except TypeError as exc:
        raise BookNLPModelSetupError(
            "This setup requires a PyTorch version that supports safe "
            "weights_only model loading. Upgrade the local BookNLP runtime."
        ) from exc
    except Exception as exc:
        raise BookNLPModelSetupError(
            f"Model asset is not a usable PyTorch state dictionary: {path.name}"
        ) from exc
    if not isinstance(value, Mapping) or not value:
        raise BookNLPModelSetupError(
            f"Model asset did not contain a non-empty state dictionary: {path.name}"
        )


def prepare_booknlp_models(
    model_dir: Path | str | None = None,
    *,
    downloader: Callable[[str, Path], None] | None = None,
    validator: Callable[[Path], None] = _validate_pytorch_state_dict,
    progress: ProgressCallback | None = None,
) -> tuple[Path, ...]:
    """Ensure all three BookNLP small weights exist and can be loaded safely.

    ``downloader`` and ``validator`` are injectable for offline tests. Existing
    valid files are preserved; invalid files are replaced only after a valid
    staged download has completed.
    """
    destination = Path(model_dir or default_booknlp_model_dir()).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    download = downloader or _download
    installed: list[Path] = []

    for filename in BOOKNLP_SMALL_MODEL_FILES:
        target = destination / filename
        if target.is_file():
            try:
                validator(target)
            except Exception:
                # Keep a bad pre-existing file until its replacement passes
                # validation; a failed download therefore preserves evidence.
                pass
            else:
                if progress:
                    progress(f"Already installed: {filename}")
                installed.append(target)
                continue

        fd, staged_name = tempfile.mkstemp(
            prefix=f".{filename}.", suffix=".part", dir=destination
        )
        os.close(fd)
        staged = Path(staged_name)
        try:
            if progress:
                progress(f"Downloading: {filename}")
            download(f"{OFFICIAL_MODEL_BASE_URL}{filename}", staged)
            validator(staged)
            os.replace(staged, target)
        except Exception as exc:
            staged.unlink(missing_ok=True)
            if isinstance(exc, BookNLPModelSetupError):
                raise
            raise BookNLPModelSetupError(
                f"Could not install {filename} into {destination}: {exc}"
            ) from exc
        if progress:
            progress(f"Installed: {filename}")
        installed.append(target)

    return tuple(installed)


def _download(url: str, target: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "Lector-BookNLP-Model-Setup/1.0"}
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response, target.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and validate BookNLP small-model assets in Lector's models directory."
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("LECTOR_BOOKNLP_MODEL_DIR") or str(default_booknlp_model_dir()),
        help="destination folder (default: LECTOR_BOOKNLP_MODEL_DIR or repository models/booknlp)",
    )
    args = parser.parse_args()
    try:
        files = prepare_booknlp_models(args.model_dir, progress=print)
    except BookNLPModelSetupError as exc:
        parser.exit(1, f"BookNLP model setup failed: {exc}\n")
    print(f"BookNLP small-model assets are ready in {Path(args.model_dir).resolve()}")
    for path in files:
        print(f"  {path.name} ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
