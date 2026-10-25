"""Targeted, offline checks for BookNLP model asset setup and path resolution."""

from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
import builtins
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch

from lector_rag_api.booknlp_graph import (
    BOOKNLP_HF_MODEL_IDS,
    BOOKNLP_SMALL_MODEL_FILES,
    _BOOKNLP_TEXT_READER_MODULES,
    BookNLPUnavailableError,
    LocalBookNLPRunner,
    default_booknlp_model_dir,
    default_huggingface_home,
)
from lector_rag_api.booknlp_base_model import (
    BOOKNLP_HF_MODEL_ID,
    REQUIRED_BASE_MODEL_FILES,
    BookNLPBaseModelSetupError,
    build_parser as build_base_model_parser,
    configure_huggingface_environment,
    prepare_booknlp_base_model,
)
from lector_rag_api.booknlp_models import (
    BookNLPModelSetupError,
    prepare_booknlp_models,
)
from lector_rag_api.schemas import SourceUnitInput
from lector_rag_api.service import InMemoryRagService


class BookNLPModelSetupTests(unittest.TestCase):
    def test_default_directory_is_under_repository_models(self):
        model_dir = default_booknlp_model_dir()
        self.assertEqual(model_dir.parts[-2:], ("models", "booknlp"))
        self.assertTrue((model_dir.parents[1] / "apps" / "rag-api").is_dir())
        self.assertNotIn("booknlp_models", str(model_dir))

    def test_huggingface_cache_defaults_to_repository_models_directory(self):
        cache_home = default_huggingface_home()
        self.assertEqual(cache_home.parts[-2:], ("models", "huggingface"))
        self.assertEqual(cache_home.drive.upper(), Path(__file__).resolve().drive.upper())

    def test_setup_parser_defaults_and_check_flag(self):
        defaults = build_base_model_parser().parse_args([])
        check = build_base_model_parser().parse_args(["--check"])
        self.assertEqual(vars(defaults), {"check": False})
        self.assertFalse(defaults.check)
        self.assertTrue(check.check)

    def test_setup_verifies_existing_assets_without_downloading(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary)
            snapshot = home / "snapshot"
            snapshot.mkdir()
            for filename in REQUIRED_BASE_MODEL_FILES:
                (snapshot / filename).write_bytes(b"model-file")
            calls: list[dict[str, object]] = []

            def downloader(**kwargs: object) -> str:
                calls.append(kwargs)
                return str(snapshot)

            files = prepare_booknlp_base_model(
                home, downloader=downloader, local_only=True
            )
            self.assertEqual(
                [path.name for path in files],
                list(REQUIRED_BASE_MODEL_FILES) * len(BOOKNLP_HF_MODEL_IDS),
            )
            self.assertEqual([call["repo_id"] for call in calls], list(BOOKNLP_HF_MODEL_IDS))
            self.assertTrue(all(call["allow_patterns"] == list(REQUIRED_BASE_MODEL_FILES) for call in calls))
            self.assertTrue(all(call["local_files_only"] is True for call in calls))

    def test_explicit_setup_enables_hub_only_in_its_process(self):
        original_environment = {
            name: os.environ.get(name)
            for name in ("HF_HOME", "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
        }
        with tempfile.TemporaryDirectory() as temporary, patch.dict(
            os.environ,
            {"HF_HOME": "C:/old-cache", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        ):
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            for filename in REQUIRED_BASE_MODEL_FILES:
                (snapshot / filename).write_bytes(b"model-file")
            observed: list[dict[str, object]] = []

            def downloader(**kwargs: object) -> str:
                observed.append({
                    "hf_home": os.environ.get("HF_HOME"),
                    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
                    "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
                    **kwargs,
                })
                if kwargs["local_files_only"]:
                    raise OSError("not in local cache")
                return str(snapshot)

            prepare_booknlp_base_model(temporary, downloader=downloader)
            remote_calls = [call for call in observed if call["local_files_only"] is False]
            self.assertEqual(len(remote_calls), len(BOOKNLP_HF_MODEL_IDS))
            self.assertTrue(all(call["hf_home"] == str(Path(temporary).resolve()) for call in observed))
            self.assertTrue(all(call["hf_hub_offline"] == "0" for call in observed))
            self.assertTrue(all(call["transformers_offline"] == "0" for call in observed))
        for name, value in original_environment.items():
            self.assertEqual(os.environ.get(name), value)

    def test_setup_reports_missing_assets_without_network_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "snapshot"
            snapshot.mkdir()
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(BookNLPBaseModelSetupError, "L-4.*pytorch_model.bin, vocab.txt"):
                prepare_booknlp_base_model(
                    temporary,
                    downloader=lambda **_kwargs: str(snapshot),
                    local_only=True,
                )

    def test_explicit_setup_fetches_only_missing_component_snapshots(self):
        with tempfile.TemporaryDirectory() as temporary:
            cache_home = Path(temporary)
            cached = cache_home / "cached-l4"
            cached.mkdir()
            for filename in REQUIRED_BASE_MODEL_FILES:
                (cached / filename).write_bytes(b"existing")
            fetched = cache_home / "fetched"
            fetched.mkdir()
            for filename in REQUIRED_BASE_MODEL_FILES:
                (fetched / filename).write_bytes(b"new")
            remote_repos: list[str] = []

            def downloader(**kwargs: object) -> str:
                repo_id = str(kwargs["repo_id"])
                if kwargs["local_files_only"]:
                    if repo_id == BOOKNLP_HF_MODEL_IDS[0]:
                        return str(cached)
                    raise OSError("not cached")
                remote_repos.append(repo_id)
                return str(fetched)

            files = prepare_booknlp_base_model(cache_home, downloader=downloader)
            self.assertEqual(remote_repos, list(BOOKNLP_HF_MODEL_IDS[1:]))
            self.assertEqual(len(files), len(REQUIRED_BASE_MODEL_FILES) * 3)

    def test_runner_fails_actionably_when_base_model_cache_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in BOOKNLP_SMALL_MODEL_FILES:
                (directory / filename).write_bytes(b"local asset")
            with patch(
                "lector_rag_api.booknlp_base_model.prepare_booknlp_base_model",
                side_effect=BookNLPBaseModelSetupError("missing vocab.txt"),
            ):
                with self.assertRaisesRegex(BookNLPUnavailableError, "booknlp_base_model"):
                    LocalBookNLPRunner(model_path=str(directory)).process("A test.", "book")

    def test_huggingface_environment_is_process_local_and_configurable(self):
        with tempfile.TemporaryDirectory() as temporary, patch.dict(os.environ, {}, clear=True):
            configured = configure_huggingface_environment(temporary, offline=True)
            self.assertEqual(configured, Path(temporary).resolve())
            self.assertEqual(os.environ["HF_HOME"], str(configured))
            self.assertEqual(os.environ["HF_HUB_CACHE"], str(configured / "hub"))
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
            self.assertEqual(os.environ["TRANSFORMERS_OFFLINE"], "1")

    def test_environment_override_is_resolved(self):
        with tempfile.TemporaryDirectory() as temporary:
            override = Path(temporary) / "assets"
            with patch.dict(os.environ, {"LECTOR_BOOKNLP_MODEL_DIR": str(override)}):
                runner = LocalBookNLPRunner()
            self.assertEqual(Path(runner.model_path), override.resolve())

    def test_existing_valid_assets_are_not_downloaded_again(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for name in BOOKNLP_SMALL_MODEL_FILES:
                (directory / name).write_bytes(b"valid")
            downloads: list[str] = []
            result = prepare_booknlp_models(
                directory,
                downloader=lambda url, target: downloads.append(url),
                validator=lambda path: self.assertEqual(path.read_bytes(), b"valid"),
            )
            self.assertEqual(len(result), 3)
            self.assertEqual(downloads, [])

    def test_missing_assets_download_to_staging_then_install(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "lector" / "models" / "booknlp"
            downloaded: list[str] = []

            def download(url: str, target: Path) -> None:
                downloaded.append(url)
                target.write_bytes(b"valid:" + target.name.encode())

            files = prepare_booknlp_models(
                directory,
                downloader=download,
                validator=lambda path: self.assertTrue(path.read_bytes().startswith(b"valid:")),
            )
            self.assertTrue(directory.is_dir())
            self.assertEqual(len(downloaded), 3)
            self.assertTrue(all(path.is_file() for path in files))
            self.assertFalse(list(directory.glob("*.part")))

    def test_interrupted_download_leaves_no_partial_asset(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)

            def interrupted(_url: str, target: Path) -> None:
                target.write_bytes(b"partial")
                raise OSError("connection interrupted")

            with self.assertRaisesRegex(BookNLPModelSetupError, "Could not install"):
                prepare_booknlp_models(
                    directory, downloader=interrupted, validator=lambda _path: None
                )
            self.assertEqual(list(directory.iterdir()), [])

    def test_invalid_existing_file_is_replaced_only_after_valid_download(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            first = directory / BOOKNLP_SMALL_MODEL_FILES[0]
            first.write_bytes(b"corrupt")
            calls = 0

            def download(_url: str, target: Path) -> None:
                nonlocal calls
                calls += 1
                target.write_bytes(b"valid")

            def validate(path: Path) -> None:
                if path.read_bytes() != b"valid":
                    raise ValueError("bad weights")

            prepare_booknlp_models(directory, downloader=download, validator=validate)
            self.assertEqual(first.read_bytes(), b"valid")
            self.assertEqual(calls, 3)

    def test_runner_passes_exact_assets_to_booknlp_custom_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            for filename in BOOKNLP_SMALL_MODEL_FILES:
                (directory / filename).write_bytes(b"local asset")
            captured: dict[str, object] = {}
            original_builtin_open = builtins.open

            class FakePipeline:
                def __init__(self, language: str, params: dict[str, object]):
                    captured["language"] = language
                    captured["params"] = params

                def process(self, input_file: str, output: str, book_id: str) -> None:
                    with english_implementation.open(input_file) as source:
                        captured["input_text"] = source.read()
                    with gender_implementation.open(hyperparameter_path) as source:
                        captured["gender_text"] = source.read()
                    binary_path = Path(input_file).with_suffix(".bin")
                    binary_path.write_bytes(b"\x00\x9d\xff")
                    with english_implementation.open(binary_path, "rb") as source:
                        captured["binary_data"] = source.read()
                    target = Path(output)
                    (target / f"{book_id}.book").write_text('{"characters": []}', encoding="utf-8")
                    (target / f"{book_id}.entities").write_text("COREF\n", encoding="utf-8")
                    (target / f"{book_id}.tokens").write_text("token_ID_within_document\n", encoding="utf-8")

            package = types.ModuleType("booknlp")
            package.__path__ = []  # type: ignore[attr-defined]
            english = types.ModuleType("booknlp.english")
            english.__path__ = []  # type: ignore[attr-defined]
            english_implementation = types.ModuleType("booknlp.english.english_booknlp")
            gender_implementation = types.ModuleType("booknlp.english.gender_inference_model_1")
            common = types.ModuleType("booknlp.common")
            common.__path__ = []  # type: ignore[attr-defined]
            reader_modules = {
                name: types.ModuleType(name) for name in _BOOKNLP_TEXT_READER_MODULES
            }
            hyperparameter_path = directory / "gender-hyperparameters.tsv"
            hyperparameter_text = "name\tvalue\nÉlodie\t“café”—Tess’s\n"
            hyperparameter_path.write_text(hyperparameter_text, encoding="utf-8")

            def windows_default_open(path: str, mode: str = "r", *args: object, **kwargs: object):
                # Simulate Windows' cp1252 locale when BookNLP omits encoding.
                if "b" in mode:
                    return builtins.open(path, mode, *args, **kwargs)
                if "encoding" not in kwargs and len(args) < 2:
                    kwargs["encoding"] = "cp1252"
                return builtins.open(path, mode, *args, **kwargs)

            english_implementation.open = windows_default_open  # type: ignore[attr-defined]
            gender_implementation.open = windows_default_open  # type: ignore[attr-defined]
            reader_modules["booknlp.english.english_booknlp"] = english_implementation
            reader_modules["booknlp.english.gender_inference_model_1"] = gender_implementation
            for name, module in reader_modules.items():
                if name not in {"booknlp.english.english_booknlp", "booknlp.english.gender_inference_model_1"}:
                    module.open = windows_default_open  # type: ignore[attr-defined]
            original_reader_opens = {
                name: module.open for name, module in reader_modules.items()
            }
            implementation = types.ModuleType("booknlp.booknlp")
            implementation.BookNLP = FakePipeline  # type: ignore[attr-defined]
            huggingface_hub = types.ModuleType("huggingface_hub")
            snapshot = directory / "hf-snapshot"
            snapshot.mkdir()
            for filename in REQUIRED_BASE_MODEL_FILES:
                (snapshot / filename).write_bytes(b"model file")

            def fake_snapshot_download(**_kwargs: object) -> str:
                return str(snapshot)

            huggingface_hub.snapshot_download = fake_snapshot_download  # type: ignore[attr-defined]
            original_import = __import__

            def checked_import(name: str, *args: object, **kwargs: object):
                if name in {"huggingface_hub", "booknlp.booknlp"}:
                    self.assertEqual(os.environ.get("HF_HOME"), str(default_huggingface_home()))
                    self.assertEqual(
                        os.environ.get("HF_HUB_CACHE"),
                        str(default_huggingface_home() / "hub"),
                    )
                    self.assertEqual(os.environ.get("HF_HUB_OFFLINE"), "1")
                    self.assertEqual(os.environ.get("TRANSFORMERS_OFFLINE"), "1")
                return original_import(name, *args, **kwargs)

            with patch.dict(sys.modules, {
                "booknlp": package,
                "booknlp.english": english,
                "booknlp.common": common,
                **reader_modules,
                "booknlp.booknlp": implementation,
                "huggingface_hub": huggingface_hub,
            }), \
                 patch("lector_rag_api.booknlp_compat.booknlp_state_dict_compatibility", return_value=nullcontext()), \
                 patch("builtins.__import__", side_effect=checked_import):
                unicode_text = "“Tess”—Élodie’s café: ‘déjà vu!’"
                result = LocalBookNLPRunner(model_path=str(directory)).process(unicode_text, "book")

            params = captured["params"]
            self.assertIsInstance(params, dict)
            assert isinstance(params, dict)
            self.assertEqual(params["model"], "custom")
            self.assertEqual(params["entity_model_path"], (directory / BOOKNLP_SMALL_MODEL_FILES[0]).as_posix())
            self.assertEqual(params["coref_model_path"], (directory / BOOKNLP_SMALL_MODEL_FILES[1]).as_posix())
            self.assertEqual(params["quote_attribution_model_path"], (directory / BOOKNLP_SMALL_MODEL_FILES[2]).as_posix())
            self.assertEqual(result["entities"], [])
            self.assertEqual(captured["input_text"], unicode_text)
            self.assertEqual(captured["gender_text"], hyperparameter_text)
            self.assertEqual(captured["binary_data"], b"\x00\x9d\xff")
            for name, module in reader_modules.items():
                self.assertIs(module.open, original_reader_opens[name])
            self.assertIs(builtins.open, original_builtin_open)

    def test_failed_graph_job_can_be_queued_again_after_assets_are_ready(self):
        class ReadyRunner:
            name = "booknlp"
            version = "booknlp-small-test"

        service = InMemoryRagService(booknlp_runner=ReadyRunner())
        self.addCleanup(service.close)
        units = [SourceUnitInput(
            source_unit_id="unit", section_index=0, unit_index=0,
            kind="paragraph", text="A short story.",
        )]
        chunks = service.ingest_book("retry-book", "v1", units)
        service.store.schedule_book_preparation(
            book_id="retry-book", content_version_id="v1",
            source_units=[unit.model_dump(mode="json") for unit in units],
            graph_rag_enabled=True,
        )
        service.store.schedule_booknlp_job(
            book_id="retry-book", content_version_id="v1",
            total_chunks=len(chunks), extraction_version=ReadyRunner.version,
        )
        service.store.set_booknlp_job_status(
            book_id="retry-book", content_version_id="v1",
            status="unavailable", stage="unavailable", error="assets missing",
        )
        with patch.object(service, "start_booknlp_worker"):
            result = service.schedule_full_book_story_graph(
                book_id="retry-book", content_version_id="v1", retry=True,
            )
        self.assertEqual(result["status"], "queued")
        self.assertIsNone(result["error"])


if __name__ == "__main__":
    unittest.main()
