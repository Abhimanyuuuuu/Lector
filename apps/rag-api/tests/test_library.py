from __future__ import annotations

import hashlib
import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from fastapi.testclient import TestClient

from lector_rag_api.library import InvalidEpubUpload, save_epub_upload
from lector_rag_api.main import create_app


def epub_bytes() -> bytes:
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as archive:
        archive.writestr("mimetype", "application/epub+zip")
        archive.writestr("META-INF/container.xml", "<container/>")
    return data.getvalue()


class LibraryTests(unittest.TestCase):
    def test_stream_save_hashes_and_uses_generated_filename(self):
        import asyncio

        content = epub_bytes()
        version = "sha256:" + hashlib.sha256(content).hexdigest()

        async def pieces():
            yield content[:10]
            yield content[10:]

        with tempfile.TemporaryDirectory() as directory:
            path, key, size = asyncio.run(
                save_epub_upload(pieces(), book_id="../a", content_version_id=version,
                                 directory=Path(directory))
            )
            self.assertEqual(path.parent, Path(directory).resolve())
            self.assertEqual(path.name, key)
            self.assertEqual(size, len(content))
            self.assertEqual(path.read_bytes(), content)

    def test_rejects_mismatched_fingerprint_and_non_zip(self):
        import asyncio

        async def upload(value):
            yield value

        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(InvalidEpubUpload):
                asyncio.run(save_epub_upload(upload(b"not an epub"), book_id="b",
                    content_version_id="sha256:" + "0" * 64, directory=Path(directory)))

    def test_api_upload_library_and_download_survive_app_restart(self):
        content = epub_bytes()
        version = "sha256:" + hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            db_path = Path(directory) / "db.sqlite3"
            epub_dir = Path(directory) / "epubs"
            previous = os.environ.get("LECTOR_EPUB_STORAGE_DIR")
            os.environ["LECTOR_EPUB_STORAGE_DIR"] = str(epub_dir)
            try:
                first = TestClient(create_app(database_path=str(db_path)))
                upload = first.put(
                    "/api/v1/books/book-a/epub",
                    params={"content_version_id": version, "title": "Book A", "original_filename": "a.epub"},
                    content=content,
                    headers={"content-type": "application/epub+zip"},
                )
                self.assertEqual(upload.status_code, 200, upload.text)
                self.assertEqual(upload.json()["indexing_status"], "uploaded")
                indexed = first.post(
                    "/api/v1/books/book-a/source-units",
                    json={"content_version_id": version, "source_units": [{
                        "source_unit_id": "u1", "section_index": 0, "unit_index": 0,
                        "kind": "paragraph", "text": "A persistent passage.",
                    }]},
                )
                self.assertEqual(indexed.status_code, 202, indexed.text)
                library = first.get("/api/v1/books")
                self.assertEqual(library.status_code, 200)
                self.assertTrue(library.json()["books"][0]["file_available"])
                self.assertEqual(library.json()["books"][0]["indexing_status"], "indexed")
                first.app.state.rag_service.close()
                first.close()

                second = TestClient(create_app(database_path=str(db_path)))
                self.assertEqual(second.get("/api/v1/books").json()["books"][0]["content_version_id"], version)
                downloaded = second.get("/api/v1/books/book-a/epub")
                self.assertEqual(downloaded.status_code, 200)
                self.assertEqual(downloaded.content, content)
                next(epub_dir.glob("*.epub")).unlink()
                self.assertFalse(second.get("/api/v1/books").json()["books"][0]["file_available"])
                missing = second.get("/api/v1/books/book-a/epub")
                self.assertEqual(missing.status_code, 410)
                self.assertEqual(missing.json()["error"]["code"], "saved_epub_missing")
                second.app.state.rag_service.close()
                second.close()
            finally:
                if previous is None:
                    os.environ.pop("LECTOR_EPUB_STORAGE_DIR", None)
                else:
                    os.environ["LECTOR_EPUB_STORAGE_DIR"] = previous


if __name__ == "__main__":
    unittest.main()
