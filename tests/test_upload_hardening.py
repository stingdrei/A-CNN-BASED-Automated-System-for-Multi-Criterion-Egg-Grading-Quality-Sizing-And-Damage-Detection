import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.main import UploadGuardMiddleware
from app.services import file_service as file_service_module
from app.services.file_service import FileService, validate_image_signature


class UploadGuardMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, headers, path="/api/v1/grade"):
        calls = {"app": 0}
        sent = []

        async def app(scope, receive, send):
            calls["app"] += 1

        async def send(message):
            sent.append(message)

        async def receive():
            return {"type": "http.request", "body": b"", "more_body": False}

        middleware = UploadGuardMiddleware(app)
        scope = {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": headers,
        }
        await middleware(scope, receive, send)
        return calls["app"], sent

    async def test_rejects_oversized_content_length_before_route(self):
        limit = 51 * 1024 * 1024
        called, sent = await self._run(
            [(b"content-length", str(limit).encode())]
        )

        self.assertEqual(called, 0)
        self.assertEqual(sent[0]["type"], "http.response.start")
        self.assertEqual(sent[0]["status"], 413)

    async def test_allows_within_limit(self):
        called, sent = await self._run([(b"content-length", b"1024")])

        self.assertEqual(called, 1)
        self.assertEqual(sent, [])

    async def test_ignores_other_paths(self):
        limit = 51 * 1024 * 1024
        called, _ = await self._run(
            [(b"content-length", str(limit).encode())],
            path="/api/v1/settings",
        )

        self.assertEqual(called, 1)


class ImageSignatureTests(unittest.TestCase):
    def test_rejects_mismatched_extension_and_content(self):
        with self.assertRaises(ValueError):
            validate_image_signature(b"GIF89a\x01\x00", ".jpg")

    def test_accepts_matching_jpeg_header(self):
        validate_image_signature(b"\xff\xd8\xff\xe0\x00\x10JFIF", ".jpg")

    def test_rejects_unknown_extension(self):
        with self.assertRaises(ValueError):
            validate_image_signature(b"\xff\xd8\xff", ".tiff")


class SaveFileSyncTests(unittest.TestCase):
    def test_disguised_upload_is_deleted_and_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                file_service_module.settings, "UPLOAD_DIR", tmp
            ), patch.object(
                file_service_module.settings, "MAX_FILE_SIZE_MB", 5
            ):
                service = FileService()
                upload = UploadFile(
                    file=io.BytesIO(b"GIF89a" + b"\x00" * 64),
                    filename="egg.jpg",
                )

                with self.assertRaises(ValueError):
                    service.save_file_sync(upload)

                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_valid_jpeg_is_saved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(
                file_service_module.settings, "UPLOAD_DIR", tmp
            ), patch.object(
                file_service_module.settings, "MAX_FILE_SIZE_MB", 5
            ):
                service = FileService()
                upload = UploadFile(
                    file=io.BytesIO(b"\xff\xd8\xff\xe0" + b"\x00" * 64),
                    filename="egg.jpg",
                )

                saved_path, file_type = service.save_file_sync(upload)

                self.assertEqual(file_type, "image")
                self.assertTrue(Path(saved_path).is_file())


if __name__ == "__main__":
    unittest.main()
