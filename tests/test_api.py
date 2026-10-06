from __future__ import annotations

import unittest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from asr_api.main import Settings, create_app


class FakeModel:
    def transcribe(self, audio, **kwargs):
        return iter([SimpleNamespace(text=" Xin chào thế giới.")]), SimpleNamespace(
            duration=1.25,
            language=kwargs.get("language") or "vi",
            language_probability=0.98,
        )


def multipart(audio: bytes = b"RIFFfake", language: str = "vi") -> tuple[bytes, str]:
    boundary = "test-boundary"
    payload = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n"
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"sample.wav\"\r\n"
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + audio + f"\r\n--{boundary}--\r\n".encode()
    return payload, f"multipart/form-data; boundary={boundary}"


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            token="test-secret",
            model_path="",
            model_cache=__import__("pathlib").Path("/unused"),
            device="cpu",
            compute_type="int8",
            model_workers=1,
            max_inflight=1,
            max_audio_bytes=1024,
            queue_timeout_s=1,
            allowed_origins=("https://qa.example.com",),
            allowed_origin_regex=r"^https://.*\.vercel\.app$",
        )
        self.client_context = TestClient(
            create_app(self.settings, model_loader=lambda _: FakeModel())
        )
        self.client = self.client_context.__enter__()

    def tearDown(self):
        self.client_context.__exit__(None, None, None)

    def test_health_reports_model(self):
        response = self.client.get("/healthz")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model"], "faster-whisper-large-v3")

    def test_transcribe_requires_token(self):
        body, content_type = multipart()
        response = self.client.post(
            "/api/transcribe",
            content=body,
            headers={"Content-Type": content_type},
        )
        self.assertEqual(response.status_code, 401)

    def test_transcribe_returns_transcript_and_timings(self):
        body, content_type = multipart()
        response = self.client.post(
            "/api/transcribe",
            content=body,
            headers={"Content-Type": content_type, "X-ASR-Token": "test-secret"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        result = response.json()
        self.assertEqual(result["text"], "Xin chào thế giới.")
        self.assertEqual(result["language"], "vi")
        self.assertAlmostEqual(result["audio_s"], 1.25)
        self.assertIn("inference_s", result)
        self.assertIn("request_s", result)
        self.assertIn("request_id", result)

    def test_invalid_language_is_rejected(self):
        body, content_type = multipart(language="fr")
        response = self.client.post(
            "/api/transcribe",
            content=body,
            headers={"Content-Type": content_type, "X-ASR-Token": "test-secret"},
        )
        self.assertEqual(response.status_code, 400)

    def test_vercel_origin_is_allowed(self):
        response = self.client.options(
            "/api/transcribe",
            headers={
                "Origin": "https://asr-qa-abc.vercel.app",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "x-asr-token,content-type",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "https://asr-qa-abc.vercel.app")


if __name__ == "__main__":
    unittest.main()
