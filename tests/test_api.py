from __future__ import annotations

import unittest
import threading
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
        self.assertAlmostEqual(result["rtf"], result["inference_s"] / result["audio_s"])

    def test_metrics_requires_token(self):
        response = self.client.get("/api/metrics")
        self.assertEqual(response.status_code, 401)

    def test_metrics_tracks_requests_without_transcript(self):
        body, content_type = multipart()
        response = self.client.post(
            "/api/transcribe",
            content=body,
            headers={"Content-Type": content_type, "X-ASR-Token": "test-secret"},
        )
        self.assertEqual(response.status_code, 200, response.text)
        request_id = response.json()["request_id"]

        metrics = self.client.get(
            "/api/metrics", headers={"X-ASR-Token": "test-secret"}
        )
        self.assertEqual(metrics.status_code, 200, metrics.text)
        result = metrics.json()
        self.assertEqual(result["max_inflight"], 1)
        self.assertEqual(result["active_requests"], 0)
        self.assertEqual(result["total_requests"], 1)
        self.assertEqual(result["succeeded_requests"], 1)
        self.assertEqual(result["failed_requests"], 0)
        entry = result["recent_requests"][0]
        self.assertEqual(entry["request_id"], request_id)
        self.assertEqual(entry["status"], 200)
        self.assertIn("rtf", entry)
        self.assertNotIn("text", entry)

    def test_metrics_shows_inference_while_request_is_running(self):
        entered = threading.Event()
        release = threading.Event()

        class BlockingModel:
            def transcribe(self, audio, **kwargs):
                entered.set()
                if not release.wait(timeout=3):
                    raise TimeoutError("test inference release timed out")
                return iter([SimpleNamespace(text=" Xin chào.")]), SimpleNamespace(
                    duration=1.25,
                    language="vi",
                    language_probability=0.98,
                )

        body, content_type = multipart()
        with TestClient(
            create_app(self.settings, model_loader=lambda _: BlockingModel())
        ) as client:
            result = {}

            def transcribe():
                result["response"] = client.post(
                    "/api/transcribe",
                    content=body,
                    headers={"Content-Type": content_type, "X-ASR-Token": "test-secret"},
                )

            worker = threading.Thread(target=transcribe, daemon=True)
            worker.start()
            try:
                self.assertTrue(entered.wait(timeout=2))
                metrics = client.get(
                    "/api/metrics", headers={"X-ASR-Token": "test-secret"}
                ).json()
                self.assertEqual(metrics["active_requests"], 1)
                self.assertEqual(metrics["processing_requests"], 1)
                self.assertEqual(metrics["waiting_requests"], 0)
            finally:
                release.set()
                worker.join(timeout=3)

            self.assertFalse(worker.is_alive())
            self.assertEqual(result["response"].status_code, 200)

    def test_invalid_language_is_rejected(self):
        body, content_type = multipart(language="fr")
        response = self.client.post(
            "/api/transcribe",
            content=body,
            headers={"Content-Type": content_type, "X-ASR-Token": "test-secret"},
        )
        self.assertEqual(response.status_code, 400)
        metrics = self.client.get(
            "/api/metrics", headers={"X-ASR-Token": "test-secret"}
        ).json()
        self.assertEqual(metrics["failed_requests"], 1)
        self.assertEqual(metrics["recent_requests"][0]["status"], 400)

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
