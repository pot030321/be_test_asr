"""Small FastAPI wrapper around a locally cached faster-whisper model.

The API deliberately loads a model path from disk and never asks Hugging Face to
download missing weights. It is intended for a private QA endpoint behind TLS.
"""

from __future__ import annotations

import asyncio
import hmac
import io
import os
import re
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool


DEFAULT_MODEL_CACHE = Path(
    "/home/phongnth/asr-bestmodel/models/faster-whisper-large-v3/weights"
)
MODEL_SUBPATH = Path(
    "models--Systran--faster-whisper-large-v3/snapshots"
)
LANGUAGE_NAMES = {"vi": "Tiếng Việt", "en": "English"}


def _positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value < 1:
        raise RuntimeError(f"{name} must be at least 1")
    return value


@dataclass(frozen=True)
class Settings:
    token: str
    model_path: str
    model_cache: Path
    device: str
    compute_type: str
    model_workers: int
    max_inflight: int
    max_audio_bytes: int
    queue_timeout_s: float
    allowed_origins: tuple[str, ...]
    allowed_origin_regex: str | None

    @classmethod
    def from_env(cls) -> "Settings":
        explicit_path = os.environ.get("ASR_MODEL_PATH", "").strip()
        cache_value = os.environ.get("ASR_MODEL_CACHE", str(DEFAULT_MODEL_CACHE))
        regex = os.environ.get("ASR_ALLOWED_ORIGIN_REGEX", "").strip() or None
        if regex:
            try:
                re.compile(regex)
            except re.error as exc:
                raise RuntimeError("ASR_ALLOWED_ORIGIN_REGEX is invalid") from exc
        origins = tuple(
            item.strip().rstrip("/")
            for item in os.environ.get("ASR_ALLOWED_ORIGINS", "").split(",")
            if item.strip()
        )
        try:
            queue_timeout = float(os.environ.get("ASR_QUEUE_TIMEOUT_S", "60"))
        except ValueError as exc:
            raise RuntimeError("ASR_QUEUE_TIMEOUT_S must be numeric") from exc
        if queue_timeout <= 0:
            raise RuntimeError("ASR_QUEUE_TIMEOUT_S must be greater than zero")

        return cls(
            token=os.environ.get("ASR_API_TOKEN", "").strip(),
            model_path=explicit_path,
            model_cache=Path(cache_value).expanduser(),
            device=os.environ.get("ASR_DEVICE", "cuda").strip(),
            compute_type=os.environ.get("ASR_COMPUTE_TYPE", "float16").strip(),
            model_workers=_positive_int("ASR_MODEL_WORKERS", 4),
            max_inflight=_positive_int("ASR_MAX_INFLIGHT", 2),
            max_audio_bytes=_positive_int("ASR_MAX_AUDIO_MB", 100) * 1024 * 1024,
            queue_timeout_s=queue_timeout,
            allowed_origins=origins,
            allowed_origin_regex=regex,
        )


def resolve_model_path(settings: Settings) -> Path:
    """Return a verified local model snapshot; never resolve a remote model ID."""
    if settings.model_path:
        candidate = Path(settings.model_path).expanduser()
        if _is_model_snapshot(candidate):
            return candidate
        raise RuntimeError(f"ASR_MODEL_PATH is not a valid local checkpoint: {candidate}")

    snapshots = settings.model_cache / MODEL_SUBPATH
    for model_file in sorted(snapshots.glob("*/model.bin"), reverse=True):
        candidate = model_file.parent
        if _is_model_snapshot(candidate):
            return candidate
    raise RuntimeError(
        "Local faster-whisper large-v3 weights not found. Set ASR_MODEL_PATH or "
        f"ASR_MODEL_CACHE to the existing model cache (looked under {snapshots}); "
        "this service does not download weights."
    )


def _is_model_snapshot(path: Path) -> bool:
    return (path / "model.bin").is_file() and (path / "config.json").is_file()


def load_model(settings: Settings) -> Any:
    from faster_whisper import WhisperModel

    snapshot = resolve_model_path(settings)
    return WhisperModel(
        str(snapshot),
        device=settings.device,
        compute_type=settings.compute_type,
        num_workers=settings.model_workers,
    )


def parse_multipart(content_type: str, payload: bytes, max_audio_bytes: int) -> tuple[bytes, str]:
    try:
        header = (
            f"Content-Type: {content_type}\r\nMIME-Version: 1.0\r\n\r\n"
        ).encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Invalid multipart Content-Type") from exc
    message = BytesParser(policy=policy.default).parsebytes(header + payload)
    if not message.is_multipart():
        raise ValueError("Expected multipart/form-data")

    audio: bytes | None = None
    language: str | None = None
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        if name == "file":
            audio = part.get_payload(decode=True)
        elif name == "language":
            language = str(part.get_content()).strip()

    if not audio:
        raise ValueError("Audio file is missing or empty")
    if len(audio) > max_audio_bytes:
        raise ValueError(f"Audio file exceeds the {max_audio_bytes // (1024 * 1024)} MB limit")
    if language not in {"auto", "vi", "en"}:
        raise ValueError("Language must be auto, vi, or en")
    return audio, language


def transcribe(model: Any, audio: bytes, language: str) -> dict[str, Any]:
    started = time.perf_counter()
    segments, info = model.transcribe(
        io.BytesIO(audio),
        language=None if language == "auto" else language,
        task="transcribe",
        beam_size=5,
        vad_filter=False,
        condition_on_previous_text=False,
    )
    text = "".join(segment.text for segment in segments).strip()
    inference_s = time.perf_counter() - started
    audio_s = float(info.duration or 0.0)
    probability = getattr(info, "language_probability", None)
    return {
        "text": text,
        "language": info.language,
        "language_probability": float(probability) if probability is not None else None,
        "audio_s": audio_s,
        "inference_s": inference_s,
        "rtf": inference_s / audio_s if audio_s > 0 else 0.0,
    }


def create_app(
    settings: Settings | None = None,
    model_loader: Callable[[Settings], Any] = load_model,
) -> FastAPI:
    config = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not config.token:
            raise RuntimeError("Set ASR_API_TOKEN before starting the API")
        app.state.model = model_loader(config)
        app.state.inflight = asyncio.Semaphore(config.max_inflight)
        print(
            "Loaded local faster-whisper checkpoint; "
            f"device={config.device}, max_inflight={config.max_inflight}, "
            f"model_workers={config.model_workers}",
            flush=True,
        )
        try:
            yield
        finally:
            del app.state.model

    app = FastAPI(
        title="ASR QA API",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(config.allowed_origins),
        allow_origin_regex=config.allowed_origin_regex,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-ASR-Token"],
        max_age=600,
    )

    @app.get("/healthz")
    async def health() -> dict[str, str]:
        return {"status": "ready", "model": "faster-whisper-large-v3"}

    @app.post("/api/transcribe")
    async def api_transcribe(request: Request) -> dict[str, Any]:
        request_id = uuid.uuid4().hex[:12]
        supplied = request.headers.get("X-ASR-Token", "")
        if not hmac.compare_digest(supplied, config.token):
            raise HTTPException(status_code=401, detail="Access token is invalid.")

        content_type = request.headers.get("content-type", "")
        if not content_type.lower().startswith("multipart/form-data;"):
            raise HTTPException(status_code=415, detail="Expected multipart/form-data.")
        content_length = request.headers.get("content-length")
        max_request_bytes = config.max_audio_bytes + 1024 * 1024
        if content_length:
            try:
                if int(content_length) > max_request_bytes:
                    raise HTTPException(status_code=413, detail="Request exceeds the upload limit.")
            except ValueError as exc:
                raise HTTPException(status_code=400, detail="Invalid Content-Length.") from exc

        arrived = time.perf_counter()
        try:
            await asyncio.wait_for(
                request.app.state.inflight.acquire(),
                timeout=config.queue_timeout_s,
            )
        except TimeoutError as exc:
            raise HTTPException(status_code=429, detail="ASR queue is full; retry shortly.") from exc

        try:
            admitted = time.perf_counter()
            body_parts: list[bytes] = []
            body_size = 0
            async for chunk in request.stream():
                body_size += len(chunk)
                if body_size > max_request_bytes:
                    raise HTTPException(status_code=413, detail="Request exceeds the upload limit.")
                body_parts.append(chunk)
            try:
                audio, language = parse_multipart(content_type, b"".join(body_parts), config.max_audio_bytes)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

            inference_started = time.perf_counter()
            try:
                result = await run_in_threadpool(
                    transcribe, request.app.state.model, audio, language
                )
            except Exception as exc:
                raise HTTPException(
                    status_code=422,
                    detail=f"Audio could not be decoded or transcribed ({type(exc).__name__}).",
                ) from exc

            finished = time.perf_counter()
            return {
                **result,
                "language_name": LANGUAGE_NAMES.get(result["language"], result["language"]),
                "size_mb": len(audio) / (1024 * 1024),
                "queue_s": max(0.0, admitted - arrived),
                "request_s": finished - arrived,
                "server_s": finished - inference_started,
                "request_id": request_id,
            }
        finally:
            request.app.state.inflight.release()

    return app


app = create_app()


def main() -> None:
    import uvicorn

    host = os.environ.get("ASR_HOST", "127.0.0.1")
    port = _positive_int("ASR_PORT", 8767)
    if not os.environ.get("ASR_API_TOKEN", "").strip():
        raise SystemExit("Set ASR_API_TOKEN before starting the API")
    uvicorn.run(app, host=host, port=port, workers=1, access_log=False)


if __name__ == "__main__":
    main()
