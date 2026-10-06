#!/usr/bin/env python3
"""Run sequential multipart ASR requests per client and report latency percentiles."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
import urllib.error
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def make_body(audio: bytes, filename: str, language: str) -> tuple[bytes, str]:
    boundary = f"----asr-{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{language}\r\n".encode(),
        (
            f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"{Path(filename).name}\"\r\nContent-Type: application/octet-stream\r\n\r\n"
        ).encode(),
        audio,
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Backend origin, for example https://api.qa.example.com")
    parser.add_argument("--token", default=None, help="API token; defaults to ASR_API_TOKEN environment variable")
    parser.add_argument("--audio", required=True, type=Path, help="Local WAV/MP3/M4A/etc. test audio")
    parser.add_argument("--language", choices=("auto", "vi", "en"), default="vi")
    parser.add_argument("--ccu", type=int, nargs="+", default=[1, 4, 8, 12, 16, 20])
    parser.add_argument("--requests-per-client", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    import os

    token = args.token or os.environ.get("ASR_API_TOKEN", "")
    if not token:
        parser.error("provide --token or set ASR_API_TOKEN")
    if not args.audio.is_file():
        parser.error(f"audio file not found: {args.audio}")
    if not args.ccu or any(level < 1 or level > 20 for level in args.ccu):
        parser.error("CCU levels must be between 1 and 20")
    if args.requests_per_client < 1:
        parser.error("--requests-per-client must be at least 1")

    audio = args.audio.read_bytes()
    base_url = args.url.rstrip("/")
    all_rows: list[dict] = []

    def request_one(client: int, sequence: int) -> dict:
        body, content_type = make_body(audio, args.audio.name, args.language)
        request = urllib.request.Request(
            f"{base_url}/api/transcribe",
            data=body,
            headers={"Content-Type": content_type, "X-ASR-Token": token},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
                return {
                    "ok": response.status == 200,
                    "elapsed_s": time.perf_counter() - started,
                    "inference_s": float(payload.get("inference_s", 0)),
                    "queue_s": float(payload.get("queue_s", 0)),
                    "rtf": float(payload.get("rtf", 0)),
                    "error": "" if response.status == 200 else f"HTTP {response.status}",
                    "client": client,
                    "sequence": sequence,
                }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, ValueError) as exc:
            detail = getattr(exc, "reason", exc)
            return {
                "ok": False,
                "elapsed_s": time.perf_counter() - started,
                "inference_s": 0,
                "queue_s": 0,
                "rtf": 0,
                "error": f"{type(exc).__name__}: {detail}",
                "client": client,
                "sequence": sequence,
            }

    print(f"audio={args.audio} bytes={len(audio)} requests/client={args.requests_per_client}")
    for ccu in args.ccu:
        rows: list[dict] = []
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=ccu) as pool:
            futures = [
                pool.submit(request_one, client, sequence)
                for client in range(ccu)
                for sequence in range(args.requests_per_client)
            ]
            for future in as_completed(futures):
                rows.append(future.result())
        elapsed = time.perf_counter() - started
        successes = [row for row in rows if row["ok"]]
        latencies = [row["elapsed_s"] for row in successes]
        failures = [row for row in rows if not row["ok"]]
        all_rows.extend({"ccu": ccu, **row} for row in rows)
        print(
            f"CCU={ccu:>2} requests={len(rows):>3} ok={len(successes):>3} "
            f"errors={len(failures):>3} throughput={len(successes) / elapsed:.2f} req/s "
            f"p50={percentile(latencies, .50):.3f}s p95={percentile(latencies, .95):.3f}s "
            f"max={max(latencies, default=0):.3f}s "
            f"mean_inference={statistics.mean([r['inference_s'] for r in successes]):.3f}s"
            if successes else
            f"CCU={ccu:>2} requests={len(rows):>3} ok=0 errors={len(failures):>3}"
        )
        for row in failures[:3]:
            print(f"  error client={row['client']} request={row['sequence']}: {row['error']}")

    print(f"completed_requests={len(all_rows)} success={sum(row['ok'] for row in all_rows)}")
    return 0 if all(row["ok"] for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
