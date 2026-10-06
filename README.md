# ASR QA API

FastAPI test service for the locally cached `faster-whisper-large-v3` CTranslate2 checkpoint. It serves `/healthz` and an authenticated multipart endpoint at `/api/transcribe`. It never downloads model weights, writes uploaded audio, or stores transcripts. The model stays resident in one API process; upload and inference concurrency are capped.

## Run on the existing GPU host

This host already has the model and its CUDA-enabled ASR virtualenv. From this directory:

```bash
cp .env.example .env
set -a && source .env && set +a
export ASR_API_TOKEN="$(/home/phongnth/asr-bestmodel/.venv/bin/python -c 'import secrets; print(secrets.token_urlsafe(32))')"
./run.sh
```

The launcher reuses `/home/phongnth/asr-bestmodel/.venv`; it does not create another 2.9 GB CUDA environment. `ASR_MODEL_CACHE` points at the retained local checkpoint. To use another checkpoint location, set `ASR_MODEL_PATH` to a local CTranslate2 snapshot containing `model.bin` and `config.json`. A missing model fails startup rather than downloading anything.

By default the API binds to `127.0.0.1:8767`. Put it behind the host's HTTPS reverse proxy and access controls. Keep the API token private and send it only to QA testers. The API has one Uvicorn worker because each worker would load another model copy into GPU memory.

## Connect the Vercel QA frontend

Deploy [fe_test_asr](https://github.com/pot030321/fe_test_asr) as a static Vercel project, then enter the backend HTTPS origin and API token in the UI. On the backend, set:

```bash
ASR_ALLOWED_ORIGINS=https://your-project.vercel.app
ASR_ALLOWED_ORIGIN_REGEX='^https://[a-z0-9-]+(-[a-z0-9-]+)?\.vercel\.app$'
```

Replace the exact origin with your QA deployment. The regex permits Vercel preview hostnames; remove it if previews should not call the API. Vercel and the tester's browser cannot reach a private `192.168.x.x` server address over the public internet. The backend needs an HTTPS route reachable from the browser, or a VPN/private connectivity arrangement. The browser uses a custom token header, so the API's CORS preflight must be allowed.

## API contract

`POST /api/transcribe` accepts multipart form fields `file` and `language` (`auto`, `vi`, or `en`) with `X-ASR-Token`. It returns the transcript, detected language, audio duration, inference time, server request time, queue wait, file size, and RTF. `request_s` is measured inside the API; browser E2E additionally includes upload and network time.

Example:

```bash
curl -H "X-ASR-Token: $ASR_API_TOKEN" \
  -F 'file=@sample.wav' -F 'language=vi' \
  https://api.example.com/api/transcribe
```

## Tests

The unit tests mock the model and validate auth, multipart parsing, timing output, health, and Vercel CORS. Run them with the existing virtualenv:

```bash
PYTHONPATH=src /home/phongnth/asr-bestmodel/.venv/bin/python -m unittest discover -s tests -v
```

To measure real HTTP request concurrency from a machine that has an audio sample and can reach the API:

```bash
ASR_API_TOKEN='...' python scripts/ccu_test.py \
  --url https://api.example.com --audio ./sample.wav --language vi \
  --ccu 1 4 8 12 16 20 --requests-per-client 5
```

This sends real inference traffic. Start with CCU 1, then increase gradually while watching GPU memory, server queue time, p95 latency, and error rate. The script caps a stage at 20 clients and does not upload or retain the sample in the repository.
