#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${ASR_PYTHON:-/home/phongnth/asr-bestmodel/.venv/bin/python}"

if [[ ! -x "$PY" ]]; then
  echo "Python environment not found: $PY" >&2
  echo "Set ASR_PYTHON to an environment with faster-whisper, FastAPI, and Uvicorn." >&2
  exit 1
fi
if [[ -z "${ASR_API_TOKEN:-}" ]]; then
  echo "Set ASR_API_TOKEN before starting the API." >&2
  exit 2
fi

CUDA_LIB_DIRS="$("$PY" -c 'import os, nvidia.cublas.lib, nvidia.cudnn.lib, nvidia.cuda_nvrtc.lib; print(os.pathsep.join((nvidia.cublas.lib.__path__[0], nvidia.cudnn.lib.__path__[0], nvidia.cuda_nvrtc.lib.__path__[0])))')"
export LD_LIBRARY_PATH="$CUDA_LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m asr_api.main
