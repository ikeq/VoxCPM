#!/usr/bin/env bash
# Run TTS inference locally. Usage:
#   ./run.sh "your text"
#   ./run.sh "your text" 2.0 10           # text, cfg_value, inference_timesteps
#   TEXT="..." OUTPUT_PATH=out.wav ./run.sh
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found. Install with: brew install uv" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  echo "error: .venv not found. Run ./install.sh first." >&2
  exit 1
fi

export TEXT="${1:-${TEXT:-(一位年轻女性，温柔甜美的声音)你好，欢迎使用 VoxCPM2！}}"
export CFG_VALUE="${2:-${CFG_VALUE:-2.0}}"
export INFERENCE_TIMESTEPS="${3:-${INFERENCE_TIMESTEPS:-10}}"
export OUTPUT_PATH="${OUTPUT_PATH:-demo.wav}"
export MODEL_DIR="${MODEL_DIR:-./pretrained_models/VoxCPM2}"

exec uv run python scripts/tts_run.py
