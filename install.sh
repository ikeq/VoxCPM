#!/usr/bin/env bash
# Install voxcpm locally using uv (https://docs.astral.sh/uv/).
#
# Why uv:
# - Manages its own Python (no brew Python / libexpat issues)
# - Global wheel cache shared across projects (torch/numba downloaded ONCE)
# - Reproducible from uv.lock — same versions on every machine
# - 10-30x faster than pip
#
# Usage on a fresh Mac:
#   brew install uv
#   git clone https://github.com/ikeq/VoxCPM.git
#   cd VoxCPM
#   ./install.sh
#   ./run.sh "你好"

set -euo pipefail

cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "error: uv not found. Install with: brew install uv" >&2
  echo "       or: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  exit 1
fi

echo "[install] uv: $(uv --version)"
echo "[install] syncing project (uses uv.lock + .python-version) ..."

# uv sync will:
#  - read .python-version, install that Python if missing (cached globally)
#  - create .venv if missing
#  - install all deps from uv.lock into .venv (using the global wheel cache)
uv sync

echo
echo "[install] done. Next: ./run.sh \"your text here\""
echo "[install] first run will download ~4.5GB model to ./pretrained_models/VoxCPM2"
