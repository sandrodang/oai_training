#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HOME="$PWD/hf"
uv sync
echo "OK: .venv ready. Chay script bang: uv run python src/<script>.py"
