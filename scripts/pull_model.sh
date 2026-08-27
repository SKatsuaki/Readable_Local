#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:-qwen2.5:3b-instruct}"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$ROOT_DIR/.ollama/models}"

mkdir -p "$OLLAMA_MODELS"
"$ROOT_DIR/.local/ollama/ollama" pull "$MODEL"
