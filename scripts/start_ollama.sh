#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$ROOT_DIR/.ollama/models}"
export OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-true}"

LOCAL_OLLAMA_BIN="$ROOT_DIR/.local/ollama/ollama"
if [[ -x "${OLLAMA_BIN:-}" ]]; then
  OLLAMA_BIN="$OLLAMA_BIN"
elif [[ -x "$LOCAL_OLLAMA_BIN" ]]; then
  OLLAMA_BIN="$LOCAL_OLLAMA_BIN"
elif command -v ollama >/dev/null 2>&1; then
  OLLAMA_BIN="$(command -v ollama)"
else
  echo "Ollama binary not found."
  echo "Install Ollama or place the binary at $LOCAL_OLLAMA_BIN"
  exit 1
fi

if [[ "${READABLE_FORCE_CPU:-0}" == "1" ]]; then
  export OLLAMA_LLM_LIBRARY="${OLLAMA_LLM_LIBRARY:-cpu}"
  export OLLAMA_IGPU_ENABLE="${OLLAMA_IGPU_ENABLE:-false}"
  export LLAMA_ARG_N_GPU_LAYERS="${LLAMA_ARG_N_GPU_LAYERS:-0}"
  export LLAMA_ARG_FIT="${LLAMA_ARG_FIT:-off}"
else
  if [[ "${OLLAMA_LLM_LIBRARY:-}" == "cpu" ]]; then
    unset OLLAMA_LLM_LIBRARY
  fi
  if [[ "${LLAMA_ARG_DEVICE:-}" == "none" ]]; then
    unset LLAMA_ARG_DEVICE
  fi
  if [[ "${LLAMA_ARG_N_GPU_LAYERS:-}" == "0" ]]; then
    unset LLAMA_ARG_N_GPU_LAYERS
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    export OLLAMA_IGPU_ENABLE="${OLLAMA_IGPU_ENABLE:-true}"
    export LLAMA_ARG_FIT="${LLAMA_ARG_FIT:-on}"
  fi
fi

mkdir -p "$OLLAMA_MODELS"

OLLAMA_URL="http://$OLLAMA_HOST"
if command -v curl >/dev/null 2>&1 && curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama is already running at $OLLAMA_URL"
  echo "If it was started before this GPU setting change, stop it and run ./scripts/start_ollama.sh again."
  exit 0
fi

echo "Starting Ollama with automatic GPU detection. Set READABLE_FORCE_CPU=1 to force CPU mode."
echo "Using: $OLLAMA_BIN"
exec "$OLLAMA_BIN" serve
