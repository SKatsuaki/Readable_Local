#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OLLAMA_DIR="$ROOT_DIR/.local/ollama"
LOCAL_OLLAMA_BIN="$OLLAMA_DIR/ollama"
LLAMA_SERVER="$OLLAMA_DIR/llama-server"
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
OLLAMA_URL="http://$OLLAMA_HOST"

if [[ -x "${OLLAMA_BIN:-}" ]]; then
  OLLAMA_BIN="$OLLAMA_BIN"
elif [[ -x "$LOCAL_OLLAMA_BIN" ]]; then
  OLLAMA_BIN="$LOCAL_OLLAMA_BIN"
elif command -v ollama >/dev/null 2>&1; then
  OLLAMA_BIN="$(command -v ollama)"
else
  OLLAMA_BIN=""
fi

echo "== Apple GPU =="
if command -v system_profiler >/dev/null 2>&1; then
  system_profiler SPDisplaysDataType | awk '
    /Chipset Model:/ || /Type:/ || /Total Number of Cores:/ || /Metal:/ {
      sub(/^[[:space:]]+/, "")
      print
    }
  '
else
  echo "system_profiler not found"
fi

echo
echo "== NVIDIA GPU =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
else
  echo "nvidia-smi not found"
fi

echo
echo "== Ollama =="
if [[ -n "$OLLAMA_BIN" ]]; then
  "$OLLAMA_BIN" --version
else
  echo "Ollama binary not found"
fi

echo
echo "== Local runner devices =="
if [[ -x "$LLAMA_SERVER" ]]; then
  devices="$("$LLAMA_SERVER" --list-devices 2>&1 || true)"
  echo "$devices"
  if echo "$devices" | grep -Eiq "Metal|CUDA|GPU|AGX|Apple|NVIDIA"; then
    echo "GPU device appears to be available to the local runner."
  else
    echo "WARNING: the local runner did not list a GPU. Ollama may fall back to CPU."
  fi
else
  echo "llama-server not found: $LLAMA_SERVER"
  echo "This is OK when using a system Ollama install."
fi

echo
echo "== Running server =="
if command -v curl >/dev/null 2>&1 && curl -fsS "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "Ollama is running at $OLLAMA_URL"
  echo "For detailed GPU startup logs, stop Ollama and run:"
  echo "  OLLAMA_DEBUG=1 ./scripts/start_ollama.sh"

  echo
  echo "== Loaded models =="
  if [[ -n "$OLLAMA_BIN" ]]; then
    "$OLLAMA_BIN" ps || true
  else
    echo "Ollama binary not found"
  fi

  echo
  echo "== Installed model formats =="
  tags_json="$(curl -fsS "$OLLAMA_URL/api/tags" 2>/dev/null || true)"
  if [[ -n "$tags_json" ]] && command -v python3 >/dev/null 2>&1; then
    TAGS_JSON="$tags_json" python3 - <<'PY'
import json
import os

data = json.loads(os.environ.get("TAGS_JSON", "{}") or "{}")
for model in data.get("models", []):
    details = model.get("details", {}) or {}
    name = model.get("name", "")
    model_format = details.get("format", "unknown")
    params = details.get("parameter_size", "unknown")
    quant = details.get("quantization_level", "unknown")
    print(f"{name}: format={model_format}, parameters={params}, quantization={quant}")
PY
  else
    echo "$tags_json"
  fi
else
  echo "Ollama is not responding at $OLLAMA_URL"
fi
