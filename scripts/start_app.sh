#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUNDLED_PYTHON="$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"
PORT="${PORT:-8765}"
APP_URL="${APP_URL:-http://127.0.0.1:$PORT/translate}"

cd "$ROOT_DIR"

if command -v curl >/dev/null 2>&1 && curl -fsS "$APP_URL" >/dev/null 2>&1; then
  echo "Readable Local is already running at $APP_URL"
  exit 0
fi

if [[ -x "$BUNDLED_PYTHON" ]]; then
  exec "$BUNDLED_PYTHON" app.py
fi

exec python3 app.py
