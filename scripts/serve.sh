#!/usr/bin/env bash
# Start the API. Set SATQUERY_FORCE_CPU=1 to exercise the no-GPU fallback path.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/uvicorn satquery.api:app --host 127.0.0.1 --port "${PORT:-8000}" --reload
