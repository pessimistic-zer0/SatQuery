#!/usr/bin/env bash
# Run every demo scenario over the synthetic fixtures.
set -euo pipefail
cd "$(dirname "$0")/.."
exec .venv/bin/python -m satquery.cli demo "$@"
