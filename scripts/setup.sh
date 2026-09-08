#!/usr/bin/env bash
# Create the virtual environment and install the Phase A dependencies.
#
# On NixOS, run this inside `nix develop` so the PyPI wheels can find libstdc++
# and the other shared libraries they expect.
set -euo pipefail

cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
  echo "creating .venv"
  python3 -m venv .venv
fi

.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt

echo
echo "generating synthetic fixtures"
.venv/bin/python -m satquery.cli fixtures

echo
echo "setup complete. next:"
echo "  ./scripts/demo.sh     run every demo scenario"
echo "  ./scripts/serve.sh    start the API on http://127.0.0.1:8000"
