#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Please install Python 3.11 or 3.12, then run this script again."
  exit 1
fi
echo "Preparing the local photo localization app..."
if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install -r requirements-enhanced.txt
.venv/bin/python -m wifi_localization web --output outputs/web
