#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Please install Python 3.11 or 3.12, then run this script again."
  exit 1
fi
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m wifi_localization demo --data data/room --output outputs/my_run
echo "Open outputs/my_run/evaluation/localization.html to view the results."
