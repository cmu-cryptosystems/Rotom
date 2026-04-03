#!/usr/bin/env bash
# Run layout + OpenEvolve smoke checks using Rotom/.venv.
# Per README.md: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VPY="${ROOT}/.venv/bin/python"
if [[ ! -x "$VPY" ]]; then
  echo "Missing ${VPY}. From README Installation:"
  echo "  cd \"${ROOT}\""
  echo "  python -m venv .venv"
  echo "  source .venv/bin/activate   # or: .venv/bin/pip install -r requirements.txt"
  echo "  pip install -r requirements.txt"
  exit 1
fi
export PYTHONPATH="${ROOT}"
echo "== pytest tests/test_sum_ops.py =="
"${VPY}" -m pytest "${ROOT}/tests/test_sum_ops.py" -q --tb=short
echo "== pytest OpenEvolve layout evolution (fast) =="
PYTHONPATH="${ROOT}" "${VPY}" -m pytest \
  "${ROOT}/tests/test_openevolve_layout_evolution.py" \
  -m "openevolve and not slow" -q --tb=short
echo "== bench_worker (baseline layout_strategy) =="
"${VPY}" "${ROOT}/evolve_openevolve/bench_worker.py" "${ROOT}/assignment/layout_strategy.py"
echo "OK"
