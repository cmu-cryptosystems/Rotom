#!/usr/bin/env bash
# Full layer pipeline benchmark: stem→l3 plus ``full`` (FC), n=32768, Toy included.
# 28 GiB cgroup cap; output tee’d to a timestamped log. If SIGKILL (OOM), the last
# ``[bench-depth-done]`` line in the log is the last depth that finished; see
# ``ROTOM_TOY_KERNEL_LOG`` for last Toy kernel inside a depth.
#
# Usage (from repo root):
#   ./scripts/bench_resnet_silu_layer_full_28g.sh
#
# Requires: systemd user session (``systemd-run --user``).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

LOG="${ROOT}/resnet_silu_layer_bench_$(date +%Y%m%d_%H%M%S).log"
KERNEL_LOG="${ROOT}/last_toy_kernel_layer_bench.log"
: >"$KERNEL_LOG"

echo "Writing: $LOG"
echo "Toy kernel progress: $KERNEL_LOG"
echo ""

/usr/bin/time -v systemd-run --user --scope \
  -p MemoryMax=28G -p MemorySwapMax=0 -- \
  env PYTHONUNBUFFERED=1 \
  ROTOM_BENCH_RESNET_SILU_LAYER_RUNTIMES=1 \
  ROTOM_BENCH_LAYOUT_N=32768 \
  ROTOM_BENCH_INCLUDE_TOY=1 \
  ROTOM_BENCH_INCLUDE_FULL=1 \
  ROTOM_PROGRESS_BARS=1 \
  ROTOM_TOY_KERNEL_LOG="$KERNEL_LOG" \
  python -m pytest \
  tests/e2e/resnet/test_layout_assignment_resnet_silu_depth.py::test_bench_resnet_silu_layer_runtimes \
  -v -s 2>&1 | tee "$LOG"

echo ""
echo "Done. Log: $LOG"
