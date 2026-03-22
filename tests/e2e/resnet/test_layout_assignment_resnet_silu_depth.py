"""Opt-in benchmarks: SiLU ResNet-20 tensor IR vs graph depth.

**Layout only** — ``LayoutAssignment`` seconds and ``post_order`` length::

    ROTOM_BENCH_LAYOUT_ASSIGNMENT_DEPTH=1 python -m pytest \\
      tests/e2e/resnet/test_layout_assignment_resnet_silu_depth.py -v -s

**Layer pipeline** — for ``stem``, ``l1``, ``l2_0``, ``l2``, ``l3`` (and optionally
``full``): cumulative seconds and **Δ** (increment vs previous depth) for layout,
lower, tensor ``eval``, and optionally Toy::

    ROTOM_BENCH_RESNET_SILU_LAYER_RUNTIMES=1 python -m pytest \\
      tests/e2e/resnet/test_layout_assignment_resnet_silu_depth.py -v -s

Optional env:

- ``ROTOM_BENCH_LAYOUT_N`` — default ``4096``; use ``32768`` to match heavy e2e.
- ``ROTOM_BENCH_INCLUDE_TOY=1`` — include ``Toy.run()`` (slow; same as heavy test, use
  ``skip_toy_eval_checks``).
- ``ROTOM_BENCH_INCLUDE_FULL=1`` — add a ``full`` row (``stem`` … FC).

If the process is **SIGKILL**’d (e.g. cgroup OOM), the log’s **last**
``[bench-depth-done]`` line is the **last depth that fully completed**; the next
depth was interrupted. With ``ROTOM_BENCH_INCLUDE_TOY=1``, set
``ROTOM_TOY_KERNEL_LOG=/path/to.log`` to record the last Toy kernel (see Toy docs).

**28 GiB cgroup + full pipeline (Toy + ``full`` row)** — see
``scripts/bench_resnet_silu_layer_full_28g.sh``.
"""

from __future__ import annotations

import os
from time import perf_counter

import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    SiluPolyDepth,
    build_resnet20_silu_poly_graph_to_depth,
    populate_resnet20_no_activation_inputs,
)
from lower.lower import Lower
from tests.test_util import get_default_args

_DEPTHS: tuple[SiluPolyDepth, ...] = (
    "stem",
    "l1",
    "l2_0",
    "l2",
    "l3",
    "full",
)

_PIPELINE_DEPTHS: tuple[SiluPolyDepth, ...] = (
    "stem",
    "l1",
    "l2_0",
    "l2",
    "l3",
)

_RUN_BENCH = os.environ.get(
    "ROTOM_BENCH_LAYOUT_ASSIGNMENT_DEPTH", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

_RUN_LAYER_BENCH = os.environ.get(
    "ROTOM_BENCH_RESNET_SILU_LAYER_RUNTIMES", ""
).strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_BENCH,
    reason="set ROTOM_BENCH_LAYOUT_ASSIGNMENT_DEPTH=1 to run this benchmark",
)
def test_bench_layout_assignment_resnet_silu_by_depth() -> None:
    """Print a table: depth → seconds for ``LayoutAssignment(...).run()``."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_resnet20_no_activation_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    n = int(os.environ.get("ROTOM_BENCH_LAYOUT_N", "4096"))
    args = get_default_args()
    args.backend = "toy"
    args.n = n
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_l2_0"

    print(
        f"\n[layout-assignment-bench] n={n} benchmark={args.benchmark!r}\n",
        flush=True,
    )
    print(f"{'depth':<8} {'LayoutAssignment_s':>18} {'post_order_len':>16}", flush=True)
    print("-" * 46, flush=True)

    for depth in _DEPTHS:
        tensor_ir = build_resnet20_silu_poly_graph_to_depth(inputs, depth)
        po = tensor_ir.post_order()
        t0 = perf_counter()
        LayoutAssignment(tensor_ir, args).run()
        elapsed = perf_counter() - t0
        print(f"{depth:<8} {elapsed:>18.3f} {len(po):>16}", flush=True)

    print(flush=True)


def _delta_str(cumulative: float, prev_cumulative: float | None) -> str:
    if prev_cumulative is None:
        return "   —"
    return f"{cumulative - prev_cumulative:+7.3f}"


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN_LAYER_BENCH,
    reason="set ROTOM_BENCH_RESNET_SILU_LAYER_RUNTIMES=1 to run layer pipeline timings",
)
def test_bench_resnet_silu_layer_runtimes() -> None:
    """Per-depth timings: layout, lower, tensor eval, optional Toy; Δ = increment vs prior depth."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_resnet20_no_activation_inputs(model, inputs)
    x = torch.randn(3, 32, 32, dtype=torch.float64)
    inputs["input"] = x.numpy()

    n = int(os.environ.get("ROTOM_BENCH_LAYOUT_N", "4096"))
    include_toy = os.environ.get("ROTOM_BENCH_INCLUDE_TOY", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    include_full = os.environ.get("ROTOM_BENCH_INCLUDE_FULL", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    depths: tuple[SiluPolyDepth, ...] = (
        *_PIPELINE_DEPTHS,
        *(("full",) if include_full else ()),
    )

    args = get_default_args()
    args.backend = "toy"
    args.n = n
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_l2_0"
    args.skip_toy_eval_checks = True

    print(
        f"\n[resnet-silu-layer-runtimes] n={n} include_toy={include_toy} "
        f"include_full={include_full}\n",
        flush=True,
    )

    head = (
        f"{'depth':<8} {'lay':>8} {'Δlay':>8} {'lower':>8} {'Δlow':>8} "
        f"{'eval':>8} {'Δev':>8}"
    )
    if include_toy:
        head += f" {'toy':>8} {'Δty':>8}"
    print(head, flush=True)
    print("-" * len(head), flush=True)

    prev_layout = prev_lower = prev_eval = prev_toy = None

    for depth in depths:
        tensor_ir = build_resnet20_silu_poly_graph_to_depth(inputs, depth)

        t0 = perf_counter()
        kernel = LayoutAssignment(tensor_ir, args).run()
        layout_s = perf_counter() - t0

        t0 = perf_counter()
        circuit_ir = Lower(kernel).run()
        lower_s = perf_counter() - t0

        toy_s = 0.0
        if include_toy:
            t0 = perf_counter()
            Toy(circuit_ir, inputs, args).run()
            toy_s = perf_counter() - t0

        t0 = perf_counter()
        tensor_ir.eval(inputs)
        eval_s = perf_counter() - t0

        row = (
            f"{depth:<8} {layout_s:8.3f} {_delta_str(layout_s, prev_layout)} "
            f"{lower_s:8.3f} {_delta_str(lower_s, prev_lower)} "
            f"{eval_s:8.3f} {_delta_str(eval_s, prev_eval)}"
        )
        if include_toy:
            row += f" {toy_s:8.3f} {_delta_str(toy_s, prev_toy)}"

        print(row, flush=True)
        print(
            f"[bench-depth-done] depth={depth!r} (row printed = this stage finished)",
            flush=True,
        )

        prev_layout, prev_lower, prev_eval, prev_toy = (
            layout_s,
            lower_s,
            eval_s,
            toy_s,
        )

    print(flush=True)
