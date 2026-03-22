"""Opt-in: time **isolated** SiLU-poly BasicBlock subgraphs on the Toy path.

The depth bench (:func:`test_bench_resnet_silu_layer_runtimes`) builds a **prefix**
from ``input`` to each cut (e.g. ``l2_0`` includes stem + all of layer1 + ``l2_0``).
Toy seconds there are **not** “one block” — use this module to compare apples to
apples:

- one layer1 block vs one ``l2_0`` block;
- three layer1 blocks (chained, no stem) vs ``l2_0``.

Run::

    ROTOM_BENCH_ISOLATED_SILU_BLOCKS=1 python -m pytest \\
      tests/e2e/resnet/test_resnet20_silu_isolated_block_timings.py -v -s

Optional: ``ROTOM_BENCH_LAYOUT_N`` (default ``4096``; heavy e2e uses ``32768``).
"""

from __future__ import annotations

import os
from time import perf_counter

import numpy as np
import pytest
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from benchmarks.e2e.resnet.resnet_model import resnet20
from benchmarks.e2e.resnet.resnet20_tensor_ir import (
    build_resnet20_silu_poly_l1_block_graph,
    build_resnet20_silu_poly_l2_0_block_graph,
    build_resnet20_silu_poly_layer1_only_graph,
    populate_resnet20_no_activation_inputs,
)
from lower.lower import Lower
from tests.test_util import get_default_args

_RUN = os.environ.get("ROTOM_BENCH_ISOLATED_SILU_BLOCKS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _time_pipeline(*, label: str, tensor_ir, inputs: dict, args) -> None:
    t0 = perf_counter()
    kernel = LayoutAssignment(tensor_ir, args).run()
    layout_s = perf_counter() - t0

    t0 = perf_counter()
    circuit_ir = Lower(kernel).run()
    lower_s = perf_counter() - t0

    t0 = perf_counter()
    Toy(circuit_ir, inputs, args).run()
    toy_s = perf_counter() - t0

    t0 = perf_counter()
    tensor_ir.eval(inputs)
    eval_s = perf_counter() - t0

    total = layout_s + lower_s + toy_s + eval_s
    print(
        f"{label:<22} {layout_s:9.3f} {lower_s:9.3f} {toy_s:9.3f} {eval_s:9.3f} {total:9.3f}",
        flush=True,
    )


@pytest.mark.slow
@pytest.mark.skipif(
    not _RUN,
    reason="set ROTOM_BENCH_ISOLATED_SILU_BLOCKS=1 to run isolated block timings",
)
def test_bench_isolated_silu_basic_block_timings() -> None:
    """Print layout / lower / Toy / eval for isolated block IRs (no full prefix)."""
    torch.manual_seed(0)
    model = resnet20(num_classes=10)
    model.eval()
    inputs: dict = {}
    populate_resnet20_no_activation_inputs(model, inputs)

    n = int(os.environ.get("ROTOM_BENCH_LAYOUT_N", "4096"))
    rng = np.random.default_rng(0)
    act16 = rng.standard_normal((16, 32, 32), dtype=np.float64)
    inputs["l1_0_block_in"] = act16.copy()
    inputs["layer1_only_in"] = act16.copy()
    inputs["l2_0_block_in"] = act16.copy()

    args = get_default_args()
    args.backend = "toy"
    args.n = n
    args.rolls = True
    args.net = "lan"
    args.benchmark = "resnet20_silu_poly_l2_0"
    args.skip_toy_eval_checks = True

    print(
        f"\n[isolated-silu-blocks] n={n}  (prefix bench 'l2_0' row is stem+l1+l2_0, "
        f"not comparable to rows below)\n",
        flush=True,
    )
    print(
        f"{'subgraph':<22} {'lay':>9} {'lower':>9} {'toy':>9} {'eval':>9} {'total':>9}",
        flush=True,
    )
    print("-" * 72, flush=True)

    ir_l1_0 = build_resnet20_silu_poly_l1_block_graph(inputs, 0)
    _time_pipeline(label="l1_0 block only", tensor_ir=ir_l1_0, inputs=inputs, args=args)

    ir_l1_all = build_resnet20_silu_poly_layer1_only_graph(inputs)
    _time_pipeline(
        label="layer1 x3 (no stem)",
        tensor_ir=ir_l1_all,
        inputs=inputs,
        args=args,
    )

    ir_l2_0 = build_resnet20_silu_poly_l2_0_block_graph(inputs)
    _time_pipeline(label="l2_0 block only", tensor_ir=ir_l2_0, inputs=inputs, args=args)

    print(flush=True)
