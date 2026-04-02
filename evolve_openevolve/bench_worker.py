#!/usr/bin/env python3
"""
Run Rotom benchmarks under a layout strategy (ROTOM_LAYOUT_STRATEGY_PATH).

Prints one JSON object to stdout: { "ok", "benchmarks", "error" }.
Invoked in a fresh subprocess by the OpenEvolve evaluator.
"""

from __future__ import annotations

import json
import os
import random
import sys


def _rotom_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main() -> None:
    if len(sys.argv) < 2:
        print(
            json.dumps({"ok": False, "benchmarks": {}, "error": "missing strategy path"}),
            flush=True,
        )
        sys.exit(2)

    strategy_path = os.path.abspath(sys.argv[1])
    os.environ["ROTOM_LAYOUT_STRATEGY_PATH"] = strategy_path

    root = _rotom_root()
    if root not in sys.path:
        sys.path.insert(0, root)

    import numpy as np
    from argparse import Namespace

    from assignment.assignment import LayoutAssignment
    from backends.toy import Toy
    from benchmarks.rotom_benchmarks.double_matmul.double_matmul_128_64_ct_ct import (
        double_matmul_128_64_ct_ct,
    )
    from benchmarks.rotom_benchmarks.matmul.matmul_128_64 import matmul_128_64
    from benchmarks.rotom_benchmarks.matmul.matmul_256_128 import matmul_256_128
    from ir.kernel_cost import KernelCost
    from lower.lower import Lower
    from util.layout_util import apply_layout

    def make_args(benchmark_name: str) -> Namespace:
        return Namespace(
            backend="toy",
            n=4096,
            benchmark=benchmark_name,
            microbenchmark="main",
            rolls=True,
            strassens=False,
            net="lan",
            cache=False,
            serialize=False,
            mock=False,
            fuzz=False,
            fuzz_result=False,
            conv_roll=False,
            fn="evolve_bench",
            not_secure=False,
            skip_toy_eval_checks=False,
        )

    specs = [
        ("matmul_128_64", matmul_128_64),
        ("matmul_256_128", matmul_256_128),
        ("double_matmul_128_64", double_matmul_128_64_ct_ct),
    ]

    out: dict = {}
    try:
        for name, builder in specs:
            random.seed(0)
            np.random.seed(0)
            tensor_ir, inputs = builder()
            args = make_args(name)
            kernel = LayoutAssignment(tensor_ir, args).run()
            circuit_ir = Lower(kernel).run()
            results = Toy(circuit_ir, inputs, args).run()

            expected_cts = apply_layout(tensor_ir.eval(inputs), kernel.layout)
            semantic_ok = True
            max_diff = 0.0
            for expected, result in zip(expected_cts, results):
                if not np.allclose(expected, result, rtol=1e-2, atol=1e-2):
                    semantic_ok = False
                    diff = np.asarray(expected) - np.asarray(result)
                    max_diff = max(max_diff, float(np.max(np.abs(diff))))

            cost = float(KernelCost(kernel, args.net).total_cost())
            out[name] = {
                "semantic_ok": semantic_ok,
                "max_diff": max_diff,
                "total_cost": cost,
            }

        ok = all(b["semantic_ok"] for b in out.values())
        print(json.dumps({"ok": ok, "benchmarks": out, "error": None}), flush=True)
    except Exception as e:
        print(
            json.dumps(
                {
                    "ok": False,
                    "benchmarks": out,
                    "error": f"{type(e).__name__}: {e}",
                }
            ),
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
