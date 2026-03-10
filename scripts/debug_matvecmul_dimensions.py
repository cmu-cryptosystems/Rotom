#!/usr/bin/env python3
"""
Debug matvecmul [1, M] @ [M, N] across different dimensions.

Tests various M, N combinations to find which trigger the BSGS bug.
Non-power-of-two dimensions: 100, 130, 512, 784, etc.
"""

import sys

sys.path.insert(0, "/usr0/home/ejchen/code/packing/Rotom")

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.conftest import run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_matvecmul(m, n, args, verbose=False):
    """Run [1, m] @ [m, n] and return (pass, max_diff)."""
    np.random.seed(42)
    inputs = {
        "a": np.random.randn(1, m).astype(np.float64) * 0.1,
        "b": np.random.randn(m, n).astype(np.float64) * 0.1,
    }
    a = TensorTerm.Tensor("a", [1, m], True)
    b = TensorTerm.Tensor("b", [m, n], False)
    tensor_ir = a @ b
    expected = inputs["a"] @ inputs["b"]

    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()

    # Check which matmul path was used
    from ir.kernel import KernelOp

    matmul_ops = []
    for term in kernel.post_order():
        if term.op == KernelOp.BSGS_MATMUL:
            matmul_ops.append("BSGS_MATMUL")
        elif term.op == KernelOp.MATMUL:
            matmul_ops.append("MATMUL")

    args.toy_verify = False
    results = run_backend("toy", circuit_ir, inputs, args)
    expected_cts = apply_layout(expected, kernel.layout)

    max_diff = 0.0
    for exp, res in zip(expected_cts, results):
        diff = np.max(np.abs(np.asarray(exp) - np.asarray(res)))
        max_diff = max(max_diff, diff)

    close = max_diff < 1e-2
    if verbose or not close:
        path = matmul_ops[0] if matmul_ops else "?"
        print(
            f"  [1,{m}] @ [{m},{n}] -> {path}: max_diff={max_diff:.6f} {'PASS' if close else 'FAIL'}"
        )
    return close, max_diff


def main():
    args = get_default_args()
    args.n = 4096
    args.rolls = True

    # Test dimensions: (m, n) for [1,m] @ [m,n]
    dims = [
        # Powers of 2
        (4, 4),
        (8, 8),
        (16, 16),
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 64),
        (128, 128),
        # Non-power-of-two
        (100, 50),
        (130, 66),
        (256, 128),
        (512, 256),
        (512, 512),
        (784, 512),  # MNIST FC1
        (128, 65),
        (65, 32),
        (80, 48),
        (200, 100),
    ]

    print("Testing matvecmul dimensions [1,M] @ [M,N] with n=4096:")
    print("=" * 60)

    passed = []
    failed = []
    for m, n in dims:
        ok, max_diff = test_matvecmul(m, n, args, verbose=False)
        path = "?"  # We'd need to re-run to get path
        if ok:
            passed.append((m, n, max_diff))
        else:
            failed.append((m, n, max_diff))

    print("\nPASSED:")
    for m, n, diff in passed:
        print(f"  [1,{m}] @ [{m},{n}]  max_diff={diff:.2e}")

    print("\nFAILED:")
    for m, n, diff in failed:
        print(f"  [1,{m}] @ [{m},{n}]  max_diff={diff:.6f}")

    # Find boundary: smallest failing dimension
    print("\n" + "=" * 60)
    print("Summary: passed", len(passed), "failed", len(failed))


if __name__ == "__main__":
    main()
