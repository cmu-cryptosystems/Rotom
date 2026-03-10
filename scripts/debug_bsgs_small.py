#!/usr/bin/env python3
"""
Debug BSGS matmul: small test, analyze partial products, verify rotations.
Also sanity-check: does matmul work without BSGS (rolls=False)?
"""

import sys

sys.path.insert(0, "/usr0/home/ejchen/code/packing/Rotom")

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from lower.lower import Lower
from lower.lower_util import bsgs, find_bsgs_interval
from tests.conftest import run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_without_bsgs(m, n):
    """Run matmul with rolls=False so BSGS is not applied (sanity check)."""
    np.random.seed(42)
    inputs = {
        "a": np.random.randn(1, m).astype(np.float64) * 0.1,
        "b": np.random.randn(m, n).astype(np.float64) * 0.1,
    }
    a = TensorTerm.Tensor("a", [1, m], True)
    b = TensorTerm.Tensor("b", [m, n], False)
    tensor_ir = a @ b

    args = get_default_args()
    args.n = 4096
    args.rolls = False  # No rolls -> no BSGS
    args.toy_verify = False
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = run_backend("toy", circuit_ir, inputs, args)
    expected = apply_layout(inputs["a"] @ inputs["b"], kernel.layout)
    max_diff = max(
        np.max(np.abs(np.asarray(e) - np.asarray(r))) for e, r in zip(expected, results)
    )
    return max_diff < 1e-2, max_diff


def test_with_bsgs(m, n):
    """Run matmul with rolls=True (BSGS applied) and report layout."""
    np.random.seed(42)
    inputs = {
        "a": np.random.randn(1, m).astype(np.float64) * 0.1,
        "b": np.random.randn(m, n).astype(np.float64) * 0.1,
    }
    a = TensorTerm.Tensor("a", [1, m], True)
    b = TensorTerm.Tensor("b", [m, n], False)
    tensor_ir = a @ b

    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.toy_verify = False
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()
    results = run_backend("toy", circuit_ir, inputs, args)
    expected = apply_layout(inputs["a"] @ inputs["b"], kernel.layout)
    max_diff = max(
        np.max(np.abs(np.asarray(e) - np.asarray(r))) for e, r in zip(expected, results)
    )
    return max_diff < 1e-2, max_diff, kernel, circuit_ir


def analyze_bsgs_structure(m, n):
    """Print BSGS parameters: dim_size, stride, baby_step, giant_step, rotations."""
    np.random.seed(42)
    a = TensorTerm.Tensor("a", [1, m], True)
    b = TensorTerm.Tensor("b", [m, n], False)
    tensor_ir = a @ b
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.toy_verify = False
    kernel = LayoutAssignment(tensor_ir, args).run()

    # Check if BSGS_MATMUL is used
    def has_bsgs(k):
        if k.op == KernelOp.BSGS_MATMUL:
            return True
        for c in getattr(k, "cs", []):
            if isinstance(c, type(k)) and has_bsgs(c):
                return True
        return False

    if not has_bsgs(kernel):
        print("Layout does not use BSGS_MATMUL")
        return

    # Get BSGS params from lower_matmul
    if kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL:
        rot_rolled = kernel.cs[1]
        roll = kernel.cs[0]
    else:
        rot_rolled = kernel.cs[2]
        roll = kernel.cs[0]

    dims = rot_rolled.layout.get_dims()
    dim_size = dims[roll[0]].extent
    stride = 1
    for d in rot_rolled.layout.slot_dims[1:]:
        stride *= d.extent

    baby_step, giant_step = find_bsgs_interval(dim_size)

    print("\n=== BSGS structure ===")
    print(f"  dim_size (sum dimension): {dim_size}")
    print(f"  stride (slot_dims[1:] product): {stride}")
    print(f"  baby_step: {baby_step}, giant_step: {giant_step}")
    print(f"  rot_rolled layout: {rot_rolled.layout.layout_str()}")
    print(f"  slot_dims: {[(str(d), d.extent) for d in rot_rolled.layout.slot_dims]}")

    print("\n=== BSGS rotation amounts (left=True) ===")
    print("  Baby steps on ct (a): rot = bs * stride")
    for bs in range(baby_step):
        rot = bs * stride
        print(f"    bs={bs}: rot {rot}")
    print("  Giant steps on pts (b): rot = -baby_step * gs * stride")
    for gs in range(giant_step):
        rot = -baby_step * gs * stride
        print(f"    gs={gs}: rot {rot}")
    print("  Giant steps on sum: rot = baby_step * gs * stride")
    for gs in range(giant_step):
        rot = baby_step * gs * stride
        print(f"    gs={gs}: rot {rot}")

    # Manual trace: for a simple inner product at slot 0, what slots contribute?
    print("\n=== Partial product alignment (slot 0) ===")
    print("  For output slot 0, we sum over dim_size indices.")
    print("  ct is rotated by bs*stride -> slot i gets ct[i - bs*stride]")
    print(
        "  pt is rotated by -baby_step*gs*stride -> slot i gets pt[i + baby_step*gs*stride]"
    )
    print("  Mul: slot i contributes ct[i-bs*stride] * pt[i+baby_step*gs*stride]")
    print("  For slot 0: need ct[-bs*stride] * pt[baby_step*gs*stride]")
    print(
        "  After baby-step sum: slot 0 has sum over bs of ct[-bs*stride]*pt[bs*stride] (gs=0)"
    )
    print("  After giant-step sum: slot 0 has sum over gs of (rotated sum)")
    print("  Final: slot 0 = sum_{k=0}^{dim_size-1} ct[-k*stride] * pt[k*stride]")
    print(
        f"  So we need ct at slots 0, -stride, -2*stride, ... and pt at 0, stride, 2*stride, ..."
    )

    return dim_size, stride, baby_step, giant_step


def trace_bsgs_slot_by_slot(m, n):
    """
    For a small case: manually compute expected inner product per output slot,
    then compare to what BSGS would produce given the packing.
    """
    np.random.seed(42)
    a = np.random.randn(1, m).astype(np.float64) * 0.1
    b = np.random.randn(m, n).astype(np.float64) * 0.1
    expected = a @ b  # shape (1, n)

    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.toy_verify = False

    a_term = TensorTerm.Tensor("a", [1, m], True)
    b_term = TensorTerm.Tensor("b", [m, n], False)
    tensor_ir = a_term @ b_term
    kernel = LayoutAssignment(tensor_ir, args).run()

    if kernel.op != KernelOp.BSGS_MATMUL:
        print("Kernel does not use BSGS_MATMUL, skipping trace")
        return

    rot_rolled = (
        kernel.cs[1] if kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL else kernel.cs[2]
    )
    roll_idx = kernel.cs[0]
    dims = rot_rolled.layout.get_dims()
    dim_size = dims[roll_idx[0]].extent
    stride = 1
    for d in rot_rolled.layout.slot_dims[1:]:
        stride *= d.extent
    baby_step, giant_step = find_bsgs_interval(dim_size)

    print("\n=== Slot-by-slot trace ===")
    print(f"Output layout: {kernel.layout.layout_str()}")
    print(f"dim_size={dim_size}, stride={stride}, baby={baby_step}, giant={giant_step}")

    # Apply layout to get packed form
    from util.layout_util import apply_layout, get_dim_indices

    packed_a = apply_layout(
        a.reshape(1, m), kernel.cs[1].cs[1].layout
    )  # REPLICATE input
    packed_b = apply_layout(b, kernel.cs[2].layout)  # weights

    # For each output ct, first 512 slots (or whatever) map to output elements
    # We need to understand the slot->output mapping
    out_layout = kernel.layout
    layout_len = len(out_layout)
    dim_indices = get_dim_indices(out_layout.get_dims())
    print(f"Output layout_len={layout_len}, n={args.n}")


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 4

    print(f"=== Small matvecmul [1,{m}] @ [{m},{n}] ===\n")

    print("1. Sanity check: WITHOUT BSGS (rolls=False)")
    ok, md = test_without_bsgs(m, n)
    print(f"   Result: {'PASS' if ok else 'FAIL'} max_diff={md:.6f}\n")

    print("2. WITH BSGS (rolls=True)")
    ok, md, kernel, _ = test_with_bsgs(m, n)
    print(f"   Result: {'PASS' if ok else 'FAIL'} max_diff={md:.6f}")

    print("\n3. BSGS structure analysis")
    analyze_bsgs_structure(m, n)

    print("\n4. Slot-by-slot trace (if BSGS used)")
    trace_bsgs_slot_by_slot(m, n)


if __name__ == "__main__":
    main()
