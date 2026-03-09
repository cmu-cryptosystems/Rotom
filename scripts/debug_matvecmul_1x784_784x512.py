#!/usr/bin/env python3
"""
Debug matvecmul [1, M] @ [M, N] - layer by layer for BSGS_MATMUL.

Usage:
  python scripts/debug_matvecmul_1x784_784x512.py           # 784x512 (MNIST)
  python scripts/debug_matvecmul_1x784_784x512.py 130 66  # smaller failing case

Findings:
- Failing dims: (130,66), (200,100), (784,512) - all use BSGS_MATMUL
- Passing dims: (100,50), (80,48), (128,65), (65,32), (512,512), etc.
- First mismatch at slot 512 for 130x66 and 200x100 (layout [1:8:16][G:32][1:16:1])
- First mismatch at slot 832 for 784x512 (layout [1:128:4][G:8][1:4:1])
- Bug appears at ct/group boundary - second output group is wrong
"""

import sys
sys.path.insert(0, "/usr0/home/ejchen/code/packing/Rotom")

import numpy as np
from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def main():
    m = int(sys.argv[1]) if len(sys.argv) > 1 else 784
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 512

    print(f"Debug matvecmul [1,{m}] @ [{m},{n}]")
    print("=" * 60)

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

    toy = Toy(circuit_ir, inputs, args)

    for kernel_term, layout_cts in circuit_ir.items():
        op = kernel_term.op
        if op in (KernelOp.TENSOR, KernelOp.PUNCTURED_TENSOR):
            continue
        if op in (KernelOp.SPLIT_ROLL, KernelOp.REPLICATE):
            print(f"\n--- {op} (skipped) ---")
            continue

        results = []
        for ct_idx in sorted(layout_cts.keys()):
            ct = layout_cts[ct_idx]
            if isinstance(ct, list):
                for c in ct:
                    for ct_term in c.post_order():
                        toy.env[ct_term] = toy.eval(ct_term)
                    results.append(toy.env[ct_term])
            else:
                for ct_term in ct.post_order():
                    toy.env[ct_term] = toy.eval(ct_term)
                results.append(toy.env[ct_term])

        eval_result = kernel_term.layout.term.eval(inputs)
        expected = apply_layout(eval_result, kernel_term.layout)

        exp_flat = np.concatenate([np.asarray(v).flatten() for v in expected])
        res_flat = np.concatenate([np.asarray(v).flatten() for v in results])
        max_diff = np.max(np.abs(exp_flat - res_flat))
        mismatch_slots = np.where(np.abs(exp_flat - res_flat) > 1e-2)[0]

        print(f"\n--- {op} ---")
        print(f"  shape: {np.asarray(eval_result).shape}, max_diff: {max_diff:.6f}")
        if len(mismatch_slots) > 0:
            print(f"  first mismatch at slot: {mismatch_slots[0]}")
            print(f"  layout slot_dims: {kernel_term.layout.slot_dims}")
            # Show pattern: is first_mismatch at a boundary?
            fm = mismatch_slots[0]
            print(f"  slot {fm}: exp={exp_flat[fm]:.6f} res={res_flat[fm]:.6f}")


if __name__ == "__main__":
    main()
