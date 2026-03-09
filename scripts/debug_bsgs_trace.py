#!/usr/bin/env python3
"""
Debug BSGS: trace actual values through replicate -> BSGS, compare to expected.
Find where the mismatch originates.
"""

import sys
sys.path.insert(0, "/usr0/home/ejchen/code/packing/Rotom")

import numpy as np
from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from lower.lower import Lower as LowerPass
from lower.layout_cts import LayoutCiphertexts
from tests.test_util import get_default_args
from util.layout_util import apply_layout, get_ct_idxs_by_dim


def main():
    m, n = 130, 66
    np.random.seed(42)
    inputs = {
        "a": np.random.randn(1, m).astype(np.float64) * 0.1,
        "b": np.random.randn(m, n).astype(np.float64) * 0.1,
    }
    expected_matmul = inputs["a"] @ inputs["b"]  # (1, n)

    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.toy_verify = False

    a = TensorTerm.Tensor("a", [1, m], True)
    b = TensorTerm.Tensor("b", [m, n], False)
    tensor_ir = a @ b
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = LowerPass(kernel).run()

    if kernel.op != KernelOp.BSGS_MATMUL:
        print("Not BSGS_MATMUL, skipping")
        return

    # Get kernels
    rep_kernel = kernel.cs[1].cs[1]
    rot_rolled = kernel.cs[1]
    rolled_kernel = kernel.cs[2]

    # Run toy and capture env at each step
    toy = Toy(circuit_ir, inputs, args)

    # Evaluate up to REPLICATE
    rep_cts = circuit_ir.get(rep_kernel)
    if rep_cts is None:
        print("REPLICATE not in circuit_ir directly - it's a sub-kernel")
        # circuit_ir keys are top-level terms; we need to eval the full circuit
        # and the env gets populated. Let me run the full circuit but check each term.
        pass

    # Run full circuit to populate env
    results = toy.run()

    # Get expected output layout
    expected_layout = apply_layout(expected_matmul, kernel.layout)
    exp_flat = np.concatenate([np.asarray(v).flatten() for v in expected_layout])
    res_flat = np.concatenate([np.asarray(v).flatten() for v in results])

    # Find first mismatch
    diff = np.abs(exp_flat - res_flat)
    mismatch_idx = np.where(diff > 1e-2)[0]
    if len(mismatch_idx) == 0:
        print("No mismatch!")
        return
    first_mismatch = mismatch_idx[0]
    print(f"First mismatch at flat index {first_mismatch}")
    print(f"  expected[{first_mismatch}] = {exp_flat[first_mismatch]:.6f}")
    print(f"  result[{first_mismatch}]   = {res_flat[first_mismatch]:.6f}")

    # Map flat index to (ct_idx, slot_in_ct)
    # Output layout: [1:8:16][G:32][1:16:1] -> 8 cts, 512 slots each
    slots_per_ct = 512  # 32*16
    ct_idx = first_mismatch // slots_per_ct
    slot_in_ct = first_mismatch % slots_per_ct
    print(f"  -> output ct {ct_idx}, slot {slot_in_ct}")

    # BSGS ct_groups
    ct_groups = get_ct_idxs_by_dim(
        rot_rolled.layout.ct_dims,
        rot_rolled.layout.rolls[0].dim_to_roll,
    )
    print(f"\nct_groups (by dim_to_roll): {ct_groups}")
    print(f"  Group 0 uses replicate ct {ct_groups[0][0]}")
    print(f"  Group {ct_idx} uses replicate ct {ct_groups[ct_idx][0]}")

    # Check: are replicate cts 0 and 1 the same underlying term?
    # We need to look at the circuit IR. The replicate produces LayoutCiphertexts.
    # The keys are 0..7. The values are HETerms. For 1 input ct, they might all
    # reference the same term (or have rotations).
    print("\n--- Inspecting lowered circuit ---")
    # The circuit_ir maps kernel -> LayoutCiphertexts
    # We need the env from the Lower pass, not the Toy. The Toy has its own env
    # with evaluated vectors. Let me re-run Lower and inspect the structure.
    lower = LowerPass(kernel)
    lower.lower()
    rep_layout_cts = lower.env.get(rep_kernel)
    if rep_layout_cts:
        print(f"REPLICATE output: {len(rep_layout_cts.cts)} cts")
        for i, ct in list(rep_layout_cts.cts.items())[:3]:
            # Get the root term (before rotations in post_order)
            terms = list(ct.post_order()) if hasattr(ct, 'post_order') else []
            rot_terms = [t for t in terms if hasattr(t, 'op') and str(t.op) == 'HEOp.ROT']
            print(f"  ct[{i}]: {len(terms)} terms, {len(rot_terms)} ROTs")
            if rot_terms:
                print(f"    first ROT: {rot_terms[0]}")

    # Compare replicate ct 0 vs ct 1 actual values (from toy env)
    # The toy env has HETerm -> vector. We need to find the terms for rep ct 0 and 1.
    # The results we have are for the final BSGS_MATMUL output. The toy evaluates
    # in post_order, so we get the final output. To get intermediate values, we'd
    # need to evaluate step by step. Let me check if we can get the rep output.
    # Actually the circuit_ir has one entry per top-level term. The kernel is the
    # root. So we have kernel -> list of cts. Each ct is a DAG of HETerms. The
    # toy evaluates the full DAG. So we have the final output. To get the rep
    # output, we'd need to run a modified circuit that stops at REPLICATE. Or we
    # could add a debug mode to the Toy. For now, let me try to understand the
    # structure by looking at the BSGS matmul lowering.
    print("\n--- BSGS matmul structure ---")
    print(f"replicated_kernel layout: {rep_kernel.layout.layout_str()}")
    print(f"  num_ct: {rep_kernel.layout.num_ct()}")
    print(f"  ct_dims: {[(str(d), d.extent) for d in rep_kernel.layout.ct_dims]}")
    print(f"rot_rolled layout: {rot_rolled.layout.layout_str()}")
    print(f"  ct_dims: {[(str(d), d.extent) for d in rot_rolled.layout.ct_dims]}")
    print(f"  slot_dims: {[(str(d), d.extent) for d in rot_rolled.layout.slot_dims]}")

    # Key: for output group ct_idx, we use bsgs_terms[ct_idx] = replicated_cts[ct_group[0]]
    # So we use replicate ct = ct_groups[ct_idx][0]. For ct_idx=0, we use rep ct 0.
    # For ct_idx=1, we use rep ct 1. So we use different replicate cts. If they're
    # all the same (no rotation), then we're using the same data. The weights
    # (other_terms) are different - rolled_cts[ct] for ct in ct_group. So we
    # multiply the same input by different weights. That should give different
    # (correct) outputs. So the bug might be in the weights packing - maybe
    # rolled_cts[1] has wrong data for the second group.
    print("\n--- Checking weight packing ---")
    # rolled_kernel has the weights. Its layout packs B (m,n).
    b_packed = apply_layout(inputs["b"], rolled_kernel.layout)
    print(f"B shape: {inputs['b'].shape}")
    print(f"B packed: {len(b_packed)} cts")
    for i, pt in enumerate(b_packed[:3]):
        arr = np.asarray(pt).flatten()
        print(f"  packed_B[{i}]: len={len(arr)}, first 4 slots: {arr[:4]}")
        if i == 1:
            print(f"    slots 0,512,1024,1536: {arr[0]}, {arr[512]}, {arr[1024]}, {arr[1536]}")

    # For BSGS, we need pt at slots 0, stride, 2*stride, ... for the inner product.
    # Stride = 512. So we need pt[0], pt[512], pt[1024], pt[1536], ...
    # For output group 1, we use a different set of weights. The weights for
    # group 1 should be columns that correspond to output indices 512..1023 in
    # the flattened output. The output layout [1:8:16][G:32][1:16:1] has 8*32*16
    # = 4096. So we have 8 groups of 512. Group 0 = output cols 0..511, group 1 =
    # 512..1023, etc. So group 1 corresponds to output columns 512..1023 in the
    # flattened layout. The matmul output is (1, 66). So we have 66 output
    # elements. The layout packs 66 elements into 4096 slots (with gaps). So
    # the mapping from output element to slot is complex. Let me check the
    # output layout more carefully.
    print("\n--- Output layout mapping ---")
    out_layout = kernel.layout
    from util.layout_util import get_dim_indices
    dim_indices = get_dim_indices(out_layout.get_dims())
    # Which slot holds output element (0, j)?
    # The layout has dims. For [1:8:16][G:32][1:16:1], we have 8, 32, 16.
    # So we have 8*32*16 = 4096 slots. But we only have 66 output elements.
    # So there are gaps (G:32). The layout packs 66 elements with gaps.
    print(f"Output layout dims: {[(str(d), d.extent, d.dim) for d in out_layout.get_dims()]}")


if __name__ == "__main__":
    main()
