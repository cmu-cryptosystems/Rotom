#!/usr/bin/env python3
"""
Debug the MNIST MLP toy backend layer by layer.

Compares plaintext, apply_layout (expected packed), and toy circuit output
at each intermediate layer to find where the numerical mismatch occurs.

FINDINGS (run: python scripts/debug_mnist_layer_by_layer.py):
- BSGS_ROT_ROLL (input replication): OK
- First BSGS_MATMUL (input @ fc1): MISMATCH - root cause
  - First ~832 slots (slots 0-831) match exactly
  - Mismatch starts at slot 832; ~368+ slots wrong
  - Suggests bug in rotate_and_sum or BSGS for certain slot ranges
- ADD (hidden), POLY (ReLU), second BSGS_MATMUL: all inherit the error
"""

import sys

import numpy as np

# Add project root
sys.path.insert(0, "/usr0/home/ejchen/code/packing/Rotom")

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorOp
from ir.kernel import KernelOp
from lower.lower import Lower
from tests.e2e.mnist.test_mlp_mnist import (
    _build_rotom_mnist_ir,
    _extract_traced_mnist_linears,
)
from tests.e2e.mnist.test_traced_model_plaintext import (
    MODEL_FILE,
    _load_mnist_test_set,
)
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def layer_name(kernel_term):
    """Human-readable name for a kernel term."""
    op = kernel_term.op
    if op == KernelOp.TENSOR or op == KernelOp.PUNCTURED_TENSOR:
        t = kernel_term.layout.term
        if t.op == TensorOp.TENSOR and len(t.cs) > 0:
            return f"TENSOR({t.cs[0]})"
        return f"TENSOR"
    if op == KernelOp.MATMUL:
        return "MATMUL"
    if op == KernelOp.ADD:
        return "ADD"
    if op == KernelOp.POLY:
        return "POLY(relu)"
    if op == KernelOp.SPLIT_ROLL:
        return "SPLIT_ROLL"
    if op == KernelOp.REPLICATE:
        return "REPLICATE"
    return str(op)


def _debug_bsgs_matmul(kernel_term, expected, results, eval_result):
    """Dump layout and slot details for BSGS_MATMUL mismatch."""
    layout = kernel_term.layout
    print(f"\n  [BSGS_MATMUL DEBUG]")
    print(f"    layout.term shape: {layout.term}")
    print(f"    layout.get_dims(): {layout.get_dims()}")
    print(f"    layout.ct_dims: {layout.ct_dims}")
    print(f"    layout.slot_dims: {layout.slot_dims}")
    print(f"    layout.n: {layout.n}")
    print(f"    eval_result shape: {np.asarray(eval_result).shape}")
    exp_flat = np.concatenate([np.asarray(v).flatten() for v in expected])
    res_flat = np.concatenate([np.asarray(v).flatten() for v in results])
    # Find slots where we have non-zero expected (actual data, not padding)
    nonzero_exp = np.where(np.abs(exp_flat) > 1e-6)[0]
    mismatch_slots = np.where(np.abs(exp_flat - res_flat) > 1e-2)[0]
    print(f"    nonzero expected slots (first 20): {nonzero_exp[:20].tolist()}")
    print(f"    mismatch slots (first 20): {mismatch_slots[:20].tolist()}")
    print(f"    first mismatch at slot: {mismatch_slots[0] if len(mismatch_slots) > 0 else 'N/A'}")
    if len(nonzero_exp) > 0:
        for i in nonzero_exp[:5]:
            print(f"      slot {i}: exp={exp_flat[i]:.6f} res={res_flat[i]:.6f} diff={exp_flat[i]-res_flat[i]:.6f}")


def compare_vectors(expected, result, name, rtol=1e-2, atol=1e-2):
    """Compare two vectors and return max diff. Print summary if mismatch."""
    exp_arr = np.asarray(expected)
    res_arr = np.asarray(result)
    if exp_arr.shape != res_arr.shape:
        print(f"  {name}: SHAPE MISMATCH expected {exp_arr.shape} vs {res_arr.shape}")
        return float("inf")
    close = np.allclose(exp_arr, res_arr, rtol=rtol, atol=atol)
    max_diff = np.max(np.abs(exp_arr - res_arr))
    if not close:
        # Find first few mismatching indices
        diff = np.abs(exp_arr - res_arr)
        bad = np.where(diff > atol)[0]
        n_show = min(5, len(bad))
        print(f"  {name}: MISMATCH max_diff={max_diff:.6f}")
        for i in bad[:n_show]:
            print(f"    slot {i}: expected={exp_arr.flat[i]:.6f} got={res_arr.flat[i]:.6f}")
        if len(bad) > n_show:
            print(f"    ... and {len(bad) - n_show} more")
    return max_diff


def main():
    print("Loading MNIST data and model...")
    images, labels = _load_mnist_test_set()
    params = _extract_traced_mnist_linears(MODEL_FILE)
    in_dim = params["in_dim"]
    hidden_dim = params["hidden_dim"]
    out_dim = params["out_dim"]

    # Single sample
    idx = 0
    x_flat = images[idx : idx + 1].view(1, -1).numpy()
    inputs = {
        "input": x_flat,
        "fc1": params["fc1_w"].T,
        "b1": params["fc1_b"].reshape(1, hidden_dim),
        "fc2": params["fc2_w"].T,
        "b2": params["fc2_b"].reshape(1, out_dim),
    }

    tensor_ir = _build_rotom_mnist_ir(in_dim, hidden_dim, out_dim)
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.benchmark = "mlp_mnist_debug"
    args.toy_verify = False

    print("Running layout assignment and lowering...")
    kernel = LayoutAssignment(tensor_ir, args).run()
    circuit_ir = Lower(kernel).run()

    # Build intermediate tensor terms for plaintext comparison
    from frontends.tensor import TensorTerm

    inp = TensorTerm.Tensor("input", [1, in_dim], True)
    fc1 = TensorTerm.Tensor("fc1", [in_dim, hidden_dim], False)
    b1 = TensorTerm.Tensor("b1", [1, hidden_dim], False)
    fc2 = TensorTerm.Tensor("fc2", [hidden_dim, out_dim], False)
    b2 = TensorTerm.Tensor("b2", [1, out_dim], False)
    hidden_term = inp @ fc1 + b1
    hidden_relu_term = hidden_term.poly("relu_exact")
    logits_term = hidden_relu_term @ fc2 + b2

    # Map kernel terms to our layer names for reporting
    layer_terms = {
        "hidden (input@fc1+b1)": hidden_term,
        "hidden_relu (ReLU)": hidden_relu_term,
        "logits (hidden_relu@fc2+b2)": logits_term,
    }

    print("\n" + "=" * 60)
    print("LAYER-BY-LAYER DEBUG")
    print("=" * 60)

    toy = Toy(circuit_ir, inputs, args)

    # Process each kernel term and compare
    for kernel_term, layout_cts in circuit_ir.items():
        op = kernel_term.op
        # Skip input tensors (we only care about computed outputs)
        if op in (KernelOp.TENSOR, KernelOp.PUNCTURED_TENSOR):
            continue
        if op in (KernelOp.SPLIT_ROLL, KernelOp.REPLICATE):
            print(f"\n--- {layer_name(kernel_term)} (skipped for verification) ---")
            continue

        # Evaluate toy circuit for this term
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

        # Expected: plaintext eval + apply_layout
        eval_result = kernel_term.layout.term.eval(inputs)
        expected = apply_layout(eval_result, kernel_term.layout)

        name = layer_name(kernel_term)
        print(f"\n--- {name} ---")
        print(f"  Output shape: {np.asarray(eval_result).shape}")
        print(f"  Num ciphertexts: {len(results)} (expected {len(expected)})")

        max_diff_overall = 0.0
        for i, (exp_vec, res_vec) in enumerate(zip(expected, results)):
            md = compare_vectors(exp_vec, res_vec, f"CT[{i}]")
            max_diff_overall = max(max_diff_overall, md)

        if max_diff_overall < 1e-9:
            print(f"  OK (max_diff={max_diff_overall:.2e})")
        elif max_diff_overall < 1e-2:
            print(f"  SMALL DIFF (max_diff={max_diff_overall:.6f})")
        else:
            print(f"  *** MISMATCH (max_diff={max_diff_overall:.6f}) ***")

        # For first BSGS_MATMUL, dump layout and slot analysis
        if op == KernelOp.BSGS_MATMUL and max_diff_overall > 0.1:
            _debug_bsgs_matmul(kernel_term, expected, results, eval_result)

        # For logits, also show predicted class
        if "logits" in str(kernel_term.layout.term) or (
            op == KernelOp.ADD and len(kernel_term.cs) >= 2
        ):
            # This might be the final ADD (logits)
            flat_exp = np.concatenate([np.asarray(v).flatten() for v in expected])
            flat_res = np.concatenate([np.asarray(v).flatten() for v in results])
            pred_exp = int(np.argmax(flat_exp[:out_dim]))
            pred_res = int(np.argmax(flat_res[:out_dim]))
            print(f"  Plaintext pred: {pred_exp}, Toy pred: {pred_res}, Label: {int(labels[idx].item())}")

    # Final summary: full logits
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    logits_pt = tensor_ir.eval(inputs)
    logits_pt = np.asarray(logits_pt).reshape(-1)[:out_dim]
    pred_pt = int(np.argmax(logits_pt))
    print(f"Plaintext logits (first 10): {logits_pt}")
    print(f"Plaintext pred: {pred_pt}, Label: {int(labels[idx].item())}")


if __name__ == "__main__":
    main()
