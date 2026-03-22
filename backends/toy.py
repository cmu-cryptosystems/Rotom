import os

import numpy as np

np.set_printoptions(legacy="1.25")

from typing import Any, Literal

from frontends.tensor import TensorOp
from ir.he import HEOp, HETerm
from ir.kernel import KernelOp
from util.layout_util import *


def _collect_het_nodes(root: HETerm) -> set[HETerm]:
    """All ``HETerm`` nodes reachable from ``root`` (follows every ``cs`` edge)."""
    nodes: set[HETerm] = set()
    stack: list[HETerm] = [root]
    while stack:
        t = stack.pop()
        if t in nodes:
            continue
        nodes.add(t)
        for c in t.cs:
            if isinstance(c, HETerm):
                stack.append(c)
    return nodes


def _build_remaining_uses(root: HETerm) -> dict[HETerm, int]:
    """Incoming-use counts: how many parent terms reference each node as an operand."""
    nodes = _collect_het_nodes(root)
    rem = {t: 0 for t in nodes}
    for t in nodes:
        for c in t.cs:
            if isinstance(c, HETerm):
                rem[c] += 1
    return rem


def _toy_post_order(root: HETerm) -> list[HETerm]:
    """Post-order like ``HETerm.helper_post_order``, but recurses through ``HEOp.CS``.

    The stock ``post_order`` omits nodes only reachable under ``CS``, which forced
    coarse ``env.clear()`` per ciphertext. Including ``CS`` children yields a full
    topological order so we can drop ``env`` entries when uses hit zero (same idea
    as ``OpenFHE.update_dependencies``).

    Implemented with an explicit stack so very deep DAGs (e.g. large ResNet HE IR)
    do not hit Python's recursion limit.
    """
    order: list[HETerm] = []
    seen: set[HETerm] = set()
    stack: list[tuple[HETerm, Literal["enter", "exit"]]] = [(root, "enter")]

    while stack:
        term, phase = stack.pop()
        if phase == "enter":
            if term in seen:
                continue
            seen.add(term)
            stack.append((term, "exit"))
            match term.op:
                case HEOp.CS:
                    if isinstance(term.cs[0], HETerm):
                        stack.append((term.cs[0], "enter"))
                case (
                    HEOp.PACK
                    | HEOp.PUNCTURED_PACK
                    | HEOp.INDICES
                    | HEOp.CS_PACK
                    | HEOp.CONST
                ):
                    pass
                case HEOp.ADD | HEOp.SUB | HEOp.MUL:
                    stack.append((term.cs[1], "enter"))
                    stack.append((term.cs[0], "enter"))
                case HEOp.ROT | HEOp.POLY_CALL | HEOp.RESCALE:
                    stack.append((term.cs[0], "enter"))
                case HEOp.MASK | HEOp.ZERO_MASK:
                    pass
                case _:
                    raise NotImplementedError(term.op)
        else:
            order.append(term)

    return order


def _packed_ct_lists_to_float64(packed_cts: list) -> np.ndarray:
    """One allocation: ``apply_layout`` / ``apply_punctured_layout`` output (list of slot lists) → ``float64``.

    Rows must be rectangular (same slot count per ciphertext); that matches layout packing invariants.
    """
    return np.asarray(packed_cts, dtype=np.float64)


class Toy:
    """
    Toy Backend for plaintext simulation of FHE circuits.

    This backend executes FHE circuits in plaintext, providing a fast
    simulation environment for development, testing, and debugging. It
    implements the same interface as other backends but performs all
    operations on plaintext data.

    **Memory:** For each ciphertext root DAG, Toy evaluates in an extended
    post-order (including ``HEOp.CS`` children, unlike ``HETerm.post_order``)
    and removes operands from ``env`` when their incoming use count hits
    zero—same def-use idea as ``OpenFHE.update_dependencies``. That keeps peak
    memory bounded for large circuits (e.g. ResNet-scale with big ``n``)
    instead of retaining every intermediate until a full ``env`` wipe.
    Peak RSS can still be large when a single DAG holds many live vectors of
    size ``n`` (e.g. hitting a cgroup ``MemoryMax`` → SIGKILL).

    **Progress:** Set ``ROTOM_PROGRESS_BARS=1`` for a tqdm bar over circuit
    kernels. Set ``ROTOM_TOY_KERNEL_LOG=/path/to.log`` to append one line per
    kernel before it runs (last line ≈ last kernel reached if the process is
    killed).

    **Tensor eval caching (correctness):**

    - ``_tensor_term_dense_by_id``: memoizes ``TensorTerm.eval(inputs)`` by
      ``id(layout.term)``. **Valid** while ``self.inputs`` is unchanged and IR
      nodes are not mutated mid-run (normal compiler usage). Same term object ⇒
      one dense tensor; packing may still differ by ``Layout``.
    - ``input_cache``: packed slot rows keyed by ``(layout.term, layout)``.
      **Valid** for the same reasons; each key is one layout geometry applied to
      that term’s dense value at fill time. Identical geometry with two different
      ``Layout`` objects ⇒ duplicate work but still correct.

    Dense values depend only on the tensor IR node, not packing; distinct
    ``Layout`` instances that share one ``layout.term`` therefore reuse one
    ``eval`` via ``_tensor_term_dense_by_id``.

    **Packed storage:** layout helpers return Python ``list[list[scalar]]``; Toy
    converts once to a ``float64`` 2-D array so row slices are views without a
    second full copy via ``np.asarray`` + ``_as_np_vec``.

    Attributes:
        circuit_ir: The FHE circuit intermediate representation
        inputs: Dictionary of input tensors
        n: HE vector size
        env: Environment for storing intermediate results (drained per ct root)
        input_cache: Cache for packed input tensors
    """

    def __init__(self, circuit_ir, inputs, args):
        """
        Initialize the Toy backend.

        Args:
            circuit_ir: The FHE circuit intermediate representation
            inputs: Dictionary mapping tensor names to input data
            args: Command line arguments containing configuration
        """
        self.circuit_ir = circuit_ir
        self.inputs = inputs
        self.args = args
        self.n = args.n
        self.env = {}
        self.input_cache = {}
        # id(TensorTerm) -> dense result of TensorTerm.eval(inputs).  Keyed by
        # object identity (not TensorTerm.__hash__), since __eq__ is hash-based.
        self._tensor_term_dense_by_id: dict[int, Any] = {}

    def _dense_eval_for_tensor_term(self, tensor_term):
        """Run ``tensor_term.eval(self.inputs)`` at most once per IR node object."""
        tid = id(tensor_term)
        if tid in self._tensor_term_dense_by_id:
            return self._tensor_term_dense_by_id[tid]
        dense = tensor_term.eval(self.inputs)
        self._tensor_term_dense_by_id[tid] = dense
        return dense

    def _eval_ct_root_with_use_counts(self, ct_term: HETerm):
        """Evaluate one ciphertext DAG and free ``env`` entries as operands go dead."""
        remaining = _build_remaining_uses(ct_term)
        for t in _toy_post_order(ct_term):
            self.env[t] = self.eval(t)
            for c in t.cs:
                if isinstance(c, HETerm):
                    remaining[c] -= 1
                    if remaining[c] == 0:
                        self.env.pop(c, None)

    def _as_np_vec(self, v):
        """Return a 1-D ``float64`` vector; avoid ``astype`` when already ``float64``."""
        if isinstance(v, np.ndarray):
            if v.dtype == np.float64:
                return v
            return v.astype(np.float64, copy=False)
        return np.asarray(v, dtype=np.float64)

    def eval_mask(self, term):
        """
        Evaluate a mask operation.

        Args:
            term: The mask term to evaluate

        Returns:
            The mask value
        """
        return term.cs[0]

    def eval_pack(self, term):
        """
        Evaluate a pack operation for tensor data.

        Packs tensor data according to the specified layout and returns
        the packed vector at the given packing index.

        Args:
            term: The pack term containing layout and metadata

        Returns:
            The packed vector at the specified index
        """
        layout = term.cs[0]
        if (layout.term, layout) not in self.input_cache:
            tensor = self._dense_eval_for_tensor_term(layout.term)
            self.input_cache[(layout.term, layout)] = _packed_ct_lists_to_float64(
                apply_layout(tensor, layout)
            )

        # get packing index and return packed vector
        # Parse metadata: "packing_idx rot:rot_amt" or just "packing_idx"
        metadata_parts = term.metadata.split()
        packing_idx = int(metadata_parts[0])
        vector = self.input_cache[(layout.term, layout)][packing_idx]

        # Check if this pack has pre-rotation metadata
        rot_amt = None
        for part in metadata_parts:
            if part.startswith("rot:"):
                rot_amt = int(part.split(":")[1])
                break

        # Apply rotation during packing if specified (cheaper than homomorphic rotation)
        if rot_amt is not None:
            # positive rot_amt means left rotate (same as eval_rot):
            # out[i] = vec[(rot_amt + i) % n] == np.roll(vec, -rot_amt)
            return np.roll(vector, -rot_amt)
        return vector

    def eval_pack_punctured(self, term):
        """
        Evaluate a pack operation for tensor data.

        Packs tensor data according to the specified layout and returns
        the packed vector at the given packing index.

        Args:
            term: The pack term containing layout and metadata

        Returns:
            The packed vector at the specified index
        """
        layout = term.cs[0]
        if (layout.term, layout) not in self.input_cache:
            tensor = self._dense_eval_for_tensor_term(layout.term)
            self.input_cache[(layout.term, layout)] = _packed_ct_lists_to_float64(
                apply_punctured_layout(tensor, layout)
            )

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self.input_cache[(layout.term, layout)][packing_idx]

    def eval_cs_pack(self, term):
        layout = term.cs[1]
        if (layout.term, layout) not in self.input_cache:
            tensor = self.inputs[term.cs[0]]
            self.input_cache[(layout.term, layout)] = _packed_ct_lists_to_float64(
                apply_layout(tensor, layout)
            )

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self.input_cache[(layout.term, layout)][packing_idx]

    def eval_const(self, term):
        layout = term.cs[0]
        vector = np.full(self.n, term.cs[1], dtype=np.float64)
        packed = _packed_ct_lists_to_float64(apply_layout(vector, layout))
        return packed[0]

    def eval_indices(self, term):
        tensor = self.inputs[term.cs[0].cs[0]]
        vector = np.zeros(self.n, dtype=np.float64)
        for filter_index, position, _ in term.cs[1]:
            vector[position[0]] = tensor[
                filter_index[0], filter_index[1], filter_index[2]
            ]
        return vector

    def eval_rot(self, term):
        """positive == left rotate"""
        vector = self.env[term.cs[0]]
        return np.roll(vector, -int(term.cs[1]))

    def eval_add(self, term):
        return self.env[term.cs[0]] + self.env[term.cs[1]]

    def eval_sub(self, term):
        return self.env[term.cs[0]] - self.env[term.cs[1]]

    def eval_mul(self, term):
        return self.env[term.cs[0]] * self.env[term.cs[1]]

    def _apply_poly_to_vector(self, vec, poly_func, poly_channel=None):
        """Apply POLY_* element-wise to a 1D vector. Uses self.inputs for batchnorm."""
        vec = self._as_np_vec(vec)
        if poly_func is None or poly_func == "identity":
            return vec
        if poly_func == "relu_exact" or poly_func == "relu":
            return np.where(vec > 0, vec, 0.0)
        if poly_func == "silu":
            # Plaintext exact SiLU for PolyCall("silu", ...) with clipped sigmoid input.
            # Matches the prior implementation:
            #   out = v * sigmoid(clip(v, -40, 40))
            x_clip = np.clip(vec, -40.0, 40.0)
            sig = 1.0 / (1.0 + np.exp(-x_clip))
            return vec * sig
        if (
            isinstance(poly_func, tuple)
            and len(poly_func) >= 5
            and poly_func[0] == "batchnorm"
        ):
            _, mean_key, var_key, gamma_key, beta_key = poly_func[:5]
            eps = float(poly_func[5]) if len(poly_func) > 5 else 1e-5
            mean = np.asarray(self.inputs[mean_key]).flatten()
            var = np.asarray(self.inputs[var_key]).flatten()
            gamma = np.asarray(self.inputs[gamma_key]).flatten()
            beta = np.asarray(self.inputs[beta_key]).flatten()
            # Use per-channel params when poly_channel is set (one CT per channel); else first channel
            ch = 0 if poly_channel is None else min(int(poly_channel), len(mean) - 1)
            m, vv, g, b = (
                float(mean[ch]),
                float(var[ch]),
                float(gamma[ch]),
                float(beta[ch]),
            )
            inv_std = 1.0 / np.sqrt(vv + eps)
            return g * (vec - m) * inv_std + b
        if isinstance(poly_func, (list, tuple)) and len(poly_func) > 0:
            try:
                coeffs = [float(c) for c in poly_func]
            except (TypeError, ValueError):
                return vec
            # coeffs are in ascending power order: y = sum_i c[i] * x^i
            out = np.zeros_like(vec, dtype=np.float64)
            pow_x = np.ones_like(vec, dtype=np.float64)
            for c in coeffs:
                out += c * pow_x
                pow_x *= vec
            return out

    def eval_poly(self, term):
        """Apply the actual POLY function when term.poly_func is set (from lowering); else identity."""
        vec = self.env[term.cs[0]]
        metadata = term.cs[1]
        poly_func = metadata.get("poly_func", None)
        poly_channel = metadata.get("poly_channel", None)
        return self._apply_poly_to_vector(vec, poly_func, poly_channel=poly_channel)

    def eval_rescale(self, term):
        """Evaluate a rescale operation.

        Args:
            term: The rescale term to evaluate

        Returns:
            The rescaled vector
        """
        vector = self.env[term.cs[0]]
        scale_exp = term.cs[1]  # The exponent (e.g., 14 for 2^14)
        scale_value = 2**scale_exp  # Compute 2^scale_exp
        return vector / scale_value

    def eval(self, term):
        match term.op:
            case HEOp.CS:
                # If the child term is not in env yet, evaluate it first
                # This can happen with new PACK terms created by optimizations
                if term.cs[0] not in self.env:
                    self.env[term.cs[0]] = self.eval(term.cs[0])
                return self.env[term.cs[0]]
            case HEOp.MASK:
                return self.eval_mask(term)
            case HEOp.PACK:
                # Evaluate and cache the result
                if term not in self.env:
                    self.env[term] = self.eval_pack(term)
                return self.env[term]
            case HEOp.PUNCTURED_PACK:
                return self.eval_pack_punctured(term)
            case HEOp.CS_PACK:
                return self.eval_cs_pack(term)
            case HEOp.CONST:
                return self.eval_const(term)
            case HEOp.INDICES:
                return self.eval_indices(term)
            case HEOp.ROT:
                return self.eval_rot(term)
            case HEOp.ADD:
                return self.eval_add(term)
            case HEOp.SUB:
                return self.eval_sub(term)
            case HEOp.MUL:
                return self.eval_mul(term)
            case HEOp.POLY_CALL:
                return self.eval_poly(term)
            case HEOp.RESCALE:
                return self.eval_rescale(term)
            case HEOp.ZERO_MASK:
                return np.zeros(self.n, dtype=np.float64)
            case _:
                raise NotImplementedError(term.op)

    def run(self):
        results = []
        items = list(self.circuit_ir.items())
        n_kernels = len(items)
        progress = os.environ.get("ROTOM_PROGRESS_BARS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        kernel_log_path = os.environ.get("ROTOM_TOY_KERNEL_LOG", "").strip()

        pbar = None
        if progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    items,
                    desc="Toy kernels",
                    unit="kernel",
                    dynamic_ncols=True,
                )
            except ImportError:
                pbar = None

        loop = pbar if pbar is not None else items
        for k_idx, (term, cts) in enumerate(loop):
            if kernel_log_path:
                line = f"{k_idx + 1}/{n_kernels} {term!r}\n"
                with open(kernel_log_path, "a", encoding="utf-8") as lf:
                    lf.write(line)
                    lf.flush()
            if pbar is not None:
                pbar.set_postfix_str(str(term)[:100], refresh=False)

            results = []
            # Sort by ciphertext index to ensure consistent ordering
            for ct_idx in sorted(cts.keys()):
                ct = cts[ct_idx]
                if isinstance(ct, list):
                    for c in ct:
                        self._eval_ct_root_with_use_counts(c)
                        results.append(self.env.pop(c))
                else:
                    self._eval_ct_root_with_use_counts(ct)
                    results.append(self.env.pop(ct))

            # Evaluate the tensor computation to get the expected result
            # skip checks for split rolls, replicate
            if term.op in [KernelOp.SPLIT_ROLL, KernelOp.REPLICATE]:
                continue

            if getattr(self.args, "skip_toy_eval_checks", False):
                continue

            # Evaluate the tensor computation to get the expected result
            eval_result = self._dense_eval_for_tensor_term(term.layout.term)
            if term.op == KernelOp.PUNCTURED_TENSOR:
                expected = apply_punctured_layout(eval_result, term.layout)
            elif term.op == KernelOp.CONST:
                vector = np.array([eval_result] * self.n)
                expected = apply_layout(vector, term.layout)
            else:
                expected = apply_layout(eval_result, term.layout)

            # Check if values are close instead of exact equality
            all_close = True
            max_diff = 0.0

            for expected_vec, result_vec in zip(expected, results):
                if not np.allclose(expected_vec, result_vec, rtol=1e-2, atol=1e-2):
                    all_close = False
                    diff = np.array(expected_vec) - np.array(result_vec)
                    max_diff = max(max_diff, np.max(np.abs(diff)))

            if not all_close:
                print("expected:")
                for expected_vec in expected:
                    print(expected_vec)
                print()

                print("result:")
                for result_vec in results:
                    print(result_vec)
                print()

                print("diff:")
                for expected_vec, result_vec in zip(expected, results):
                    print([e - r for e, r in zip(expected_vec, result_vec)])
                print()
                print("expected layout:", term.layout)

                assert all_close, f"Values not close enough. Max diff: {max_diff}"

        return results

    def fuzz(self):
        results = []
        for term, cts in self.circuit_ir.items():
            results = []
            # Sort by ciphertext index to ensure consistent ordering
            for ct_idx in sorted(cts.keys()):
                ct = cts[ct_idx]
                if isinstance(ct, list):
                    for c in ct:
                        self._eval_ct_root_with_use_counts(c)
                        results.append(self.env.pop(c))
                else:
                    self._eval_ct_root_with_use_counts(ct)
                    results.append(self.env.pop(ct))

            expected = apply_layout(
                self._dense_eval_for_tensor_term(term.layout.term), term.layout
            )
            assert results == expected

            # check that results match up
            if term.layout.term.op == TensorOp.TENSOR:
                expected = apply_layout(
                    self.inputs[term.layout.term.cs[0]], term.layout
                )
                assert results[: len(expected)] == expected
            else:
                expected = apply_layout(
                    self._dense_eval_for_tensor_term(term.layout.term), term.layout
                )
                assert results == expected
            print("check passed:", term)
            print()

        return results
