import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

np.set_printoptions(legacy="1.25")

from typing import Any, Literal

from frontends.tensor import TensorOp
from ir.he import HEOp, HETerm
from ir.kernel import KernelOp
from util.layout_util import *
from util.silu_polycall_eval import eval_silu_polycall


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


def _packed_ct_rows_to_2d_float64(packed_cts: list) -> np.ndarray:
    """Stack packed ciphertext rows to ``(num_ct, n)`` ``float64`` (no list-of-lists round trip)."""
    if not packed_cts:
        return np.zeros((0, 0), dtype=np.float64)
    if all(
        isinstance(r, np.ndarray) and r.dtype == np.float64 and r.ndim == 1
        for r in packed_cts
    ):
        return np.stack(packed_cts, axis=0)
    return np.asarray(packed_cts, dtype=np.float64)


def _resolve_toy_ct_workers(args, num_roots: int) -> int:
    """Effective worker count for parallel ciphertext-root evaluation (1 = sequential)."""
    w = getattr(args, "toy_ct_workers", None)
    if w is None:
        w = 1
    if w == 0:
        w = min(os.cpu_count() or 1, num_roots)
    return max(1, min(int(w), num_roots))


class Toy:
    """
    Toy Backend for plaintext simulation of FHE circuits.

    This backend executes FHE circuits in plaintext, providing a fast
    simulation environment for development, testing, and debugging. It
    implements the same interface as other backends but performs all
    operations on plaintext data.

    Attributes:
        circuit_ir: The FHE circuit intermediate representation
        inputs: Dictionary of input tensors
        n: HE vector size
        env: Environment for storing intermediate results
        input_cache: Cache for packed input tensors

    Parallelism:
        Set ``args.toy_ct_workers`` (CLI: ``--toy-ct-workers``) to evaluate
        independent ciphertext DAG roots in parallel threads when a kernel has
        multiple roots. ``1`` is sequential; ``0`` caps workers at
        ``min(CPU count, number of roots)``.
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
        # Per-thread env override when evaluating ciphertext roots in parallel.
        self._eval_tls = threading.local()
        # id(TensorTerm) -> dense result of TensorTerm.eval(inputs).  Keyed by
        # object identity (not TensorTerm.__hash__), since __eq__ is hash-based.
        self._tensor_term_dense_by_id: dict[int, Any] = {}

    def _active_env(self) -> dict:
        """Environment dict for the current HE eval (thread-local when parallel)."""
        e = getattr(self._eval_tls, "env", None)
        return self.env if e is None else e

    @staticmethod
    def _pack_cache_key(layout) -> tuple[int, int]:
        """Cache key for packed tensors.

        ``TensorTerm`` and ``Layout`` use hash-based ``__eq__`` only; different
        objects can compare equal and must not share pack slots.
        """
        return (id(layout.term), id(layout))

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
        env = self._active_env()
        remaining = _build_remaining_uses(ct_term)
        for t in _toy_post_order(ct_term):
            env[t] = self.eval(t)
            for c in t.cs:
                if isinstance(c, HETerm):
                    remaining[c] -= 1
                    if remaining[c] == 0:
                        env.pop(c, None)

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
        pkey = self._pack_cache_key(layout)
        if pkey not in self.input_cache:
            tensor = self._dense_eval_for_tensor_term(layout.term)
            self.input_cache[pkey] = _packed_ct_rows_to_2d_float64(
                apply_layout(tensor, layout)
            )

        # get packing index and return packed vector
        # Parse metadata: "packing_idx rot:rot_amt" or just "packing_idx"
        metadata_parts = term.metadata.split()
        packing_idx = int(metadata_parts[0])
        vector = self.input_cache[pkey][packing_idx]

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
        pkey = self._pack_cache_key(layout)
        if pkey not in self.input_cache:
            tensor = self._dense_eval_for_tensor_term(layout.term)
            self.input_cache[pkey] = _packed_ct_rows_to_2d_float64(
                apply_punctured_layout(tensor, layout)
            )

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self._as_np_vec(self.input_cache[pkey][packing_idx])

    def eval_cs_pack(self, term):
        layout = term.cs[1]
        pkey = self._pack_cache_key(layout)
        if pkey not in self.input_cache:
            tensor = self.inputs[term.cs[0]]
            self.input_cache[pkey] = _packed_ct_rows_to_2d_float64(
                apply_layout(tensor, layout)
            )

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self._as_np_vec(self.input_cache[pkey][packing_idx])

    def eval_const(self, term):
        layout = term.cs[0]
        vector = np.full(self.n, term.cs[1], dtype=np.float64)
        packed = _packed_ct_rows_to_2d_float64(apply_layout(vector, layout))
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
        env = self._active_env()
        vector = env[term.cs[0]]
        return np.roll(vector, -int(term.cs[1]))

    def eval_add(self, term):
        env = self._active_env()
        return env[term.cs[0]] + env[term.cs[1]]

    def eval_sub(self, term):
        env = self._active_env()
        return env[term.cs[0]] - env[term.cs[1]]

    def eval_mul(self, term):
        env = self._active_env()
        return env[term.cs[0]] * env[term.cs[1]]

    def _apply_poly_to_vector(self, vec, poly_func, poly_channel=None):
        """Apply POLY_* element-wise to a 1D vector. Uses self.inputs for batchnorm."""
        vec = self._as_np_vec(vec)
        if poly_func is None or poly_func == "identity":
            return vec
        elif poly_func == "relu_exact" or poly_func == "relu":
            return np.where(vec > 0, vec, 0.0)
        elif poly_func == "silu":
            # Bounds are only on the lowered ``POLY_CALL`` path; this helper is used without
            # metadata in a few tests — use the same default interval as legacy string ``silu``.
            return eval_silu_polycall(vec, -8.0, 8.0, getattr(self, "inputs", None))
        else:
            raise NotImplementedError(
                f"Poly func {poly_func!r} not implemented for eval"
            )

    def eval_poly(self, term):
        """Apply the actual POLY function when term.poly_func is set (from lowering); else identity."""
        vec = self._active_env()[term.cs[0]]
        metadata = term.cs[1]
        poly_func = metadata.get("poly_func", None)
        poly_channel = metadata.get("poly_channel", None)
        if poly_func == "silu":
            lo = float(metadata.get("lower_bound", -8.0))
            hi = float(metadata.get("upper_bound", 8.0))
            vec64 = self._as_np_vec(vec).astype(np.float64)
            return eval_silu_polycall(vec64, lo, hi, self.inputs)
        return self._apply_poly_to_vector(vec, poly_func, poly_channel=poly_channel)

    def eval_rescale(self, term):
        """Evaluate a rescale operation.

        Args:
            term: The rescale term to evaluate

        Returns:
            The rescaled vector
        """
        vector = self._active_env()[term.cs[0]]
        scale_exp = term.cs[1]  # The exponent (e.g., 14 for 2^14)
        scale_value = 2**scale_exp  # Compute 2^scale_exp
        return vector / scale_value

    def eval(self, term):
        env = self._active_env()
        match term.op:
            case HEOp.CS:
                # If the child term is not in env yet, evaluate it first
                # This can happen with new PACK terms created by optimizations
                if term.cs[0] not in env:
                    env[term.cs[0]] = self.eval(term.cs[0])
                return env[term.cs[0]]
            case HEOp.MASK:
                return self.eval_mask(term)
            case HEOp.PACK:
                # Evaluate and cache the result
                if term not in env:
                    env[term] = self.eval_pack(term)
                return env[term]
            case HEOp.PUNCTURED_PACK:
                if term not in env:
                    env[term] = self.eval_pack_punctured(term)
                return env[term]
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

    @staticmethod
    def _ordered_ct_roots(cts: dict) -> list[HETerm]:
        roots: list[HETerm] = []
        for ct_idx in sorted(cts.keys()):
            ct = cts[ct_idx]
            if isinstance(ct, list):
                roots.extend(ct)
            else:
                roots.append(ct)
        return roots

    def _eval_ct_roots_ordered(self, roots: list[HETerm]) -> list[Any]:
        workers = _resolve_toy_ct_workers(self.args, len(roots))
        if workers > 1 and len(roots) > 1:

            def _one(c: HETerm):
                self._eval_tls.env = {}
                try:
                    self._eval_ct_root_with_use_counts(c)
                    return self._active_env().pop(c)
                finally:
                    if hasattr(self._eval_tls, "env"):
                        del self._eval_tls.env

            with ThreadPoolExecutor(max_workers=workers) as ex:
                return list(ex.map(_one, roots))
        out: list[Any] = []
        for c in roots:
            self._eval_ct_root_with_use_counts(c)
            out.append(self._active_env().pop(c))
        return out

    def run(self):
        results = []
        for term, cts in self.circuit_ir.items():
            results = []
            roots = self._ordered_ct_roots(cts)
            results = self._eval_ct_roots_ordered(roots)

            # Evaluate the tensor computation to get the expected result
            # skip checks for split rolls, replicate
            if term.op in [KernelOp.SPLIT_ROLL, KernelOp.REPLICATE]:
                continue

            if getattr(self.args, "skip_toy_eval_checks", False):
                continue

            # Evaluate the tensor computation to get the expected result
            eval_result = term.layout.term.eval(self.inputs)
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
                tensor_op = getattr(getattr(term.layout, "term", None), "op", None)
                print("[toy mismatch]")
                print("kernel op:", term.op)
                print("tensor op:", tensor_op)
                print(
                    "layout:",
                    term.layout.layout_str()
                    if hasattr(term.layout, "layout_str")
                    else term.layout,
                )
                print("term:", term.layout.term)
                print("max diff:", max_diff)

                if getattr(self.args, "toy_print_mismatch_vectors", False):
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

                assert (
                    all_close
                ), f"Toy mismatch at {term.op}/{tensor_op}. Max diff: {max_diff}"

        return results

    def fuzz(self):
        results = []
        for term, cts in self.circuit_ir.items():
            roots = self._ordered_ct_roots(cts)
            results = self._eval_ct_roots_ordered(roots)

            expected = apply_layout(
                self._dense_eval_for_tensor_term(term.layout.term), term.layout
            )
            for a, b in zip(results, expected):
                np.testing.assert_allclose(
                    np.asarray(a, dtype=np.float64),
                    np.asarray(b, dtype=np.float64),
                )

            # check that results match up
            if term.layout.term.op == TensorOp.TENSOR:
                expected = apply_layout(
                    self.inputs[term.layout.term.cs[0]], term.layout
                )
                for a, b in zip(results[: len(expected)], expected):
                    np.testing.assert_allclose(
                        np.asarray(a, dtype=np.float64),
                        np.asarray(b, dtype=np.float64),
                    )
            print("check passed:", term)
            print()

        return results
