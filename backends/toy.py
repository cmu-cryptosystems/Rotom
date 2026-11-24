from typing import Any, Literal

import numpy as np

np.set_printoptions(legacy="1.25")

from tqdm import tqdm

from frontends.tensor import TensorOp
from frontends.tensor_evaluator import TensorEvaluator
from ir.he import HEOp, HETerm
from ir.kernel import KernelOp
from util.layout_util import *
from util.silu_polycall_eval import eval_silu_polycall


def _operands(term: HETerm) -> list[HETerm]:
    """The ciphertexts and plaintexts a term is computed from."""
    match term.op:
        case HEOp.CS:
            return [term.cs[0]] if isinstance(term.cs[0], HETerm) else []
        case HEOp.ADD | HEOp.SUB | HEOp.MUL:
            return [term.cs[0], term.cs[1]]
        case HEOp.ROT | HEOp.POLY_CALL | HEOp.RESCALE:
            return [term.cs[0]]
        case (
            HEOp.PACK
            | HEOp.PUNCTURED_PACK
            | HEOp.INDICES
            | HEOp.CS_PACK
            | HEOp.CONST
            | HEOp.MASK
            | HEOp.ZERO_MASK
        ):
            return []
        case _:
            raise NotImplementedError(term.op)


def _toy_post_order(root: HETerm, seen: set[HETerm] | None = None) -> list[HETerm]:
    """Post-order like ``HETerm.helper_post_order``, but through ``HEOp.CS``.

    The stock ``post_order`` stops at a ``CS``, which stands for a ciphertext
    of an earlier kernel. Following it gives a topological order of the whole
    circuit, so a term is evaluated once and dropped after its last use.

    Terms in ``seen`` are skipped and the terms visited are added to it, so
    one set carried across roots orders each term once.

    Implemented with an explicit stack so very deep DAGs (e.g. large ResNet HE IR)
    do not hit Python's recursion limit.
    """
    order: list[HETerm] = []
    seen = set() if seen is None else seen
    stack: list[tuple[HETerm, Literal["enter", "exit"]]] = [(root, "enter")]

    while stack:
        term, phase = stack.pop()
        if phase == "enter":
            if term in seen:
                continue
            seen.add(term)
            stack.append((term, "exit"))
            for operand in reversed(_operands(term)):
                stack.append((operand, "enter"))
        else:
            order.append(term)

    return order


def _flatten_ct_roots(cts: dict) -> list[HETerm]:
    roots: list[HETerm] = []
    for ct_idx in sorted(cts.keys()):
        ct = cts[ct_idx]
        if isinstance(ct, list):
            for c in ct:
                roots.append(c)
        else:
            roots.append(ct)
    return roots


def _schedule(circuit_ir):
    """The order to evaluate a circuit in, and how often each term is read.

    Returns:
        (steps, uses): ``steps`` holds, per kernel, its ciphertexts and the
        terms to evaluate for them that no earlier kernel evaluated. ``uses``
        counts the terms that read each term, plus one for each time a kernel
        lists it as a ciphertext.
    """
    steps = []
    uses: dict[HETerm, int] = {}
    seen: set[HETerm] = set()
    for kernel, cts in circuit_ir.items():
        roots = _flatten_ct_roots(cts)
        order = []
        for root in roots:
            order.extend(_toy_post_order(root, seen))
            uses[root] = uses.get(root, 0) + 1
        for term in order:
            for operand in _operands(term):
                uses[operand] = uses.get(operand, 0) + 1
        steps.append((kernel, roots, order))
    return steps, uses


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

    Attributes:
        circuit_ir: The FHE circuit intermediate representation
        inputs: Dictionary of input tensors
        n: HE vector size
        env: Environment for storing intermediate results
        input_cache: Cache for packed input tensors

    Evaluation:
        :meth:`run` and :meth:`fuzz` evaluate each term of the circuit once,
        kernel by kernel, and drop a value after its last use.
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
        # TensorTerm -> value, for the expected result of each kernel
        self._evaluator = TensorEvaluator()
        self._tensor_env: dict[Any, Any] = {}

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

    def _expected(self, tensor_term):
        """The value of a kernel's tensor term, sharing subterms across kernels."""
        return self._evaluator.eval(tensor_term, self.inputs, self._tensor_env)

    def _release(self, term, uses):
        uses[term] -= 1
        if uses[term] == 0:
            self.env.pop(term, None)

    def kernel_results(self):
        """Evaluate the circuit once, yielding each kernel with its ciphertexts.

        A value stays in ``env`` until every term that reads it has been
        evaluated and every kernel that lists it has been yielded.
        """
        steps, uses = _schedule(self.circuit_ir)
        for kernel, roots, order in steps:
            for term in order:
                self.env[term] = self.eval(term)
                for operand in _operands(term):
                    self._release(operand, uses)
            yield kernel, [self.env[root] for root in roots]
            for root in roots:
                self._release(root, uses)

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
        if layout.term.op == TensorOp.GATHER:
            # One plaintext for one multiply, already in slot order. A network
            # has thousands of them, so none is kept past its use.
            source, weights = layout.term.cs
            return weights.vector(self._dense_eval_for_tensor_term(source))

        pkey = self._pack_cache_key(layout)
        if pkey not in self.input_cache:
            tensor = self._dense_eval_for_tensor_term(layout.term)
            self.input_cache[pkey] = _packed_ct_lists_to_float64(
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
            self.input_cache[pkey] = _packed_ct_lists_to_float64(
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
            self.input_cache[pkey] = _packed_ct_lists_to_float64(
                apply_layout(tensor, layout)
            )

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self._as_np_vec(self.input_cache[pkey][packing_idx])

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
        elif poly_func == "relu_exact" or poly_func == "relu":
            return np.where(vec > 0, vec, 0.0)
        elif poly_func == "silu":
            # Bounds are only on the lowered ``POLY_CALL`` path; this helper is used without
            # metadata in a few tests — use the same default interval as legacy string ``silu``.
            return eval_silu_polycall(vec, -8.0, 8.0, getattr(self, "inputs", None))
        elif poly_func == "hard_swish":
            return vec * np.clip(vec + 3.0, 0.0, 6.0) / 6.0
        else:
            raise NotImplementedError(
                f"Poly func {poly_func!r} not implemented for eval"
            )

    def eval_poly(self, term):
        """Apply the actual POLY function when term.poly_func is set (from lowering); else identity."""
        vec = self.env[term.cs[0]]
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
                if term not in self.env:
                    self.env[term] = self.eval_pack_punctured(term)
                return self.env[term]
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
        total_cts = sum(len(cts) for cts in self.circuit_ir.values())
        pbar = tqdm(total=total_cts, desc="Toy", unit="ct", dynamic_ncols=True)
        for term, results in self.kernel_results():
            pbar.update(len(results))

            # Evaluate the tensor computation to get the expected result
            # skip checks for replicate
            if term.op == KernelOp.REPLICATE:
                continue

            if getattr(self.args, "skip_toy_eval_checks", False):
                continue

            # Evaluate the tensor computation to get the expected result
            eval_result = self._expected(term.layout.term)
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

        pbar.close()
        return results

    def fuzz(self):
        results = []
        for term, results in self.kernel_results():
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
