import numpy as np

np.set_printoptions(legacy="1.25")

from frontends.tensor import TensorOp
from ir.he import HEOp
from ir.kernel import KernelOp
from util.layout_util import *


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
            tensor = layout.term.eval(self.inputs)
            # apply layout to tensor
            packed_tensor = apply_layout(tensor, layout)
            self.input_cache[(layout.term, layout)] = packed_tensor

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
            # Rotate the vector: positive rot_amt means left rotate (same as eval_rot)
            # rotated[i] = original[(rot_amt + i) % n]
            n = len(vector)
            vector = [vector[(rot_amt + i) % n] for i in range(n)]

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
            tensor = layout.term.eval(self.inputs)
            # apply layout to tensor
            packed_tensor = apply_punctured_layout(tensor, layout)
            self.input_cache[(layout.term, layout)] = packed_tensor

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self.input_cache[(layout.term, layout)][packing_idx]

    def eval_cs_pack(self, term):
        layout = term.cs[1]
        if (layout.term, layout) not in self.input_cache:
            tensor = self.inputs[term.cs[0]]
            # apply layout to tensor
            packed_tensor = apply_layout(tensor, layout)
            self.input_cache[(layout.term, layout)] = packed_tensor

        # get packing index and return packed vector
        packing_idx = int(term.metadata.split()[0])
        return self.input_cache[(layout.term, layout)][packing_idx]

    def eval_const(self, term):
        layout = term.cs[0]
        vector = np.array([term.cs[1]] * self.n)
        packed_tensor = apply_layout(vector, layout)
        return packed_tensor[0]

    def eval_indices(self, term):
        tensor = self.inputs[term.cs[0].cs[0]]
        vector = [0] * self.n
        for filter_index, position, _ in term.cs[1]:
            vector[position[0]] = tensor[
                filter_index[0], filter_index[1], filter_index[2]
            ]
        return vector

    def eval_rot(self, term):
        """positive == left rotate"""
        vector = self.env[term.cs[0]].copy()
        return [vector[(term.cs[1] + i) % len(vector)] for i in range(len(vector))]

    def eval_add(self, term):
        return [a + b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])]

    def eval_sub(self, term):
        return [a - b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])]

    def eval_mul(self, term):
        return [a * b for a, b in zip(self.env[term.cs[0]], self.env[term.cs[1]])]

    def _apply_poly_to_vector(self, vec, poly_func, poly_channel=None):
        """Apply polynomial descriptor element-wise to a list of values. Uses self.inputs for batchnorm."""
        if poly_func is None or poly_func == "identity":
            return list(vec)
        if poly_func == "relu_exact" or poly_func == "relu":
            return [float(v) if v > 0 else 0.0 for v in vec]
        if poly_func == "silu":
            return [
                float(v * (1.0 / (1.0 + np.exp(-np.clip(v, -20, 20))))) for v in vec
            ]
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
            return [float(g * (x - m) * inv_std + b) for x in vec]
        if isinstance(poly_func, (list, tuple)) and len(poly_func) > 0:
            try:
                coeffs = [float(c) for c in poly_func]
            except (TypeError, ValueError):
                return list(vec)
            out = []
            for x in vec:
                y = 0.0
                for i, c in enumerate(coeffs):
                    y += c * (float(x) ** i)
                out.append(y)
            return out
        return list(vec)

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
        return [v / scale_value for v in vector]

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
                return [0] * self.n
            case _:
                raise NotImplementedError(term.op)

    def run(self):
        results = []
        for term, cts in self.circuit_ir.items():
            results = []
            # Sort by ciphertext index to ensure consistent ordering
            for ct_idx in sorted(cts.keys()):
                ct = cts[ct_idx]
                if isinstance(ct, list):
                    for c in ct:
                        for ct_term in c.post_order():
                            self.env[ct_term] = self.eval(ct_term)
                        results.append(self.env[ct_term])
                else:
                    for ct_term in ct.post_order():
                        self.env[ct_term] = self.eval(ct_term)
                    results.append(self.env[ct_term])

            # Evaluate the tensor computation to get the expected result
            eval_result = term.layout.term.eval(self.inputs)
            if term.op == KernelOp.PUNCTURED_TENSOR:
                expected = apply_punctured_layout(eval_result, term.layout)
            elif term.op == KernelOp.CONST:
                vector = np.array([eval_result] * self.n)
                expected = apply_layout(vector, term.layout)
            else:
                expected = apply_layout(eval_result, term.layout)

            # skip checks for split rolls, replicate
            if term.op in [KernelOp.SPLIT_ROLL, KernelOp.REPLICATE]:
                continue

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
                        for ct_term in c.post_order():
                            self.env[ct_term] = self.eval(ct_term)
                        results.append(self.env[ct_term])
                else:
                    for ct_term in ct.post_order():
                        self.env[ct_term] = self.eval(ct_term)
                    results.append(self.env[ct_term])

            expected = apply_layout(term.layout.term.eval(self.inputs), term.layout)
            assert results == expected

            # check that results match up
            if term.layout.term.op == TensorOp.TENSOR:
                expected = apply_layout(
                    self.inputs[term.layout.term.cs[0]], term.layout
                )
                assert results[: len(expected)] == expected
            else:
                expected = apply_layout(term.layout.term.eval(self.inputs), term.layout)
                assert results == expected
            print("check passed:", term)
            print()

        return results
