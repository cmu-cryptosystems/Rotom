"""
Test suite for tensor addition operations.

This module tests the addition operation between ciphertext and plaintext tensors
in the Rotom homomorphic encryption system.
"""

import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import *
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args


class TestTensorAddition:
    """Test tensor addition operations."""

    def _create_add_computation(self, inputs):
        """Helper method to create addition computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        return a + b, inputs["a"] + inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        tensor_ir, _ = self._create_add_computation(inputs)
        expected_cts, results, _, _ = run_compiler_and_backend(
            backend, tensor_ir, inputs, args
        )
        assert_results_equal(expected_cts, results, backend)

    def test_add_ciphertext_plaintext_1(self, backend):
        """Test addition: ciphertext + plaintext (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(list(range(size)))

        self._run_test_case(inputs, args, backend)

    def test_add_ciphertext_plaintext_2(self, backend):
        """Test addition: ciphertext + plaintext (test case 2)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(list(range(size)))

        self._run_test_case(inputs, args, backend)
