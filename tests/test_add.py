"""
Test suite for tensor addition operations.

This module tests the addition operation between ciphertext and plaintext tensors
in the Rotom homomorphic encryption system.
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestTensorAddition:
    """Test tensor addition operations."""

    def _create_add_computation(self, inputs):
        """Helper method to create addition computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        return a + b, inputs["a"] + inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_add_computation(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
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
