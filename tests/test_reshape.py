"""
Test suite for tensor reshaping operations.

This module tests tensor reshaping and permutation operations on ciphertext tensors
in the Rotom homomorphic encryption system.
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestTensorReshaping:
    """Test tensor reshaping and permutation operations."""

    def _create_reshape_computation(self):
        """Helper method to create tensor reshaping computation."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 32], False)
        s = a @ b
        s2 = s.reshape(1, {1: 4, 2: 8}).permute({0: 1, 1: 0, 2: 2})
        return s2

    def _run_test_case(self, tensor_ir, inputs, args):
        """Helper method to run a test case."""
        # Generate expected result
        expected = tensor_ir.eval(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert expected_cts == results

    def test_reshape_4x4_4x32_with_permute(self):
        """Test tensor reshaping and permutation with 4x4 and 4x32 matrices."""
        # Create args
        args = get_default_args()
        args.n = 32

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
        inputs["b"] = np.array([[i * 4 + j for j in range(32)] for i in range(4)])

        # Generate test case
        tensor_ir = self._create_reshape_computation()
        self._run_test_case(tensor_ir, inputs, args)
