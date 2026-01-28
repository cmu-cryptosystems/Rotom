"""
Test suite for Strassen's algorithm matrix multiplication.

This module tests Strassen's algorithm matrix multiplication
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


class TestStrassensMatmul:
    """Test Strassen's algorithm matrix multiplication."""

    def _create_strassens_matmul_computation(self, inputs):
        """Helper method to create Strassen's algorithm matrix multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), True)
        return a @ b, inputs["a"] @ inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # set args.strassens to True
        args.strassens = True
        args.rolls = True

        # Generate test case
        tensor_ir, expected = self._create_strassens_matmul_computation(inputs)
        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        for k in kernel.post_order():
            print(k)
        print()

        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # # Check result
        # expected_cts = apply_layout(expected, kernel.layout)
        # assert_results_equal(expected_cts, results, backend)

    def test_strassens_matmul_8x8(self, backend):
        """Test Strassen's algorithm matrix multiplication with 4x4 matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "strassens_matmul_4x4"

        # Create inputs
        size = 8
        inputs = {}
        inputs["a"] = np.array([[1 for j in range(size)] for i in range(size)])
        inputs["b"] = np.array([[2 for j in range(size)] for i in range(size)])

        self._run_test_case(inputs, args, backend)

    def test_strassens_matmul_16x16(self, backend):
        """Test Strassen's algorithm matrix multiplication with 4x4 matrices."""
        # Create args
        args = get_default_args()
        args.n = 64
        args.benchmark = "strassens_matmul_16x16"

        # Create inputs
        size = 16
        inputs = {}
        inputs["a"] = np.array([[1 for j in range(size)] for i in range(size)])
        inputs["b"] = np.array([[2 for j in range(size)] for i in range(size)])

        self._run_test_case(inputs, args, backend)
