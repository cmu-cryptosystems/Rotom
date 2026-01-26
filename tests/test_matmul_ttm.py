"""
Test suite for matrix multiplication and addition operations.

This module tests combined matrix multiplication and addition operations
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


class TestMatrixMultiplicationTTM:
    """Test matrix multiplication followed by addition operations."""

    def _create_matmul_ttm_computation(self, inputs):
        """Helper method to create matrix multiplication + addition computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), True)
        return (a @ b), inputs["a"] @ inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matmul_ttm_computation(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_matmul_ttm_4x4_16(self, backend):
        """Test matrix multiplication + addition with 4x4 matrices (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[[i * size + j for j in range(size)] for i in range(size)] for k in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args, backend)

    def test_matmul_ttm_4x4_32(self, backend):
        """Test matrix multiplication + addition with 4x4 matrices (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 32
        args.rolls = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[[np.random.randint(0, 2) for j in range(size)] for i in range(size)]
            for k in range(size)]
        )
        inputs["b"] = np.array(
            [[np.random.randint(0, 2) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args, backend)
