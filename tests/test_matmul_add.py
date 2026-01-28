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


class TestMatrixMultiplicationAddition:
    """Test matrix multiplication followed by addition operations."""

    def _create_matmul_add_computation(self, inputs):
        """Helper method to create matrix multiplication + addition computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        c = TensorTerm.Tensor("c", list(inputs["c"].shape), False)
        return (a @ b) + c, (inputs["a"] @ inputs["b"]) + inputs["c"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matmul_add_computation(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        for k in kernel.post_order():
            print(k)
        print()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_matmul_add_size_4_1(self, backend):
        """Test matrix multiplication + addition with 4x4 matrices (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_add_1"

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["c"] = np.array([j for j in range(size)])

        self._run_test_case(inputs, args, backend)

    def test_matmul_add_size_4_2(self, backend):
        """Test matrix multiplication + addition with 4x4 matrices (test case 2 with rolls)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_add_2"
        args.rolls = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["c"] = np.array([j for j in range(size)])

        self._run_test_case(inputs, args, backend)

    def test_matmul_add_size_8_1(self, backend):
        """Test matrix multiplication + addition with 8x8 matrices (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_add_3"

        # Create inputs
        size = 8
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["c"] = np.array([j for j in range(size)])

        self._run_test_case(inputs, args, backend)

    def test_matmul_add_size_8_2(self, backend):
        """Test matrix multiplication + addition with 8x8 matrices (test case 2 with rolls)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_add_4"
        args.rolls = True

        # Create inputs
        size = 8
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["c"] = np.array([j for j in range(size)])

        self._run_test_case(inputs, args, backend)


    def test_matmul_add_size_16(self, backend):
        """Test matrix multiplication + addition with 16x16 matrices (test case 2 with rolls)."""
        # Create args
        args = get_default_args()
        args.n = 256
        args.benchmark = "matmul_add_5"
        args.rolls = True

        # Create inputs
        size = 16
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["c"] = np.array([j for j in range(size)])

        self._run_test_case(inputs, args, backend)

