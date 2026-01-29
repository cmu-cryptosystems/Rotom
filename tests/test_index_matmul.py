"""
Test suite for indexed matrix multiplication operations.

This module tests indexed matrix multiplication operations with reshaping and permutation
in the Rotom homomorphic encryption system.
"""

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestIndexedMatrixMultiplication:
    """Test indexed matrix multiplication with reshaping and permutation."""

    def _create_index_matmul_computation(self, index):
        """Helper method to create indexed matrix multiplication computation."""
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 16], False)
        c = TensorTerm.Tensor("c", [4, 16], False)
        s = a @ b
        s2 = s.reshape(1, {1: 4, 2: 4}).permute({0: 1, 1: 2, 2: 0})
        t = a @ c
        t2 = t.reshape(1, {1: 4, 2: 4}).permute({0: 2, 1: 1, 2: 0})

        res = []
        for i in range(4):
            res.append(s2[i] @ t2[i])
        return s2[index] @ t2[index]

    def _run_test_case(self, tensor_ir, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate expected result
        expected = tensor_ir.eval(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_index_matmul_index_0(self, backend):
        """Test indexed matrix multiplication with index 0."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_matmul_1"
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
        inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
        inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

        # Generate test case
        tensor_ir = self._create_index_matmul_computation(0)
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_index_matmul_index_1(self, backend):
        """Test indexed matrix multiplication with index 1."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_matmul_2"
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
        inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
        inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

        # Generate test case
        tensor_ir = self._create_index_matmul_computation(1)
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_index_matmul_4x4_4x16_reshape_index_0(self, backend):
        """Test indexed matrix multiplication with reshape and index 0."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_matmul_3"
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
        inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
        inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

        # Generate test case
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 16], False)
        b = b.reshape(1, {1: 4, 2: 4})
        tensor_ir = a @ b[0]

        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_index_matmul_4x4_4x16_reshape_index_1(self, backend):
        """Test indexed matrix multiplication with reshape and index 1."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_matmul_4"
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
        inputs["b"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])
        inputs["c"] = np.array([[i * 4 + j for j in range(16)] for i in range(4)])

        # Generate test case
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 16], False)
        b = b.reshape(1, {1: 4, 2: 4})
        tensor_ir = a @ b[1]

        self._run_test_case(tensor_ir, inputs, args, backend)
