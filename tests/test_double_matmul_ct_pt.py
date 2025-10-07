"""
Test suite for double matrix multiplication operations.

This module tests double matrix multiplication operations (A @ B @ C) where A is ciphertext
and B, C are plaintext tensors in the Rotom homomorphic encryption system.
"""

import random

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestDoubleMatrixMultiplicationCiphertextPlaintext:
    """Test double matrix multiplication (A @ B @ C) with ciphertext and plaintext tensors."""

    def _create_double_matmul_ct_pt_computation(self, inputs):
        """Helper method to create double matrix multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        c = TensorTerm.Tensor("c", list(inputs["c"].shape), False)
        return a @ b @ c

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

    def test_double_matmul_ct_pt_4x4(self):
        """Test double matrix multiplication with 4x4 matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "double_matmul_ct_pt_1"

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["c"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )

        # Generate test case
        tensor_ir = self._create_double_matmul_ct_pt_computation(inputs)
        self._run_test_case(tensor_ir, inputs, args)

    def test_double_matmul_ct_pt_16x16(self):
        """Test double matrix multiplication with 16x16 matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = False
        args.benchmark = "double_matmul_ct_pt_2"

        # Create inputs
        size = 16
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for i in range(size)] for j in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for i in range(size)] for j in range(size)]
        )
        inputs["c"] = np.array(
            [[random.choice(range(2)) for i in range(size)] for j in range(size)]
        )

        # Generate test case
        tensor_ir = self._create_double_matmul_ct_pt_computation(inputs)
        self._run_test_case(tensor_ir, inputs, args)

    def test_double_matmul_ct_pt_64x64_1(self):
        """Test double matrix multiplication with 64x64 matrices (variant 1)."""
        # Create args
        args = get_default_args()
        args.rolls = True
        args.benchmark = "double_matmul_ct_pt_3"

        # Create inputs
        size = 64
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
        )
        inputs["c"] = np.array(
            [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
        )

        # Generate test case
        tensor_ir = self._create_double_matmul_ct_pt_computation(inputs)
        self._run_test_case(tensor_ir, inputs, args)

    def test_double_matmul_ct_pt_64x64_2(self):
        """Test double matrix multiplication with 64x64 matrices (variant 2)."""
        # Create args
        args = get_default_args()
        args.rolls = True
        args.benchmark = "double_matmul_ct_pt_4"

        # Create inputs
        size = 64
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
        )
        inputs["c"] = np.array(
            [[random.choice(range(2)) for _ in range(size)] for _ in range(size)]
        )

        # Generate test case
        tensor_ir = self._create_double_matmul_ct_pt_computation(inputs)
        self._run_test_case(tensor_ir, inputs, args)
