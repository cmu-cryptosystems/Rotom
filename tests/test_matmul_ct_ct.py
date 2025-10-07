"""
Test suite for ciphertext-ciphertext matrix multiplication operations.

This module tests matrix multiplication operations between two ciphertext tensors
in the Rotom homomorphic encryption system.
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


class TestMatrixMultiplicationCiphertextCiphertext:
    """Test matrix multiplication between ciphertext tensors."""

    def _create_matmul_ct_ct_computation(self, inputs):
        """Helper method to create ciphertext-ciphertext matrix multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), True)
        return a @ b, inputs["a"] @ inputs["b"]

    def _run_test_case(self, inputs, args):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matmul_ct_ct_computation(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert expected_cts == results

    def test_matmul_ct_ct_4x4(self):
        """Test ciphertext-ciphertext matrix multiplication with 4x4 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_8x8(self):
        """Test ciphertext-ciphertext matrix multiplication with 8x8 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16

        # Create inputs
        size = 8
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_16x16(self):
        """Test ciphertext-ciphertext matrix multiplication with 16x16 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_ct_ct_3"

        # Create inputs
        size = 16
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_16x16(self):
        """Test ciphertext-ciphertext matrix multiplication with 16x16 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_ct_ct_3"

        # Create inputs
        size = 16
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_64x64(self):
        """Test ciphertext-ciphertext matrix multiplication with 64x64 binary random matrices."""
        # Create args
        args = get_default_args()
        args.rolls = True
        args.benchmark = "matmul_ct_ct_4"

        # Create inputs
        size = 64
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_4x4_rolls(self):
        """Test ciphertext-ciphertext matrix multiplication with 4x4 binary random matrices with rolls."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_8x8_rolls(self):
        """Test ciphertext-ciphertext matrix multiplication with 8x8 binary random matrices with rolls."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        size = 8
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_16x16_rolls(self):
        """Test ciphertext-ciphertext matrix multiplication with 16x16 binary random matrices with rolls."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "matmul_ct_ct_7"

        # Create inputs
        size = 16
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)

    def test_matmul_ct_ct_64x64_rolls(self):
        """Test ciphertext-ciphertext matrix multiplication with 64x64 binary random matrices with rolls."""
        # Create args
        args = get_default_args()
        args.rolls = True
        args.benchmark = "matmul_ct_ct_8"

        # Create inputs
        size = 64
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(size)] for i in range(size)]
        )

        self._run_test_case(inputs, args)
