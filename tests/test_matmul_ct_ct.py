"""
Test suite for ciphertext-ciphertext matrix multiplication operations.

This module tests matrix multiplication operations between two ciphertext tensors
in the Rotom homomorphic encryption system.
"""

import random

import numpy as np

from frontends.tensor import TensorTerm
from ir.dim import *
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args


class TestMatrixMultiplicationCiphertextCiphertext:
    """Test matrix multiplication between ciphertext tensors."""

    def _create_matmul_ct_ct_computation(self, inputs):
        """Helper method to create ciphertext-ciphertext matrix multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), True)
        return a @ b, inputs["a"] @ inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, _ = self._create_matmul_ct_ct_computation(inputs)
        expected_cts, results, _, _ = run_compiler_and_backend(
            backend, tensor_ir, inputs, args
        )
        assert_results_equal(expected_cts, results, backend)

    def test_matmul_ct_ct_4x4(self, backend):
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

        self._run_test_case(inputs, args, backend)

    def test_matmul_ct_ct_8x8(self, backend):
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

        self._run_test_case(inputs, args, backend)

    def test_matmul_ct_ct_64x64(self, backend):
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

        self._run_test_case(inputs, args, backend)

    def test_matmul_ct_ct_4x4_rolls(self, backend):
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

        self._run_test_case(inputs, args, backend)

    def test_matmul_ct_ct_8x8_rolls(self, backend):
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

        self._run_test_case(inputs, args, backend)

    def test_matmul_ct_ct_16x16_rolls(self, backend):
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

        self._run_test_case(inputs, args, backend)

    def test_matmul_ct_ct_64x64_rolls(self, backend):
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

        self._run_test_case(inputs, args, backend)
