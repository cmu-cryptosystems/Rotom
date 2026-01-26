"""
Test suite for ciphertext-plaintext matrix-vector multiplication operations.

This module tests matrix-vector multiplication operations between ciphertext matrices
and plaintext vectors in the Rotom homomorphic encryption system.
"""

import random

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestMatrixVectorMultiplicationCiphertextPlaintext:
    """Test matrix-vector multiplication between ciphertext matrices and plaintext vectors."""

    def _create_matvecmul_pt_ct_computation(self, inputs):
        """Helper method to create ciphertext-plaintext matrix-vector multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), False)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), True)
        return a @ b, inputs["a"] @ inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matvecmul_pt_ct_computation(inputs)

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

    def test_matvecmul_pt_ct_4x4_4x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array(
            [[np.random.randint(0, 2) for j in range(4)] for i in range(4)]
        )
        inputs["b"] = np.array([[np.random.randint(0, 2)] for i in range(4)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_pt_ct_8x2_2x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 8
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 2 + j for j in range(2)] for i in range(8)])
        inputs["b"] = np.array([[1] for i in range(2)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_pt_ct_4x2_2x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 4
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 2 + j for j in range(2)] for i in range(4)])
        inputs["b"] = np.array([[1] for i in range(2)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_pt_ct_4x2_2x1_8(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 8
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 2 + j for j in range(2)] for i in range(4)])
        inputs["b"] = np.array([[1] for i in range(2)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_pt_ct_8x4_4x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(8)])
        inputs["b"] = np.array([[i] for i in range(4)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_pt_ct_16x4_4x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(16)])
        inputs["b"] = np.array([[i] for i in range(4)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_pt_ct_1024x256_256x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 8192
        args.rolls = True

        # Create inputs
        inputs = {}
        inputs["a"] = np.array(
            [[np.random.randint(0, 2) for j in range(256)] for i in range(1024)]
        )
        inputs["b"] = np.array([[np.random.randint(0, 2)] for i in range(256)])

        self._run_test_case(inputs, args, backend)
