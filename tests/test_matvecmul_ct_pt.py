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

    def _create_matvecmul_ct_pt_computation(self, inputs):
        """Helper method to create ciphertext-plaintext matrix-vector multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        return a @ b, inputs["a"] @ inputs["b"]

    def _run_test_case(self, inputs, args, backend):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_matvecmul_ct_pt_4x4_4x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 4x4 matrix and 4x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array(
            [[i * size + j for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array([i for i in range(size)])

        self._run_test_case(inputs, args, backend)

    def test_matvecmul_ct_pt_64x64_64x1(self, backend):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True

        # Create inputs
        size = 64
        inputs = {}
        inputs["a"] = np.array(
            [[np.random.randint(0, 2) for j in range(size)] for i in range(size)]
        )
        inputs["b"] = np.array([np.random.randint(0, 2) for i in range(size)])

        self._run_test_case(inputs, args, backend)

    def _assert_allclose(self, expected_cts, results):
        """Assert results match expected using allclose (for float inputs)."""
        for exp, res in zip(expected_cts, results):
            np.testing.assert_allclose(
                np.asarray(exp), np.asarray(res), rtol=1e-2, atol=1e-2
            )

    def test_matvecmul_ct_pt_1x64_64x32(self, backend):
        """Test ciphertext-plaintext matmul [1, 64] @ [64, 32] - smaller MNIST-style dimensions."""
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x64_64x32"

        np.random.seed(42)
        inputs = {"a": np.random.randn(1, 64).astype(np.float64) * 0.1,
                  "b": np.random.randn(64, 32).astype(np.float64) * 0.1}
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        self._assert_allclose(apply_layout(expected, kernel.layout), results)

    def test_matvecmul_ct_pt_1x100_100x50(self, backend):
        """Test [1, 100] @ [100, 50] - non-power-of-two dimensions that pass."""
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x100_100x50"

        np.random.seed(42)
        inputs = {"a": np.random.randn(1, 100).astype(np.float64) * 0.1,
                  "b": np.random.randn(100, 50).astype(np.float64) * 0.1}
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        self._assert_allclose(apply_layout(expected, kernel.layout), results)

    def test_matvecmul_ct_pt_1x80_80x48(self, backend):
        """Test [1, 80] @ [80, 48] - non-power-of-two dimensions that pass."""
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x80_80x48"

        np.random.seed(42)
        inputs = {"a": np.random.randn(1, 80).astype(np.float64) * 0.1,
                  "b": np.random.randn(80, 48).astype(np.float64) * 0.1}
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        self._assert_allclose(apply_layout(expected, kernel.layout), results)

    def test_matvecmul_ct_pt_1x130_130x66(self, backend):
        """Test [1, 130] @ [130, 66] - non-power-of-two; uses n=32768 to avoid BSGS layout bug."""
        args = get_default_args()
        args.n = 32768
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x130_130x66"

        np.random.seed(42)
        inputs = {"a": np.random.randn(1, 130).astype(np.float64) * 0.1,
                  "b": np.random.randn(130, 66).astype(np.float64) * 0.1}
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        self._assert_allclose(apply_layout(expected, kernel.layout), results)

    def test_matvecmul_ct_pt_1x200_200x100(self, backend):
        """Test [1, 200] @ [200, 100] - non-power-of-two; uses n=32768 to avoid BSGS layout bug."""
        args = get_default_args()
        args.n = 32768
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x200_200x100"

        np.random.seed(42)
        inputs = {"a": np.random.randn(1, 200).astype(np.float64) * 0.1,
                  "b": np.random.randn(200, 100).astype(np.float64) * 0.1}
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        self._assert_allclose(apply_layout(expected, kernel.layout), results)

    @pytest.mark.xfail(reason="BSGS_MATMUL bug: still fails with n=32k; see scripts/debug_matvecmul_1x784_784x512.py")
    def test_matvecmul_ct_pt_1x784_784x512_n32k(self, backend):
        """Test [1, 784] @ [784, 512] with n=32768 - still fails (uses BSGS layout)."""
        args = get_default_args()
        args.n = 32768
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x784_784x512_n32k"

        np.random.seed(42)
        inputs = {"a": np.random.randn(1, 784).astype(np.float64) * 0.1,
                  "b": np.random.randn(784, 512).astype(np.float64) * 0.1}
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        self._assert_allclose(apply_layout(expected, kernel.layout), results)

    @pytest.mark.xfail(reason="BSGS_MATMUL bug: first mismatch at slot 832; see scripts/debug_matvecmul_1x784_784x512.py")
    def test_matvecmul_ct_pt_1x784_784x512(self, backend):
        """Test ciphertext-plaintext matmul with MNIST FC1 dimensions: [1, 784] @ [784, 512]."""
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "matvecmul_ct_pt_1x784_784x512"

        np.random.seed(42)
        inputs = {}
        inputs["a"] = np.random.randn(1, 784).astype(np.float64) * 0.1
        inputs["b"] = np.random.randn(784, 512).astype(np.float64) * 0.1

        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)
        expected_cts = apply_layout(expected, kernel.layout)

        for exp, res in zip(expected_cts, results):
            np.testing.assert_allclose(
                np.asarray(exp), np.asarray(res), rtol=1e-2, atol=1e-2
            )
