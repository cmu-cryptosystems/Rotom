"""
Test suite for tensor rescale operations.

This module tests the exposed rescale operation which divides tensors by powers of 2.
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestRescale:
    """Test tensor rescale operations."""

    def _create_rescale_computation(self, inputs, scale_exp):
        """Helper method to create rescale computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        return a.rescale(scale_exp), inputs["a"] / (2**scale_exp)

    def _run_test_case(self, inputs, scale_exp, args):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_rescale_computation(inputs, scale_exp)

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()

        # Check result - using allclose to handle float/int type differences
        expected_cts = apply_layout(expected, kernel.layout)
        for expected_vec, result_vec in zip(expected_cts, results):
            assert np.allclose(expected_vec, result_vec, rtol=1e-5, atol=1e-5)

    def test_rescale_ciphertext_1(self):
        """Test rescale: ciphertext / 2^14 (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True

        # Create inputs scaled by 2^14
        size = 4
        scale_exp = 14
        inputs = {}
        inputs["a"] = np.array(
            [
                [(i * size + j) * (2**scale_exp) for j in range(size)]
                for i in range(size)
            ]
        )

        self._run_test_case(inputs, 14, args)

    def test_rescale_ciphertext_2(self):
        """Test rescale: ciphertext / 2^10 (test case 2)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True

        # Create inputs scaled by 2^10
        size = 4
        scale_exp = 10
        inputs = {}
        inputs["a"] = np.array(
            [
                [(i * size + j) * (2**scale_exp) for j in range(size)]
                for i in range(size)
            ]
        )

        self._run_test_case(inputs, 10, args)

    def test_rescale_ciphertext_3(self):
        """Test rescale: ciphertext / 2^5 (test case 3)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True

        # Create inputs scaled by 2^5
        size = 2
        scale_exp = 5
        inputs = {}
        inputs["a"] = np.array(
            [
                [(i * size + j) * (2**scale_exp) for j in range(size)]
                for i in range(size)
            ]
        )

        self._run_test_case(inputs, 5, args)
