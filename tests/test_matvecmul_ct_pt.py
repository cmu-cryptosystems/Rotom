"""
Test suite for ciphertext-plaintext matrix-vector multiplication operations.

This module tests matrix-vector multiplication operations between ciphertext matrices
and plaintext vectors in the Rotom homomorphic encryption system.
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestMatrixVectorMultiplicationCiphertextPlaintext:
    """Test matrix-vector multiplication between ciphertext matrices and plaintext vectors."""
    
    def _create_matvecmul_ct_pt_computation(self, inputs):
        """Helper method to create ciphertext-plaintext matrix-vector multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        return a @ b, inputs["a"] @ inputs["b"]
    
    def _run_test_case(self, inputs, args):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matvecmul_ct_pt_computation(inputs)
        
        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()
        
        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert expected_cts == results


    def test_matvecmul_ct_pt_4x4_4x1(self):
        """Test ciphertext-plaintext matrix-vector multiplication with 4x4 matrix and 4x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        
        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
        inputs["b"] = np.array([i for i in range(size)])
        
        print(inputs["a"].shape)
        print(inputs["b"].shape)
        
        self._run_test_case(inputs, args)


    def test_matvecmul_ct_pt_64x64_64x1(self):
        """Test ciphertext-plaintext matrix-vector multiplication with 64x64 matrix and 64x1 vector."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        
        # Create inputs
        size = 64
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
        inputs["b"] = np.array([i for i in range(size)])
        
        print(inputs["a"].shape)
        print(inputs["b"].shape)
        
        self._run_test_case(inputs, args)
