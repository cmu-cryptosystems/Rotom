"""
Test suite for ciphertext-plaintext matrix multiplication operations.

This module tests matrix multiplication operations between ciphertext and plaintext tensors
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


class TestMatrixMultiplicationCiphertextPlaintext:
    """Test matrix multiplication between ciphertext and plaintext tensors."""
    
    def _create_matmul_ct_pt_computation(self, inputs):
        """Helper method to create ciphertext-plaintext matrix multiplication computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        return a @ b, inputs["a"] @ inputs["b"]
    
    def _run_test_case(self, inputs, args):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_matmul_ct_pt_computation(inputs)
        
        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()
        
        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert expected_cts == results


    def test_matmul_ct_pt_4x4_binary_random(self):
        """Test ciphertext-plaintext matrix multiplication with 4x4 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_ct_pt_1"
        # args.fuzz = True
        
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


    def test_matmul_ct_pt_16x16_binary_random(self):
        """Test ciphertext-plaintext matrix multiplication with 16x16 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_ct_pt_2"
        # args.fuzz = True
        
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


    def test_matmul_ct_pt_64x64_binary_random(self):
        """Test ciphertext-plaintext matrix multiplication with 64x64 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.benchmark = "matmul_ct_pt_3"
        
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


    def test_matmul_ct_pt_64x64_binary_random_variant(self):
        """Test ciphertext-plaintext matrix multiplication with 64x64 binary random matrices (variant)."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.benchmark = "matmul_ct_pt_4"
        
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


    def test_matmul_ct_pt_4x4_4x16_binary_random(self):
        """Test ciphertext-plaintext matrix multiplication with 4x4 and 4x16 binary random matrices."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "matmul_ct_pt_5"
        # args.fuzz = True
        
        # Create inputs
        inputs = {}
        inputs["a"] = np.array(
            [[random.choice(range(2)) for j in range(4)] for i in range(4)]
        )
        inputs["b"] = np.array(
            [[random.choice(range(2)) for j in range(16)] for i in range(4)]
        )
        
        self._run_test_case(inputs, args)
