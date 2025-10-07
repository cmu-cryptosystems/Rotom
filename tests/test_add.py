"""
Test suite for tensor addition operations.

This module tests the addition operation between ciphertext and plaintext tensors
in the Rotom homomorphic encryption system.
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestTensorAddition:
    """Test tensor addition operations."""
    
    def _create_add_computation(self, inputs):
        """Helper method to create addition computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        b = TensorTerm.Tensor("b", list(inputs["b"].shape), False)
        return a + b, inputs["a"] + inputs["b"]
    
    def _run_test_case(self, inputs, args):
        """Helper method to run a test case."""
        # Generate test case
        tensor_ir, expected = self._create_add_computation(inputs)
        
        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = Toy(circuit_ir, inputs, args).run()
        
        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
        assert expected_cts == results
    
    def test_add_ciphertext_plaintext_1(self):
        """Test addition: ciphertext + plaintext (test case 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True
        
        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
        inputs["b"] = np.array(list(range(size)))
        
        self._run_test_case(inputs, args)
    
    def test_add_ciphertext_plaintext_2(self):
        """Test addition: ciphertext + plaintext (test case 2)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.fuzz = True
        
        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
        inputs["b"] = np.array(list(range(size)))
        
        self._run_test_case(inputs, args)
