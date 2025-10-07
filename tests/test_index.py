"""
Test suite for tensor indexing operations.

This module tests tensor indexing operations on ciphertext tensors
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


class TestTensorIndexing:
    """Test tensor indexing operations."""
    
    def _create_index_computation(self, inputs, i):
        """Helper method to create tensor indexing computation."""
        a = TensorTerm.Tensor("a", list(inputs["a"].shape), True)
        return a[i]
    
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


    def test_index_4x4_matrix_row_0(self):
        """Test indexing first row of 4x4 matrix."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_1"
        
        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])
        
        # Generate test case
        tensor_ir = self._create_index_computation(inputs, 0)
        self._run_test_case(tensor_ir, inputs, args)

    def test_index_4x4_matrix_row_1(self):
        """Test indexing first row of 4x4 matrix (variant 1)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_2"

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

        # Generate test case
        tensor_ir = self._create_index_computation(inputs, 0)
        self._run_test_case(tensor_ir, inputs, args)

    def test_index_4x4_matrix_row_2(self):
        """Test indexing first row of 4x4 matrix (variant 2)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_3"

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

        # Generate test case
        tensor_ir = self._create_index_computation(inputs, 0)
        self._run_test_case(tensor_ir, inputs, args)

    def test_index_4x4_matrix_row_3(self):
        """Test indexing first row of 4x4 matrix (variant 3)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.benchmark = "index_4"

        # Create inputs
        size = 4
        inputs = {}
        inputs["a"] = np.array([[i * size + j for j in range(size)] for i in range(size)])

        # Generate test case
        tensor_ir = self._create_index_computation(inputs, 0)
        self._run_test_case(tensor_ir, inputs, args)


    def test_index_4x4_4x32_with_reshape_permute(self):
        """Test indexing with reshape and permute operations."""
        # Create args
        args = get_default_args()
        args.n = 32
        args.benchmark = "index_5"

        # Create inputs
        inputs = {}
        inputs["a"] = np.array([[i * 4 + j for j in range(4)] for i in range(4)])
        inputs["b"] = np.array([[i * 4 + j for j in range(32)] for i in range(4)])

        # Generate test case
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 32], False)
        s = a @ b
        s2 = s.reshape(1, {1: 4, 2: 8}).permute({0: 1, 1: 0, 2: 2})
        tensor_ir = s2[0]
        
        self._run_test_case(tensor_ir, inputs, args)
