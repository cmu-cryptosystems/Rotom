"""
Fuzzing utilities for testing FHE kernel implementations.

This module provides fuzzing capabilities to test the correctness of
generated FHE kernels by running them with random inputs and comparing
results against reference implementations.

Key classes:
- Fuzz: Main fuzzing class for testing kernel correctness
"""

import numpy as np

from backends.toy import Toy
from frontends.tensor import TensorOp
from lower.lower import Lower
from tests.test_util import get_default_args

from .layout_util import apply_layout


class Fuzz:
    """Fuzzing class for testing FHE kernel correctness.

    This class provides methods to test the correctness of generated FHE kernels
    by running them with random inputs and comparing results against reference
    implementations. It supports various operation types including binary operations,
    matrix multiplication, and other tensor operations.

    The fuzzing process:
    1. Generates random input data
    2. Applies layouts to create ciphertext inputs
    3. Runs the kernel through the FHE backend
    4. Compares results with expected outputs

    Args:
        n: HE vector size for the fuzzing tests
    """

    def __init__(self, n):
        self.n = n

        self.inputs = {}
        self.cache = {}

    def gen_random_np_array(self, shape):
        """Generate a random numpy array of the given shape.

        Args:
            shape: Tuple representing the shape of the array

        Returns:
            numpy.ndarray: Random array filled with 0s and 1s
        """
        return np.random.randint(0, 1, shape)

    def gen_random_inputs(self, shapes):
        """Generate a list of random numpy arrays of the given shapes.

        Args:
            shapes: List of shape tuples for the input arrays
        """
        for i, shape in enumerate(shapes):
            self.inputs[i] = self.gen_random_np_array(shape)

    def map_binop_operands(self, kernel):
        """Map inputs to kernel"""
        if kernel.layout.term.cs[0].op == TensorOp.TENSOR:
            if kernel.layout.term.cs[0] not in self.inputs:
                self.inputs[kernel.layout.term.cs[0]] = self.gen_random_np_array(
                    kernel.layout.term.cs[1]
                )
            self.inputs[0] = self.inputs[kernel.layout.term.cs[0]]
        if kernel.layout.term.cs[1].op == TensorOp.TENSOR:
            if kernel.layout.term.cs[1] not in self.inputs:
                self.inputs[kernel.layout.term.cs[1]] = self.gen_random_np_array(
                    kernel.layout.term.cs[1]
                )
            self.inputs[1] = self.inputs[kernel.layout.term.cs[1]]

    def fuzz_add(self, kernel, cs_shapes):
        """Fuzz an add kernel"""
        # create plaintext inputs
        self.gen_random_inputs(cs_shapes)

        # map inputs to kernel
        self.map_binop_operands(kernel)

        # create ciphertext inputs
        circuit_ir = Lower(kernel).run()

        # fuzz kernel
        args = get_default_args()
        args.n = self.n
        toy = Toy(circuit_ir, self.inputs, args)
        result = toy.fuzz()

        # check results
        c = self.inputs[0] + self.inputs[1]

        # apply layout to c
        c = apply_layout(c, kernel.layout)

        # check that results match up
        assert result == c

    def fuzz_matmul(self, kernel, cs_shapes):
        """Fuzz a matmul kernel"""
        # create plaintext inputs
        self.gen_random_inputs(cs_shapes)

        # map inputs to kernel
        self.map_binop_operands(kernel)

        # create ciphertext inputs
        circuit_ir = Lower(kernel).run()

        # fuzz kernel
        args = get_default_args()
        args.n = self.n
        toy = Toy(circuit_ir, self.inputs, args)
        result = toy.fuzz()

        # check results
        c = self.inputs[0] @ self.inputs[1]

        # apply layout to c
        c = apply_layout(c, kernel.layout)

        # check that results match up
        assert result == c

    def fuzz_block_matmul(self, kernel, cs_shapes):
        """Fuzz a matmul kernel"""
        # create plaintext inputs
        self.gen_random_inputs(cs_shapes)

        # map inputs to kernel
        self.map_binop_operands(kernel)

        # create ciphertext inputs
        circuit_ir = Lower(kernel).run()

        # fuzz kernel
        args = get_default_args()
        args.n = self.n
        toy = Toy(circuit_ir, self.inputs, args)
        result = toy.fuzz()

        # check results
        print(self.inputs[0].shape)
        print(self.inputs[1].shape)

        c = self.inputs[0] @ self.inputs[1]

        # apply layout to c
        c = apply_layout(c, kernel.layout)

        # check that results match up
        assert result == c

    def fuzz_kernel(self, kernel, cs_shapes):
        """Fuzz a kernel"""
        print("fuzzing kernel:", kernel)
        match kernel.layout.term.op:
            case TensorOp.TENSOR:
                # generate random inputs
                name = kernel.layout.term.cs[0]
                shape = kernel.layout.term.cs[1]
                self.inputs[name] = self.gen_random_np_array(shape)
            case TensorOp.TRANSPOSE | TensorOp.INDEX:
                pass
            case TensorOp.ADD:
                self.fuzz_add(kernel, cs_shapes)
            case TensorOp.MATMUL:
                self.fuzz_matmul(kernel, cs_shapes)
            case TensorOp.BLOCK_MATMUL:
                self.fuzz_block_matmul(kernel, cs_shapes)
            case _:
                raise NotImplementedError(kernel.layout.term.op)

        print("passed")
