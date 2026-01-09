"""
Test suite for 2D convolution operations.

This module tests 2D convolution operations in the Rotom homomorphic encryption system,
including various filter sizes and input dimensions.
"""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


class TestConvolution2D:
    """Test 2D convolution operations."""

    def _create_convolution_computation(
        self, input_size, input_channel, f_out, f_in, f_h, f_w, stride, padding
    ):
        """Helper method to create convolution computation.

        Convolution example, precursor to resnet:
        https://proceedings.mlr.press/v162/lee22e.html
        """
        input_Tensor = TensorTerm.Tensor(
            "a", [input_channel, input_size, input_size], True
        )
        weight_Tensor = TensorTerm.Tensor("b", [f_out, f_in, f_h, f_w], False)
        output_Tensor = TensorTerm.conv2d(input_Tensor, weight_Tensor, stride, padding)
        return output_Tensor

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

    def test_conv2d_4x4_filter_2x2(self):
        """Test 2D convolution with 4x4 input and 2x2 filter."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_1"
        args.toeplitz = True

        # Create inputs
        dim_size = 4
        f_size = 2
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        )
        inputs["b"] = np.array([[[[1 for i in range(f_size)] for j in range(f_size)]]])

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_4x4_filter_3x3(self):
        """Test 2D convolution with 4x4 input and 3x3 filter."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_2"

        # Create inputs
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        )
        inputs["b"] = np.array([[[[1 for i in range(f_size)] for j in range(f_size)]]])

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_4x4_3channels_filter_3x3(self):
        """Test 2D convolution with 4x4 input, 3 channels, and 3x3 filter."""
        # Create args
        args = get_default_args()
        args.n = 64
        args.rolls = True
        args.benchmark = "conv2d_3"

        # Create inputs
        input_channels = 3
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(input_channels)
            ]
        )
        inputs["b"] = np.array([[[[1 for i in range(f_size)] for j in range(f_size)]]])

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_32x32_3channels_filter_3x3(self):
        """Test 2D convolution with 32x32 input, 3 channels, and 3x3 filter (large scale)."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.benchmark = "conv2d_4"

        # Create inputs
        input_channels = 3
        dim_size = 32
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(input_channels)
            ]
        )
        inputs["b"] = np.array([[[[1 for i in range(f_size)] for j in range(f_size)]]])

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)
