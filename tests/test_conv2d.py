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

    def test_conv2d_4x4_filter_1x1_pointwise(self):
        """Test 1x1 convolution (pointwise) with 4x4 input."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_pointwise"

        # Create inputs
        dim_size = 4
        f_size = 1
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        )
        inputs["b"] = np.array([[[[2]]]])  # Scale by 2

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_4x4_filter_3x3_random_weights(self):
        """Test 2D convolution with random filter weights (integer values for FHE)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_random_weights"

        # Create inputs with random data (integers for FHE accuracy)
        np.random.seed(42)
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.random.randint(-5, 5, (1, dim_size, dim_size)).astype(float)
        inputs["b"] = np.random.randint(-3, 3, (1, 1, f_size, f_size)).astype(float)

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_4x4_filter_3x3_identity_center(self):
        """Test 2D convolution with identity filter (center weight = 1, rest = 0)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_identity"

        # Create inputs (integer values for FHE)
        np.random.seed(44)
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.random.randint(-10, 10, (1, dim_size, dim_size)).astype(float)
        # Identity filter: only center is 1
        inputs["b"] = np.zeros((1, 1, f_size, f_size))
        inputs["b"][0, 0, 1, 1] = 1.0  # Center position

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    # Note: valid padding is not currently supported in the implementation
    # The current conv2d implementation only supports "same" padding

    def test_conv2d_8x8_filter_4x4(self):
        """Test 2D convolution with 8x8 input and 4x4 filter (larger filter, power of 2)."""
        # Create args
        args = get_default_args()
        args.n = 64
        args.rolls = True
        args.benchmark = "conv2d_4x4_filter"

        # Create inputs
        dim_size = 8
        f_size = 4
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

    def test_conv2d_4x4_multichannel_to_multichannel(self):
        """Test 2D convolution with multiple input and output channels (power of 2)."""
        # Create args
        args = get_default_args()
        args.n = 64
        args.rolls = True
        args.benchmark = "conv2d_multi_in_out"

        # Create inputs
        input_channels = 2
        output_channels = 2  # Must be power of 2
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size + c for i in range(dim_size)] for j in range(dim_size)]
                for c in range(input_channels)
            ]
        )
        # Create filter for each output channel
        inputs["b"] = np.array(
            [
                [[[1 for i in range(f_size)] for j in range(f_size)] for _ in range(input_channels)]
                for _ in range(output_channels)
            ]
        )

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, input_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_4x4_edge_detection_filter(self):
        """Test 2D convolution with edge detection filter (Sobel-like)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_edge_detection"

        # Create inputs
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        # Create a simple gradient input
        inputs["a"] = np.array(
            [
                [[i for i in range(dim_size)] for j in range(dim_size)]
            ]
        )
        # Horizontal edge detection filter
        inputs["b"] = np.array([[
            [[-1, 0, 1],
             [-2, 0, 2],
             [-1, 0, 1]]
        ]])

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_8x8_2channels_filter_3x3_box_blur(self):
        """Test 2D convolution with box blur filter (uniform weights)."""
        # Create args
        args = get_default_args()
        args.n = 128
        args.rolls = True
        args.benchmark = "conv2d_box_blur"

        # Create inputs (integer values for FHE)
        input_channels = 2
        dim_size = 8
        f_size = 3
        padding = "same"
        np.random.seed(45)
        inputs = {}
        inputs["a"] = np.random.randint(-5, 5, (input_channels, dim_size, dim_size)).astype(float)
        # Box blur: all weights = 1 (simpler for FHE)
        inputs["b"] = np.ones((1, 1, f_size, f_size))

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_4x4_zeros_filter(self):
        """Test 2D convolution with all-zero filter (edge case)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_zeros"

        # Create inputs
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.random.randn(1, dim_size, dim_size)
        inputs["b"] = np.zeros((1, 1, f_size, f_size))

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, 1, 1, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)

    def test_conv2d_16x16_4channels_filter_3x3(self):
        """Test 2D convolution with 16x16 input, 4 channels (power of 2 channels)."""
        # Create args
        args = get_default_args()
        args.n = 1024
        args.rolls = True
        args.benchmark = "conv2d_16x16_4ch"

        # Create inputs
        input_channels = 4
        dim_size = 16
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

    def test_conv2d_8x8_depthwise_separable(self):
        """Test depthwise convolution (each channel filtered independently)."""
        # Create args
        args = get_default_args()
        args.n = 256
        args.rolls = True
        args.benchmark = "conv2d_depthwise"

        # Create inputs (integer values for FHE)
        input_channels = 4  # Must be power of 2
        dim_size = 8
        f_size = 3
        padding = "same"
        np.random.seed(43)
        inputs = {}
        inputs["a"] = np.random.randint(-5, 5, (input_channels, dim_size, dim_size)).astype(float)
        # Depthwise: each input channel has its own filter
        inputs["b"] = np.random.randint(-2, 2, (input_channels, 1, f_size, f_size)).astype(float)

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, input_channels, 1, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args)
