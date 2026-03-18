"""
Test suite for 2D convolution operations.

This module tests 2D convolution operations in the Rotom homomorphic encryption system,
including various filter sizes and input dimensions.
"""

import numpy as np
import pytest

from frontends.tensor import TensorTerm
from ir.dim import *
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args


class TestConvolution2D:
    """Test 2D convolution operations."""

    def _create_convolution_computation(
        self, input_size, input_channel, f_out, f_h, f_w, stride, padding
    ):
        """Helper method to create convolution computation.

        Convolution example, precursor to resnet:
        https://proceedings.mlr.press/v162/lee22e.html
        """
        input_Tensor = TensorTerm.Tensor(
            "a", [input_channel, input_size, input_size], True
        )
        weight_Tensor = TensorTerm.Tensor("b", [f_out, input_channel, f_h, f_w], False)
        output_Tensor = TensorTerm.conv2d(input_Tensor, weight_Tensor, stride, padding)
        return output_Tensor

    def _run_test_case(self, tensor_ir, inputs, args, backend):
        """Helper method to run a test case."""
        expected_cts, results, _, _ = run_compiler_and_backend(
            backend, tensor_ir, inputs, args
        )
        assert_results_equal(expected_cts, results, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_4x4_filter_2x2(self, conv_roll, backend):
        """Test 2D convolution with 4x4 input and 2x2 filter."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_1"

        # Create inputs
        dim_size = 4
        f_size = 2
        input_channels = 1
        output_channels = 1
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        )
        inputs["b"] = np.array(
            [[[[i + j * f_size + 1 for i in range(f_size)] for j in range(f_size)]]]
        )

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_4x4_filter_3x3(self, conv_roll, backend):
        """Test 2D convolution with 4x4 input and 3x3 filter."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_2"

        # Create inputs
        dim_size = 4
        input_channels = 1
        output_channels = 1
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        )
        inputs["b"] = np.array([[[[i for i in range(f_size)] for j in range(f_size)]]])

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_4x4_3channels_filter_3x3(self, conv_roll, backend):
        """Test 2D convolution with 4x4 input, 3 channels, and 3x3 filter."""
        # Create args
        args = get_default_args()
        args.n = 64
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_3"

        # Create inputs
        input_channels = 3
        output_channels = 1
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
        inputs["b"] = np.array(
            [
                [
                    [[i + j * f_size + k for i in range(f_size)] for j in range(f_size)]
                    for k in range(input_channels)
                ]
            ]
        )

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_32x32_3channels_filter_3x3(self, conv_roll, backend):
        """Test 2D convolution with 32x32 input, 3 channels, and 3x3 filter (large scale)."""
        # Create args
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_4"

        # Create inputs
        input_channels = 3
        output_channels = 1
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
        inputs["b"] = np.array(
            [
                [
                    [
                        [np.random.randint(0, 10) for i in range(f_size)]
                        for j in range(f_size)
                    ]
                    for k in range(input_channels)
                ]
            ]
        )

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_conv2d_resnet_2ch_in_4ch_out_3x3_filter(self, backend):
        """Test 2D convolution for resnet case: C_in=2, C_out=4, 4x4 spatial, 3x3 filter."""
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_resnet"

        np.random.seed(125)
        input_channels = 2
        output_channels = 4
        dim_size = 4
        f_size = 3
        padding = "same"
        inputs = {}
        # inputs["a"] = np.random.randint(0, 5, (input_channels, dim_size, dim_size)).astype(float)
        # inputs["b"] = np.random.randint(0, 3, (output_channels, input_channels, f_size, f_size)).astype(float)
        inputs["a"] = (
            np.arange(input_channels * dim_size * dim_size)
            .reshape(input_channels, dim_size, dim_size)
            .astype(float)
        )
        inputs["b"] = (
            np.arange(output_channels * input_channels * f_size * f_size)
            .reshape(output_channels, input_channels, f_size, f_size)
            .astype(float)
        )
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_4x4_filter_3x3_random_weights(self, conv_roll, backend):
        """Test 2D convolution with random filter weights (integer values for FHE)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_random_weights"

        # Create inputs with random data (integers for FHE accuracy)
        np.random.seed(42)
        dim_size = 4
        input_channels = 1
        output_channels = 1
        f_size = 3
        padding = "same"
        inputs = {}
        inputs["a"] = np.random.randint(-5, 5, (1, dim_size, dim_size)).astype(float)
        inputs["b"] = np.random.randint(-3, 3, (1, 1, f_size, f_size)).astype(float)

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_4x4_filter_3x3_identity_center(self, conv_roll, backend):
        """Test 2D convolution with identity filter (center weight = 1, rest = 0)."""
        # Create args
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_identity"

        # Create inputs (integer values for FHE)
        np.random.seed(44)
        dim_size = 4
        f_size = 3
        input_channels = 1
        output_channels = 1
        padding = "same"
        inputs = {}
        inputs["a"] = np.random.randint(-10, 10, (1, dim_size, dim_size)).astype(float)
        # Identity filter: only center is 1
        inputs["b"] = np.zeros((1, 1, f_size, f_size))
        inputs["b"][0, 0, 1, 1] = 1.0  # Center position

        # Generate test case
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_conv2d_same_8x8_filter_3x3(self, backend):
        """Test 2D convolution with stride=1 and same padding (8x8 input -> 8x8 output)."""
        args = get_default_args()
        args.n = 64
        args.rolls = True
        args.conv_roll = False
        args.benchmark = "conv2d_same"

        dim_size = 8
        f_size = 3
        stride = 1
        input_channels = 1
        output_channels = 1
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        ).astype(float)
        inputs["b"] = np.array(
            [[[[1 for i in range(f_size)] for j in range(f_size)]]]
        ).astype(float)

        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, stride, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize("conv_roll", [False])
    def test_conv2d_stride2_same_8x8_filter_3x3(self, conv_roll, backend):
        """Test 2D convolution with stride=2 and same padding (8x8 input -> 4x4 output)."""
        args = get_default_args()
        args.n = 64
        args.rolls = True
        args.conv_roll = conv_roll
        args.benchmark = "conv2d_stride2_same"

        dim_size = 8
        f_size = 3
        stride = 2
        input_channels = 1
        output_channels = 1
        padding = "same"
        inputs = {}
        inputs["a"] = np.array(
            [
                [[i + j * dim_size for i in range(dim_size)] for j in range(dim_size)]
                for _ in range(1)
            ]
        ).astype(float)
        inputs["b"] = np.array(
            [[[[1 for i in range(f_size)] for j in range(f_size)]]]
        ).astype(float)

        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, stride, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)
