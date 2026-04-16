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
from util.layout_util import apply_layout


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
        # Generate expected result
        expected = tensor_ir.eval(inputs)

        # Run compiler + backend
        results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, backend)

        # Check result
        expected_cts = apply_layout(expected, kernel.layout)
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


def test_conv2d_stride2_gap_split_channel_layout_regression():
    """Minimal regression for split-channel + gap-slot input layout."""
    args = get_default_args()
    args.n = 512
    args.rolls = True
    args.conv_roll = False
    args.backend = "toy"
    args.benchmark = "conv2d_gap_split_channel_regression"

    a = TensorTerm.Tensor(
        "a",
        [4, 8, 8],
        True,
        layout="[0:2:2][1:8:1][2:8:1][G:2][0:2:1]",
    )
    b = TensorTerm.Tensor("b", [8, 4, 3, 3], False)
    y = TensorTerm.conv2d(a, b, 2, "same")

    rng = np.random.default_rng(0)
    inputs = {
        "a": rng.normal(size=(4, 8, 8)),
        "b": rng.normal(size=(8, 4, 3, 3)),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, "toy")
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")


@pytest.mark.parametrize(
    "input_layout",
    [
        None,  # default (assignment chooses)
        "[0:1:1][1:4:1][2:4:1]",  # explicit dense
        "[G:2][0:1:1][1:4:1][2:4:1]",  # small leading gap
        "[G:8][0:1:1][1:4:1][2:4:1]",  # medium leading gap
        "[G:32][0:1:1][1:4:1][2:4:1]",  # large leading gap
        "[0:1:1][G:4][1:4:1][2:4:1]",  # gap between channel and H
        "[0:1:1][2:4:1][1:4:1]",  # swap H/W physical order
    ],
)
def test_conv2d_same_varied_input_layouts(backend, input_layout):
    """Conv2D(same) stays correct across a few input packing layouts."""
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.conv_roll = False
    args.benchmark = f"conv2d_same_layout_{abs(hash(input_layout)) % 10**8}"

    h = w = 4
    c_in, c_out = 1, 1
    k = 3
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=input_layout)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(123 if input_layout is None else 456)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": np.random.default_rng(999)
        .normal(size=(c_out, c_in, k, k))
        .astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


@pytest.mark.parametrize(
    "input_layout",
    [
        # C_in=4 as 2×2; H=W=4 ⇒ spatial stride 16; c = c0*2+c1 ⇒ offset c0*32+c1*16
        "[0:2:32][1:4:1][2:4:1][0:2:16]",
        "[0:2:32][1:4:1][2:4:1][G:2][0:2:16]",
        "[G:8][0:2:32][1:4:1][2:4:1][G:2][0:2:16]",
    ],
)
def test_conv2d_same_split_channel_layouts(backend, input_layout):
    """Conv2D(same) with input channel split across two dim-0 factors (+ optional gaps)."""
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.conv_roll = False
    args.benchmark = f"conv2d_same_splitc_{abs(hash(input_layout)) % 10**8}"

    h = w = 4
    # c_out == c_in: with split dim-0 channel packing, shape_check can otherwise disagree
    # between logical Cout and materialized output layout extents.
    c_in, c_out = 4, 4
    k = 3
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=input_layout)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(202)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": np.random.default_rng(303)
        .normal(size=(c_out, c_in, k, k))
        .astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


@pytest.mark.parametrize("c_in,c_out", [(2, 3), (3, 1), (1, 4), (4, 2), (3, 3)])
def test_conv2d_same_multichannel_io_default_layout(backend, c_in, c_out):
    """Conv2D(same): multiple input and output channels (assignment picks input layout)."""
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.conv_roll = False
    args.benchmark = f"conv2d_mc_io_{c_in}_{c_out}"

    h = w = 4
    k = 3
    a = TensorTerm.Tensor("a", [c_in, h, w], True)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(700 + c_in * 31 + c_out)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": np.random.default_rng(800 + c_out)
        .normal(size=(c_out, c_in, k, k))
        .astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


@pytest.mark.parametrize("c_in,c_out", [(2, 3), (4, 1), (1, 3)])
def test_conv2d_same_multichannel_io_dense_channel_layout(backend, c_in, c_out):
    """Conv2D(same): multi I/O channels with explicit dense channel dim in layout string."""
    args = get_default_args()
    args.n = 4096
    args.rolls = True
    args.conv_roll = False
    args.benchmark = f"conv2d_mc_dense_{c_in}_{c_out}"

    h = w = 4
    k = 3
    layout = f"[0:{c_in}:1][1:4:1][2:4:1]"
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=layout)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(900 + c_in + c_out * 17)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": np.random.default_rng(1000 + c_out)
        .normal(size=(c_out, c_in, k, k))
        .astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


def test_conv2d_depthwise_same_diagonal_channels(backend):
    """Depth-wise style conv: each output channel uses only its matching input channel."""
    args = get_default_args()
    args.n = 256
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_depthwise_same_diag"

    c_in = c_out = 4
    h = w = 8
    k = 3
    a = TensorTerm.Tensor("a", [c_in, h, w], True)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(1234)
    weights = np.zeros((c_out, c_in, k, k), dtype=np.float64)
    for c in range(c_in):
        weights[c, c, :, :] = rng.normal(size=(k, k))

    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": weights,
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


def test_conv2d_depthwise_channel_multiplier_two(backend):
    """Depth-wise style conv with multiplier=2 (2 output maps per input channel)."""
    args = get_default_args()
    args.n = 256
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_depthwise_multiplier2"

    c_in = 3
    channel_multiplier = 2
    c_out = c_in * channel_multiplier
    h = w = 8
    k = 3
    a = TensorTerm.Tensor("a", [c_in, h, w], True)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(5678)
    weights = np.zeros((c_out, c_in, k, k), dtype=np.float64)
    for c in range(c_in):
        for m in range(channel_multiplier):
            out_c = c * channel_multiplier + m
            weights[out_c, c, :, :] = rng.normal(size=(k, k))

    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": weights,
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


@pytest.mark.parametrize("stride,padding", [(2, "same"), (1, "same")])
def test_conv2d_depthwise_multiplier_two_varied_stride(backend, stride, padding):
    """Depth-wise style conv (multiplier=2) across nontrivial stride settings."""
    args = get_default_args()
    args.n = 512
    args.rolls = True
    args.conv_roll = False
    args.benchmark = f"conv2d_depthwise_m2_s{stride}"

    c_in = 4
    channel_multiplier = 2
    c_out = c_in * channel_multiplier
    h = w = 8
    k = 3
    a = TensorTerm.Tensor("a", [c_in, h, w], True)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, stride, padding)

    rng = np.random.default_rng(9100 + stride)
    weights = np.zeros((c_out, c_in, k, k), dtype=np.float64)
    for c in range(c_in):
        for m in range(channel_multiplier):
            out_c = c * channel_multiplier + m
            weights[out_c, c, :, :] = rng.normal(size=(k, k))

    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": weights,
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


def test_conv2d_depthwise_same_large_kernel(backend):
    """Depth-wise style conv with larger 5x5 kernels under same padding."""
    args = get_default_args()
    args.n = 256
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_depthwise_same_k5"

    c_in = c_out = 4
    h = w = 8
    k = 5
    a = TensorTerm.Tensor("a", [c_in, h, w], True)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(9200)
    weights = np.zeros((c_out, c_in, k, k), dtype=np.float64)
    for c in range(c_in):
        weights[c, c, :, :] = rng.normal(size=(k, k))

    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": weights,
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


def test_conv2d_depthwise_split_channel_gap_layout_multiplier_two(backend):
    """Depth-wise style conv from split-channel + gap layout with multiplier=2."""
    args = get_default_args()
    args.n = 512
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_depthwise_split_gap_m2"

    c_in = 4
    channel_multiplier = 2
    c_out = c_in * channel_multiplier
    h = w = 8
    k = 3
    input_layout = "[0:2:32][1:8:1][2:8:1][G:2][0:2:16]"
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=input_layout)
    b = TensorTerm.Tensor("b", [c_out, c_in, k, k], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(9300)
    weights = np.zeros((c_out, c_in, k, k), dtype=np.float64)
    for c in range(c_in):
        for m in range(channel_multiplier):
            out_c = c * channel_multiplier + m
            weights[out_c, c, :, :] = rng.normal(size=(k, k))

    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": weights,
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)
