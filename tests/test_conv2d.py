"""
Test suite for 2D convolution operations.

This module tests 2D convolution operations in the Rotom homomorphic encryption system,
including various filter sizes and input dimensions.
"""

import math

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.dim import *
from ir.he import HEOp
from ir.kernel import KernelOp
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def _conv2d_same_padding_and_partial_products(input_tensor, filter_tensor, stride):
    """Compute same padding and reference partial products for conv2d.

    Partial product at (c_out, c_in, f_h, f_w, h, w) is
    input_padded[c_in, h' , w'] * filter[c_out, c_in, f_h, f_w]
    where (h', w') = (h + pad_top - f_h, w + pad_left - f_w).

    Returns:
        pad_top, pad_left: int
        ref_partial: np.ndarray shape (C_out, C_in, f_h, f_w, H_out, W_out)
    """
    input_shape = input_tensor.shape
    filter_shape = filter_tensor.shape
    # same padding output size
    if stride == 1:
        output_shape = (filter_shape[0], input_shape[1], input_shape[2])
    else:
        h_o = (input_shape[1] + stride - 1) // stride
        w_o = (input_shape[2] + stride - 1) // stride
        output_shape = (filter_shape[0], h_o, w_o)

    pad_top = max(
        0,
        math.floor(
            (stride * (output_shape[1] - 1) - input_shape[1] + filter_shape[2])
            / 2
        ),
    )
    pad_bot = max(
        0,
        math.ceil(
            (stride * (output_shape[1] - 1) - input_shape[1] + filter_shape[2])
            / 2
        ),
    )
    pad_left = max(
        0,
        math.floor(
            (stride * (output_shape[2] - 1) - input_shape[2] + filter_shape[3])
            / 2
        ),
    )
    pad_right = max(
        0,
        math.ceil(
            (stride * (output_shape[2] - 1) - input_shape[2] + filter_shape[3])
            / 2
        ),
    )

    padded = np.pad(
        input_tensor,
        pad_width=((0, 0), (pad_top, pad_bot), (pad_left, pad_right)),
        mode="constant",
        constant_values=0,
    )
    # (C_out, C_in, f_h, f_w, H_out, W_out)
    c_out, c_in, f_h, f_w = filter_shape[0], filter_shape[1], filter_shape[2], filter_shape[3]
    h_out, w_out = output_shape[1], output_shape[2]
    # Partial product: for output (i,j), patch is input at (i*stride + fh, j*stride + fw)
    # (same as frontend: patch = input[i_start:i_end, j_start:j_end] with i_start = i*stride, etc.)
    ref_partial = np.zeros((c_out, c_in, f_h, f_w, h_out, w_out))
    for out_c in range(c_out):
        for in_c in range(c_in):
            for fh in range(f_h):
                for fw in range(f_w):
                    for i in range(h_out):
                        for j in range(w_out):
                            hi = i * stride + fh
                            wj = j * stride + fw
                            if 0 <= hi < padded.shape[1] and 0 <= wj < padded.shape[2]:
                                ref_partial[out_c, in_c, fh, fw, i, j] = (
                                    padded[in_c, hi, wj] * filter_tensor[out_c, in_c, fh, fw]
                                )
    return pad_top, pad_left, ref_partial


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

        # Run compiler
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()
        results = run_backend(backend, circuit_ir, inputs, args)

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
        inputs["b"] = np.array([[[[i + j * f_size + k for i in range(f_size)] for j in range(f_size)] for k in range(input_channels)]])

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
        inputs["b"] = np.array([[[[np.random.randint(0, 10) for i in range(f_size)] for j in range(f_size)] for k in range(input_channels)]])

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
        inputs["a"] = np.arange(input_channels * dim_size * dim_size).reshape(
            input_channels, dim_size, dim_size
        ).astype(float)
        inputs["b"] = np.arange(output_channels * input_channels * f_size * f_size).reshape(
            output_channels, input_channels, f_size, f_size
        ).astype(float)
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, 1, padding
        )
        self._run_test_case(tensor_ir, inputs, args, backend)

    #     def test_conv2d_resnet_partial_products_match_reference(self):
        """Verify that MUL partial products in lowered conv2d match reference conv2d partial products.

        Reference partial products: for each (c_out, c_in, f_h, f_w, h, w),
        ref = input_padded[c_in, h*stride+f_h, w*stride+f_w] * filter[c_out, c_in, f_h, f_w]
        (same formula as frontend eval_conv2d). We check:
        1) The sum of all 18 partial products per output CT equals the reference conv2d output.
        2) Each of the 18 MUL slot vectors in the circuit equals exactly one reference partial
           (same set of 18; order may differ due to reduction order in lowering).
        """
        args = get_default_args()
        args.n = 16
        args.rolls = True
        args.benchmark = "conv2d_resnet"
        input_channels = 2
        output_channels = 4
        dim_size = 4
        f_size = 3
        padding = "same"
        stride = 1
        inputs = {}
        inputs["a"] = np.arange(input_channels * dim_size * dim_size).reshape(
            input_channels, dim_size, dim_size
        ).astype(float)
        inputs["b"] = np.arange(output_channels * input_channels * f_size * f_size).reshape(
            output_channels, input_channels, f_size, f_size
        ).astype(float)

        # Reference: partial products and full conv2d output
        _, _, ref_partial = _conv2d_same_padding_and_partial_products(
            inputs["a"], inputs["b"], stride
        )
        # ref_partial[c_out, c_in, f_h, f_w, h, w]; sum over c_in, f_h, f_w -> ref_out[c_out, h, w]
        ref_out = np.sum(ref_partial, axis=(1, 2, 3))  # (C_out, H_out, W_out)
        h_out, w_out = ref_partial.shape[4], ref_partial.shape[5]

        # Build and lower conv2d
        tensor_ir = self._create_convolution_computation(
            dim_size, input_channels, output_channels, f_size, f_size, stride, padding
        )
        kernel = LayoutAssignment(tensor_ir, args).run()
        circuit_ir = Lower(kernel).run()

        # Find CONV2D term and collect MUL slot values per output CT
        conv2d_term = None
        for term in circuit_ir:
            if term.op == KernelOp.CONV2D:
                conv2d_term = term
                break
        assert conv2d_term is not None

        layout_cts = circuit_ir[conv2d_term]
        cts_dict = layout_cts.cts if hasattr(layout_cts, "cts") else layout_cts
        backend = Toy(circuit_ir, inputs, args)
        mul_results_by_ct = []
        for ct_idx in sorted(cts_dict.keys()):
            ct = cts_dict[ct_idx]
            for ct_term in ct.post_order():
                backend.env[ct_term] = backend.eval(ct_term)
            muls = [backend.env[t] for t in ct.post_order() if t.op == HEOp.MUL]
            mul_results_by_ct.append(muls)

        c_in_count = input_channels
        num_partials = c_in_count * f_size * f_size  # 18
        for c_out in range(output_channels):
            muls = mul_results_by_ct[c_out]
            assert len(muls) == num_partials, (
                f"expected {num_partials} MULs per output CT, got {len(muls)}"
            )
            # (1) Sum of partial products must equal reference conv2d output
            sum_slots = np.zeros(h_out * w_out)
            for slot_vec in muls:
                sum_slots += np.array(slot_vec)
            circuit_out_2d = sum_slots.reshape(h_out, w_out)
            np.testing.assert_allclose(
                circuit_out_2d,
                ref_out[c_out, :, :],
                err_msg=(
                    f"sum of partial products != reference conv2d output for c_out={c_out}. "
                    "This checks that the 18 MUL partial products sum to the reference conv2d."
                ),
            )

            # (2) Each MUL must equal exactly one reference partial (same set of 18)
            ref_flat = ref_partial[c_out, :, :, :, :, :].reshape(
                num_partials, h_out * w_out
            )
            used = [False] * num_partials
            for slot_vec in muls:
                v = np.array(slot_vec)
                found = False
                for k in range(num_partials):
                    if not used[k] and np.allclose(v, ref_flat[k]):
                        used[k] = True
                        found = True
                        break
                assert found, (
                    f"c_out={c_out}: one MUL result did not match any reference partial product. "
                    "Each multiplication in the lowered circuit should equal one (c_in, f_h, f_w) partial."
                )
            assert all(used), f"c_out={c_out}: not all reference partials matched"

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

