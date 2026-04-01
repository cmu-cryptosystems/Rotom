"""
Test suite for 3D convolution: layout assignment, lowering, and backend correctness.

Shapes follow ``TensorTerm.conv3d`` / ``TensorEvaluator``:
  - input:  [C_in, D, H, W]
  - filter: [C_out, C_in, K_d, K_h, K_w]
  - output: [C_out, D_out, H_out, W_out]

Full-stack tests primarily use ``padding="same"`` (aligned with ``tests/test_conv2d.py``).
``padding="valid"`` is covered both by an independent NumPy reference vs ``TensorTerm.eval``
and an end-to-end backend check.
"""

import numpy as np
import pytest

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def _naive_conv3d_valid(inp, flt, stride: int) -> np.ndarray:
    """Reference conv3d with valid padding (loops ordered differently from ``TensorEvaluator``)."""
    c_in, d_in, h_in, w_in = inp.shape
    c_out, cif, k_d, k_h, k_w = flt.shape
    assert cif == c_in
    d_o = (d_in - k_d) // stride + 1
    h_o = (h_in - k_h) // stride + 1
    w_o = (w_in - k_w) // stride + 1
    out = np.zeros((c_out, d_o, h_o, w_o), dtype=np.float64)
    for oc in range(c_out):
        for ic in range(c_in):
            for od in range(d_o):
                ds = od * stride
                for oh in range(h_o):
                    hs = oh * stride
                    for ow in range(w_o):
                        ws = ow * stride
                        patch = inp[ic, ds : ds + k_d, hs : hs + k_h, ws : ws + k_w]
                        out[oc, od, oh, ow] += float(np.sum(patch * flt[oc, ic]))
    return out


class TestConvolution3D:
    """3D convolution: eval vs compiled circuit (Toy / CKKS)."""

    def test_conv3d_valid_eval_matches_naive_reference(self):
        """``TensorTerm.eval`` for valid padding matches an independent NumPy implementation."""
        d = h = w = 4
        c_in, c_out = 1, 1
        k = 3
        rng_a = np.random.default_rng(11)
        rng_b = np.random.default_rng(22)
        inputs = {
            "a": rng_a.normal(size=(c_in, d, h, w)).astype(np.float64),
            "b": rng_b.normal(size=(c_out, c_in, k, k, k)).astype(np.float64),
        }
        a = TensorTerm.Tensor("a", [c_in, d, h, w], True)
        b = TensorTerm.Tensor("b", [c_out, c_in, k, k, k], False)
        tensor_ir = TensorTerm.conv3d(a, b, 1, "valid")
        got = tensor_ir.eval(inputs)
        ref = _naive_conv3d_valid(inputs["a"], inputs["b"], 1)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12)

    def test_conv3d_valid_end_to_end(self, backend):
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.conv_roll = False
        args.benchmark = "conv3d_valid_e2e"

        d = h = w = 4
        c_in, c_out = 1, 1
        k = 3
        inputs = {
            "a": np.random.default_rng(1)
            .normal(size=(c_in, d, h, w))
            .astype(np.float64),
            "b": np.random.default_rng(2)
            .normal(size=(c_out, c_in, k, k, k))
            .astype(np.float64),
        }
        tensor_ir = self._create_conv3d(d, h, w, c_in, c_out, k, k, k, 1, "valid")
        self._run_test_case(tensor_ir, inputs, args, backend)

    @pytest.mark.parametrize(
        "input_layout",
        [
            None,  # default (assignment chooses)
            "[0:1:1][1:4:1][2:4:1][3:4:1]",  # explicit dense
            "[G:8][0:1:1][1:4:1][2:4:1][3:4:1]",  # extra leading gap
            "[0:1:1][2:4:1][1:4:1][3:4:1]",  # swap D/H physical order
        ],
    )
    def test_conv3d_same_varied_input_layouts(self, backend, input_layout):
        """Conv3D(same) stays correct across a few input packing layouts."""
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.conv_roll = False
        args.benchmark = f"conv3d_same_layout_{abs(hash(input_layout)) % 10**8}"

        d = h = w = 4
        c_in, c_out = 1, 1
        k = 3
        a = TensorTerm.Tensor("a", [c_in, d, h, w], True, layout=input_layout)
        b = TensorTerm.Tensor("b", [c_out, c_in, k, k, k], False)
        y = TensorTerm.conv3d(a, b, 1, "same")

        rng = np.random.default_rng(123 if input_layout is None else 456)
        inputs = {
            "a": rng.normal(size=(c_in, d, h, w)).astype(np.float64),
            "b": np.random.default_rng(999)
            .normal(size=(c_out, c_in, k, k, k))
            .astype(np.float64),
        }
        self._run_test_case(y, inputs, args, backend)

    def _create_conv3d(
        self, d_in, h_in, w_in, c_in, c_out, k_d, k_h, k_w, stride, padding
    ):
        a = TensorTerm.Tensor("a", [c_in, d_in, h_in, w_in], True)
        b = TensorTerm.Tensor("b", [c_out, c_in, k_d, k_h, k_w], False)
        return TensorTerm.conv3d(a, b, stride, padding)

    def _run_test_case(self, tensor_ir, inputs, args, backend):
        expected = tensor_ir.eval(inputs)
        results, kernel = run_compiler_and_backend(tensor_ir, inputs, args, backend)
        expected_cts = apply_layout(expected, kernel.layout)
        assert_results_equal(expected_cts, results, backend)

    def test_conv3d_4x4x4_kernel_3_same(self, backend):
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.conv_roll = False
        args.benchmark = "conv3d_1"

        d = h = w = 4
        c_in, c_out = 1, 1
        k = 3
        inputs = {
            "a": np.arange(c_in * d * h * w, dtype=np.float64).reshape(c_in, d, h, w),
            "b": np.ones((c_out, c_in, k, k, k), dtype=np.float64),
        }
        tensor_ir = self._create_conv3d(d, h, w, c_in, c_out, k, k, k, 1, "same")
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_conv3d_multichannel_same(self, backend):
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.conv_roll = False
        args.benchmark = "conv3d_2"

        d = h = w = 4
        c_in, c_out = 2, 3
        k = 3
        np.random.seed(0)
        inputs = {
            "a": np.random.randn(c_in, d, h, w).astype(np.float64),
            "b": np.random.randn(c_out, c_in, k, k, k).astype(np.float64),
        }
        tensor_ir = self._create_conv3d(d, h, w, c_in, c_out, k, k, k, 1, "same")
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_conv3d_stride2_same(self, backend):
        args = get_default_args()
        args.n = 8192
        args.rolls = True
        args.conv_roll = False
        args.benchmark = "conv3d_stride2"

        d = h = w = 8
        c_in, c_out = 1, 1
        k = 3
        inputs = {
            "a": np.array(
                [
                    [
                        [[i + j * w + z * w * h for i in range(w)] for j in range(h)]
                        for z in range(d)
                    ]
                ],
                dtype=np.float64,
            ),
            "b": np.ones((c_out, c_in, k, k, k), dtype=np.float64),
        }
        tensor_ir = self._create_conv3d(d, h, w, c_in, c_out, k, k, k, 2, "same")
        self._run_test_case(tensor_ir, inputs, args, backend)

    def test_conv3d_identity_center(self, backend):
        args = get_default_args()
        args.n = 4096
        args.rolls = True
        args.conv_roll = False
        args.benchmark = "conv3d_identity"

        d = h = w = 4
        c_in, c_out = 1, 1
        k = 3
        wts = np.zeros((c_out, c_in, k, k, k), dtype=np.float64)
        wts[0, 0, k // 2, k // 2, k // 2] = 1.0
        inputs = {
            "a": np.random.default_rng(3)
            .integers(-5, 5, (c_in, d, h, w))
            .astype(np.float64),
            "b": wts,
        }
        tensor_ir = self._create_conv3d(d, h, w, c_in, c_out, k, k, k, 1, "same")
        self._run_test_case(tensor_ir, inputs, args, backend)
