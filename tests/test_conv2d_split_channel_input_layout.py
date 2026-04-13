"""Conv2d with logical channel dim ``0`` split across multiple slot axes.

A ciphertext-packed layout such as ``[0:64:1];[G:32][1:8:1][G:4][2:8:1][G:4]`` (Cout on
the CT axis) can be compacted into a single-CT slot packing where dim ``0`` is
factored across two in-slot dimensions, e.g.:

``[0:32:4][1:8:1][0:4:1][2:8:1][G:4]``  (``n = 32768``)

These tests assert that **starting** a ``conv2d`` from such packings (and a smaller
analog) still matches dense ``tensor_ir.eval`` on the Toy backend (and CKKS where
cheap). This isolates split-traversal channel layout handling from full ResNet
assignment, where other kernels (e.g. CT-packed Cout) still miscompare Toy vs eval.
"""

from __future__ import annotations

import numpy as np
import pytest

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout

# Compact single-CT analogue of ``[0:64:1];[G:32][1:8:1][G:4][2:8:1][G:4]`` for [64,8,8].
LAYOUT_64CH_8X8_COMPACT = "[0:32:4][1:8:1][0:4:1][2:8:1][G:4]"
N_RESNET_SLOT = 32768

# Small analogue: 4 channels as [0:2:2] * [0:2:1], 4×4 spatial, ``n = 64``.
LAYOUT_4CH_4X4_SPLIT0 = "[0:2:2][1:4:1][0:2:1][2:4:1]"
N_SMALL = 64


@pytest.mark.parametrize("backend", ["toy", "ckks"])
def test_conv2d_same_split_dim0_channel_small_4x4(backend: str) -> None:
    """``[4,4,4]`` CHW with dim ``0`` split; ``same`` stride-1; filter ``[1,4,3,3]``."""
    args = get_default_args()
    args.n = N_SMALL
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_split_dim0_4ch_4x4"

    c_in, h, w = 4, 4, 4
    c_out = 1
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=LAYOUT_4CH_4X4_SPLIT0)
    b = TensorTerm.Tensor("b", [c_out, c_in, 3, 3], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(2026)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": rng.normal(size=(c_out, c_in, 3, 3)).astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


@pytest.mark.parametrize("backend", ["toy", "ckks"])
def test_conv2d_same_split_dim0_channel_stride2_8x8(backend: str) -> None:
    """Stride-2 ``same`` with split dim ``0``: ``[4,8,8]`` → ``[8,4,4]``; filter ``[8,4,3,3]``."""
    args = get_default_args()
    args.n = 256
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_split_dim0_stride2_8x8"

    layout = "[0:2:2][1:8:1][0:2:1][2:8:1]"
    c_in, h, w = 4, 8, 8
    c_out = 8
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=layout)
    b = TensorTerm.Tensor("b", [c_out, c_in, 3, 3], False)
    y = TensorTerm.conv2d(a, b, 2, "same")

    rng = np.random.default_rng(2027)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": rng.normal(size=(c_out, c_in, 3, 3)).astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)


@pytest.mark.slow
@pytest.mark.parametrize("backend", ["toy", "ckks"])
def test_conv2d_same_compact_64ch_8x8_input_matches_eval(backend: str) -> None:
    """ResNet-scale slot packing: ``[64,8,8]`` activations, compact dim-0 split; ``[8,64,3,3]`` filter."""
    args = get_default_args()
    args.n = N_RESNET_SLOT
    args.rolls = True
    args.conv_roll = False
    args.benchmark = "conv2d_compact_64ch_8x8_in"

    c_in, h, w = 64, 8, 8
    c_out = 8
    a = TensorTerm.Tensor("a", [c_in, h, w], True, layout=LAYOUT_64CH_8X8_COMPACT)
    b = TensorTerm.Tensor("b", [c_out, c_in, 3, 3], False)
    y = TensorTerm.conv2d(a, b, 1, "same")

    rng = np.random.default_rng(2028)
    inputs = {
        "a": rng.normal(size=(c_in, h, w)).astype(np.float64),
        "b": rng.normal(size=(c_out, c_in, 3, 3)).astype(np.float64),
    }

    expected = y.eval(inputs)
    results, kernel = run_compiler_and_backend(y, inputs, args, backend)
    expected_cts = apply_layout(expected, kernel.layout)
    assert_results_equal(expected_cts, results, backend)
