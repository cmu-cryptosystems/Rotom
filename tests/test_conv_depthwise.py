"""Depthwise convolution on the direct conv kernel.

A depthwise term carries TFLite's ``[1, K_h, K_w, C]`` filter. Its spec keeps
the channel, so the kernel is one masked rotation per tap and sums nothing.
"""

import numpy as np
import pytest
import torch

from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from ir.kernel_cost import KernelCost
from lower.lower import Lower
from tests.test_conv_direct import _compile, _he_ops, _run


def _depthwise(in_shape, k, stride, padding, layout=None, seed=0):
    rng = np.random.default_rng(seed)
    channels = in_shape[0]
    a = TensorTerm.Tensor("a", list(in_shape), True, layout)
    b = TensorTerm.Tensor("b", [1, k, k, channels], False)
    inputs = {
        "a": rng.normal(size=in_shape),
        "b": rng.normal(size=(1, k, k, channels)),
    }
    return TensorTerm.depthwise_conv2d(a, b, stride, padding), inputs


@pytest.mark.parametrize(
    "in_shape,k,stride,padding",
    [((4, 8, 8), 3, 1, "same"), ((4, 8, 8), 5, 2, "same"), ((3, 9, 9), 3, 2, "valid")],
)
def test_eval_matches_torch(in_shape, k, stride, padding):
    term, inputs = _depthwise(in_shape, k, stride, padding)
    channels = in_shape[0]
    # [1, K_h, K_w, C] -> torch's [C, 1, K_h, K_w]
    weight = torch.from_numpy(inputs["b"][0]).permute(2, 0, 1).unsqueeze(1)
    pad = 0 if padding == "valid" else k // 2
    expected = torch.nn.functional.conv2d(
        torch.from_numpy(inputs["a"])[None],
        weight,
        stride=stride,
        padding=pad,
        groups=channels,
    )[0]
    np.testing.assert_allclose(term.eval(inputs), expected.numpy(), atol=1e-12)


@pytest.mark.parametrize(
    "in_shape,k,stride,padding,layout,n",
    [
        ((4, 8, 8), 3, 1, "same", None, 256),
        ((4, 8, 8), 3, 2, "same", None, 256),
        ((4, 8, 8), 5, 1, "same", None, 256),
        ((4, 8, 8), 5, 2, "same", None, 256),
        ((4, 8, 8), 3, 1, "valid", None, 256),
        ((4, 8, 8), 3, 2, "valid", None, 256),
        # channels and spatial extents that are not powers of two
        ((6, 8, 8), 3, 1, "same", None, 512),
        ((3, 6, 6), 5, 1, "same", None, 256),
        ((5, 11, 11), 3, 2, "same", None, 2048),
        # channels across ciphertexts
        ((8, 8, 8), 3, 1, "same", None, 64),
        ((8, 8, 8), 5, 2, "same", None, 128),
        ((4, 8, 8), 3, 2, "same", "[0:4:1];[1:8:1][2:8:1]", 64),
        # split spatial dims, as a stride-2 conv upstream leaves them
        ((4, 8, 8), 3, 1, "same", "[1:4:2][0:4:1][2:8:1][1:2:1]", 256),
        ((4, 8, 8), 3, 2, "same", "[1:2:4][2:8:1][0:4:1][1:4:1]", 256),
    ],
)
def test_depthwise_matches_eval(in_shape, k, stride, padding, layout, n):
    term, inputs = _depthwise(in_shape, k, stride, padding, layout)
    kernel = _run(term, inputs, n)
    assert any(k.op == KernelOp.CONV_DIRECT for k in kernel.post_order())


def test_depthwise_is_one_multiply_per_tap():
    """Against the same convolution written as a full conv with a diagonal
    filter, which multiplies and sums over the input channel too."""
    term, inputs = _depthwise((4, 8, 8), 3, 1, "same")
    kernel = _compile(term, 256)
    (conv,) = [k for k in kernel.post_order() if k.op == KernelOp.CONV_DIRECT]
    ops = KernelCost(conv, "lan").ops()
    assert ops == {"mul": 9, "rot": 8, "add": 8}
    lowered = Lower(conv)
    lowered.lower()
    assert _he_ops({0: lowered.env[conv]}) == ops

    a = TensorTerm.Tensor("a", [4, 8, 8], True)
    b = TensorTerm.Tensor("b", [4, 4, 3, 3], False)
    full = _compile(TensorTerm.conv2d(a, b, 1, "same"), 256)
    (full_conv,) = [k for k in full.post_order() if k.op == KernelOp.CONV_DIRECT]
    full_ops = KernelCost(full_conv, "lan").ops()
    assert full_ops == {"mul": 40, "rot": 16, "add": 40}


def test_inverted_bottleneck_matches_eval():
    """1x1 expand, depthwise 3x3 stride 2, 1x1 project: a MobileNet block."""
    rng = np.random.default_rng(1)
    x = TensorTerm.Tensor("x", [4, 8, 8], True)
    inputs = {
        "x": rng.normal(size=(4, 8, 8)),
        "expand": rng.normal(size=(8, 4, 1, 1)),
        "depthwise": rng.normal(size=(1, 3, 3, 8)),
        "project": rng.normal(size=(4, 8, 1, 1)),
    }
    y = TensorTerm.conv2d(
        x, TensorTerm.Tensor("expand", [8, 4, 1, 1], False), 1, "same"
    )
    y = TensorTerm.depthwise_conv2d(
        y, TensorTerm.Tensor("depthwise", [1, 3, 3, 8], False), 2, "same"
    )
    y = TensorTerm.conv2d(
        y, TensorTerm.Tensor("project", [4, 8, 1, 1], False), 1, "same"
    )
    kernel = _run(y, inputs, 1024)
    assert sum(k.op == KernelOp.CONV_DIRECT for k in kernel.post_order()) == 3


def test_a_channel_multiplier_is_not_supported():
    a = TensorTerm.Tensor("a", [4, 8, 8], True)
    b = TensorTerm.Tensor("b", [1, 3, 3, 8], False)
    with pytest.raises(NotImplementedError, match="channel multiplier"):
        _compile(TensorTerm.depthwise_conv2d(a, b, 1, "same"), 256)
