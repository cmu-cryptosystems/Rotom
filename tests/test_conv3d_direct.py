"""3D convolution on the direct conv kernel.

Covers what a (2+1)D network asks of it: a stride per axis, padding given per
axis (TensorFlow's ``SAME``, causal in time) and depthwise filters.
"""

import numpy as np
import pytest
import torch

from frontends.tensor import TensorTerm
from frontends.tensor_args import tf_same_padding, window_axis
from ir.kernel import KernelOp
from ir.kernel_cost import KernelCost
from lower.lower import Lower
from tests.test_conv_direct import _compile, _he_ops, _run

CAUSAL_3 = ((2, 0), (0, 0), (0, 0))


def _same(size, k, stride):
    """TensorFlow SAME on H and W, nothing in time."""
    return ((0, 0), tf_same_padding(size, k, stride), tf_same_padding(size, k, stride))


def _terms(in_shape, filter_shape, stride, padding, depthwise, layout=None, seed=0):
    rng = np.random.default_rng(seed)
    a = TensorTerm.Tensor("a", list(in_shape), True, layout)
    b = TensorTerm.Tensor("b", list(filter_shape), False)
    inputs = {"a": rng.normal(size=in_shape), "b": rng.normal(size=filter_shape)}
    make = TensorTerm.depthwise_conv3d if depthwise else TensorTerm.conv3d
    return make(a, b, stride, padding), inputs


def _torch_conv3d(inputs, stride, pads, depthwise):
    """The same convolution in PyTorch, padded by hand."""
    x = torch.from_numpy(inputs["a"])[None]
    weight = torch.from_numpy(inputs["b"])
    groups = 1
    if depthwise:
        # [1, K_d, K_h, K_w, C] -> [C, 1, K_d, K_h, K_w]
        weight = weight[0].permute(3, 0, 1, 2).unsqueeze(1)
        groups = x.shape[1]
    # F.pad takes the last axis first
    flat = [p for before_after in reversed(pads) for p in before_after]
    x = torch.nn.functional.pad(x, flat)
    return torch.nn.functional.conv3d(x, weight, stride=stride, groups=groups)[0]


@pytest.mark.parametrize(
    "extent,kernel,stride,pads,out",
    [
        (172, 3, 2, (0, 1), 86),
        (86, 5, 2, (1, 2), 43),
        (43, 3, 2, (1, 1), 22),
        (8, 3, 1, (1, 1), 8),
        (8, 1, 2, (0, 0), 4),
    ],
)
def test_tf_same_padding(extent, kernel, stride, pads, out):
    assert tf_same_padding(extent, kernel, stride) == pads
    assert window_axis(extent, kernel, stride, pads) == (out, *pads)


CASES = [
    # in_shape, filter_shape, stride, padding, depthwise
    ((3, 4, 8, 8), (8, 3, 1, 3, 3), (1, 2, 2), _same(8, 3, 2), False),  # a stem
    ((4, 4, 8, 8), (8, 4, 1, 1, 1), 1, "same", False),  # an expansion
    ((6, 4, 8, 8), (1, 1, 3, 3, 6), 1, "same", True),
    ((6, 4, 8, 8), (1, 1, 5, 5, 6), (1, 2, 2), _same(8, 5, 2), True),
    ((6, 4, 8, 8), (1, 3, 1, 1, 6), 1, CAUSAL_3, True),
    ((6, 8, 4, 4), (1, 5, 1, 1, 6), 1, ((4, 0), (0, 0), (0, 0)), True),
    ((2, 4, 4, 4), (4, 2, 3, 3, 3), 1, "same", False),
    ((2, 5, 6, 6), (4, 2, 3, 3, 3), (1, 2, 2), ((2, 0), (0, 1), (0, 1)), False),
]


@pytest.mark.parametrize("case", range(len(CASES)))
def test_eval_matches_torch(case):
    in_shape, filter_shape, stride, padding, depthwise = CASES[case]
    term, inputs = _terms(in_shape, filter_shape, stride, padding, depthwise)
    pads = [
        (before, after)
        for _, before, after in term_args(term).window(in_shape, filter_shape)
    ]
    expected = _torch_conv3d(inputs, stride, pads, depthwise)
    np.testing.assert_allclose(term.eval(inputs), expected.numpy(), atol=1e-12)


def term_args(term):
    from frontends.tensor_args import Conv3dArgs

    return Conv3dArgs.from_term(term)


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("n", [256, 2048])
def test_conv3d_matches_eval(case, n):
    in_shape, filter_shape, stride, padding, depthwise = CASES[case]
    term, inputs = _terms(in_shape, filter_shape, stride, padding, depthwise)
    kernel = _run(term, inputs, n)
    assert any(k.op == KernelOp.CONV_DIRECT for k in kernel.post_order())


def test_a_temporal_conv_over_frames_on_ciphertexts_does_not_rotate():
    """One frame per ciphertext: a causal conv in time adds weighted frames."""
    term, inputs = _terms(
        (4, 4, 8, 8),
        (1, 3, 1, 1, 4),
        1,
        CAUSAL_3,
        True,
        layout="[1:4:1];[0:4:1][2:8:1][3:8:1]",
    )
    kernel = _run(term, inputs, 256)
    (conv,) = [k for k in kernel.post_order() if k.op == KernelOp.CONV_DIRECT]
    ops = KernelCost(conv, "lan").ops()
    # frame t reads frames t-2, t-1 and t: 1 + 2 + 3 + 3 products
    assert ops == {"mul": 9, "rot": 0, "add": 5}
    lowered = Lower(conv)
    lowered.lower()
    assert _he_ops({0: lowered.env[conv]}) == ops


def test_two_plus_one_d_block_matches_eval():
    """1x1x1 expand, depthwise 1x3x3 stride 2, depthwise 3x1x1 causal, 1x1x1
    project: the convolutions of a MoViNet block."""
    rng = np.random.default_rng(1)
    x = TensorTerm.Tensor("x", [4, 4, 8, 8], True)
    shapes = {
        "expand": (8, 4, 1, 1, 1),
        "spatial": (1, 1, 3, 3, 8),
        "temporal": (1, 3, 1, 1, 8),
        "project": (4, 8, 1, 1, 1),
    }
    inputs = {name: rng.normal(size=shape) for name, shape in shapes.items()}
    inputs["x"] = rng.normal(size=(4, 4, 8, 8))
    w = {
        name: TensorTerm.Tensor(name, list(shape), False)
        for name, shape in shapes.items()
    }
    y = TensorTerm.conv3d(x, w["expand"], 1, "same")
    y = TensorTerm.depthwise_conv3d(y, w["spatial"], (1, 2, 2), _same(8, 3, 2))
    y = TensorTerm.depthwise_conv3d(y, w["temporal"], 1, CAUSAL_3)
    y = TensorTerm.conv3d(y, w["project"], 1, "same")
    kernel = _run(y, inputs, 2048)
    assert sum(k.op == KernelOp.CONV_DIRECT for k in kernel.post_order()) == 4


def test_a_conv3d_costs_what_its_circuit_does():
    term, _ = _terms((2, 4, 4, 4), (4, 2, 3, 3, 3), 1, "same", False)
    kernel = _compile(term, 512)
    (conv,) = [k for k in kernel.post_order() if k.op == KernelOp.CONV_DIRECT]
    lowered = Lower(conv)
    lowered.lower()
    assert _he_ops({0: lowered.env[conv]}) == KernelCost(conv, "lan").ops()
