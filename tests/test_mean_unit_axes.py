"""Mean over a tensor that has axes of extent 1.

A layout has no piece for such an axis, so the layout's dims are the tensor's
other axes, and the mean over an axis of extent 1 is the tensor itself.
"""

import numpy as np
import pytest

from assignment.gen.gen_mean import _layout_dims_of_axes
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from tests.test_conv_direct import _run


@pytest.mark.parametrize(
    "shape,axes,ops",
    [
        ((4, 4, 4), (1, 2), ["TENSOR", "MEAN"]),
        # frames of extent 1: H and W are the layout's second and third dims
        ((4, 1, 4, 4), (2, 3), ["TENSOR", "MEAN"]),
        ((4, 2, 4, 1), (2, 3), ["TENSOR", "MEAN"]),
        ((8, 2, 1, 4), (1, 3), ["TENSOR", "MEAN"]),
        # one of the axes has extent 1: only the other is reduced
        ((4, 1, 4, 4), (1, 2), ["TENSOR", "MEAN"]),
        # every axis has extent 1: no homomorphic operation
        ((4, 2, 1, 1), (2, 3), ["TENSOR", "CAST"]),
        ((4, 1, 4, 4), (1,), ["TENSOR", "CAST"]),
    ],
)
def test_mean_matches_eval(shape, axes, ops):
    rng = np.random.default_rng(0)
    x = TensorTerm.Tensor("x", list(shape), True)
    kernel = _run(x.mean(axes), {"x": rng.normal(size=shape)}, 256)
    assert [k.op.name for k in kernel.post_order()] == ops


def test_a_mean_over_axes_of_extent_one_costs_nothing():
    rng = np.random.default_rng(1)
    x = TensorTerm.Tensor("x", [4, 2, 1, 1], True)
    kernel = _run(x.mean((2, 3)), {"x": rng.normal(size=(4, 2, 1, 1))}, 256)
    assert kernel.op == KernelOp.CAST
    assert kernel.layout.get_dims() == kernel.cs[0].layout.get_dims()


@pytest.mark.parametrize(
    "axes,shape,layout_dims,expected",
    [
        ((1, 2), [4, 4, 4], [0, 1, 2], [1, 2]),
        ((2, 3), [4, 1, 4, 4], [0, 2, 3], [2, 3]),
        ((1, 2), [4, 1, 4, 4], [0, 2, 3], [2]),
        ((2, 3), [4, 2, 1, 1], [0, 1], []),
        # no shape: the axes are positions among the layout's dims, as before
        ((1, 2), None, [0, 1, 2], [1, 2]),
        ((3,), None, [0, 1, 2], None),
        # the layout does not hold exactly the axes of extent above 1
        ((2, 3), [4, 2, 4, 4], [0, 1, 2], None),
    ],
)
def test_layout_dims_of_axes(axes, shape, layout_dims, expected):
    assert _layout_dims_of_axes(axes, shape, layout_dims) == expected
