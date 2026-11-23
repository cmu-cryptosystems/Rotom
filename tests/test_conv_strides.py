"""Per-axis conv strides: one window geometry shared by eval, shapes and layouts."""

import numpy as np
import pytest

from assignment.alignment import spec_for_term
from frontends.tensor import TensorTerm
from frontends.tensor_args import axis_strides, window_axis
from ir.analysis.shape import Shape
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_axis_strides():
    assert axis_strides(2, 3) == (2, 2, 2)
    assert axis_strides((1, 2), 2) == (1, 2)
    assert axis_strides([1, 2, 2], 3) == (1, 2, 2)
    with pytest.raises(ValueError):
        axis_strides((1, 2), 3)
    with pytest.raises(ValueError):
        axis_strides(0, 2)


@pytest.mark.parametrize(
    "extent,kernel,stride,padding,expected",
    [
        (8, 3, 1, "valid", (6, 0, 0)),
        (8, 3, 2, "valid", (3, 0, 0)),
        (8, 3, 1, "same", (8, 1, 1)),
        (8, 2, 1, "same", (8, 0, 1)),
        (8, 3, 2, "same", (4, 1, 1)),
        (32, 7, 2, "same", (16, 3, 3)),
    ],
)
def test_window_axis(extent, kernel, stride, padding, expected):
    assert window_axis(extent, kernel, stride, padding) == expected


def test_window_axis_rejects_unknown_padding():
    with pytest.raises(NotImplementedError):
        window_axis(8, 3, 1, "full")


def _conv2d(stride):
    a = TensorTerm.Tensor("a", [2, 8, 8], True)
    b = TensorTerm.Tensor("b", [4, 2, 3, 3], False)
    return TensorTerm.conv2d(a, b, stride, "same")


def _conv3d(stride):
    a = TensorTerm.Tensor("a", [2, 4, 4, 4], True)
    b = TensorTerm.Tensor("b", [2, 2, 3, 3, 3], False)
    return TensorTerm.conv3d(a, b, stride, "same")


def test_spec_has_one_stride_per_axis():
    spec = spec_for_term(_conv2d((1, 2)), [[2, 8, 8], [4, 2, 3, 3]])
    assert [r.stride for r in spec.relations] == [1, 2]
    spec = spec_for_term(_conv3d((1, 2, 2)), [[2, 4, 4, 4], [2, 2, 3, 3, 3]])
    assert [r.stride for r in spec.relations] == [1, 2, 2]


@pytest.mark.parametrize(
    "term,expected",
    [
        (_conv2d((1, 2)), [4, 8, 4]),
        (_conv2d((2, 1)), [4, 4, 8]),
        (_conv2d(2), [4, 4, 4]),
        (_conv3d((1, 2, 2)), [2, 4, 2, 2]),
        (_conv3d((2, 1, 1)), [2, 2, 4, 4]),
    ],
)
def test_shape_analysis_uses_each_axis_stride(term, expected):
    shape = Shape(term)
    shape.run()
    assert shape.get_shape(term) == expected


def _inputs(term):
    rng = np.random.default_rng(0)
    return {operand.cs[0]: rng.normal(size=operand.cs[1]) for operand in term.cs[:2]}


@pytest.mark.parametrize(
    "build,stride",
    [
        (_conv2d, (1, 2)),
        (_conv2d, (2, 1)),
        (_conv2d, (2, 2)),
        (_conv3d, (1, 2, 2)),
        (_conv3d, (2, 1, 1)),
    ],
)
def test_per_axis_strides_match_eval(build, stride):
    term = build(stride)
    inputs = _inputs(term)
    # A tuple of equal strides is the int stride.
    if len(set(stride)) == 1:
        assert np.array_equal(term.eval(inputs), build(stride[0]).eval(inputs))

    args = get_default_args()
    args.n = 512
    args.backend = "toy"
    args.benchmark = f"conv_strides_{len(stride)}d_{'_'.join(map(str, stride))}"
    results, kernel = run_compiler_and_backend(term, inputs, args, "toy")
    expected = apply_layout(term.eval(inputs), kernel.layout)
    assert_results_equal(expected, results, "toy")
