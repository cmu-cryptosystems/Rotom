"""Operator specs: the alignment, sum dims and output dims derived from labels."""

import numpy as np
import pytest

from assignment.alignment import (
    OperatorSpec,
    Window,
    block_matmul_spec,
    conv_spec,
    elementwise_spec,
    get_dim_alignment,
    matmul_spec,
    spec_for_term,
)
from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout
from util.util import get_sum_dim


def _tensors(lhs_rank, rhs_rank):
    a = TensorTerm.Tensor("a", [4] * lhs_rank, True)
    b = TensorTerm.Tensor("b", [4] * rhs_rank, False)
    return a, b, [[4] * lhs_rank, [4] * rhs_rank]


# The alignment sets the hand-written get_dim_alignment returned before specs.
@pytest.mark.parametrize(
    "lhs_rank,rhs_rank,expected",
    [
        (1, 1, {(0, 0)}),
        (2, 2, {(0, 0), (1, 1)}),
        (3, 3, {(0, 0), (1, 1), (2, 2)}),
        (2, 1, {(1, 0), (0, None)}),
        (3, 1, {(2, 0), (1, None), (0, None)}),
        (1, 2, {(0, 1), (None, 0)}),
        (1, 3, {(0, 2), (None, 1), (None, 0)}),
    ],
)
def test_elementwise_alignment_matches_previous(lhs_rank, rhs_rank, expected):
    a, b, shapes = _tensors(lhs_rank, rhs_rank)
    for term in (a + b, a - b, a * b):
        assert get_dim_alignment(term, shapes) == expected


@pytest.mark.parametrize(
    "lhs_rank,rhs_rank,expected",
    [
        (2, 1, {(0, None), (1, 0)}),
        (2, 2, {(0, None), (1, 0), (None, 1)}),
        (3, 2, {(0, None), (1, None), (2, 0), (None, 1)}),
    ],
)
def test_matmul_alignment_matches_previous(lhs_rank, rhs_rank, expected):
    a, b, shapes = _tensors(lhs_rank, rhs_rank)
    assert get_dim_alignment(a @ b, shapes) == expected


def test_vector_matrix_alignment_contracts_the_first_rhs_dim():
    # The previous table had {(None, 0), (0, 1)}, which is b @ a.
    a, b, shapes = _tensors(1, 2)
    assert get_dim_alignment(a @ b, shapes) == {(0, 0), (None, 1)}


def test_matmul_unsupported_ranks_still_raise():
    a, b, shapes = _tensors(3, 3)
    with pytest.raises(NotImplementedError):
        get_dim_alignment(a @ b, shapes)


def test_block_matmul_alignment_matches_previous():
    assert block_matmul_spec().alignment() == {(0, 0), (1, None), (2, 1), (None, 2)}
    assert block_matmul_spec().lhs_sum_dims() == (2,)
    assert block_matmul_spec().rhs_sum_dims() == (1,)


@pytest.mark.parametrize("lhs_rank,rhs_rank", [(2, 1), (1, 2), (2, 2), (3, 2)])
def test_matmul_sum_dims_match_get_sum_dim(lhs_rank, rhs_rank):
    a, b, shapes = _tensors(lhs_rank, rhs_rank)
    term = a @ b
    spec = spec_for_term(term, shapes)

    class _Kernel:
        def __init__(self, operand, rank):
            self.layout = type("L", (), {})()
            self.layout.term = operand
            self.layout.get_dims = lambda: [
                type("D", (), {"dim": d})() for d in range(rank)
            ]

    assert spec.lhs_sum_dims() == (get_sum_dim(term, _Kernel(a, lhs_rank)),)
    assert spec.rhs_sum_dims() == (get_sum_dim(term, _Kernel(b, rhs_rank)),)


def test_output_dims():
    assert matmul_spec(2, 2).lhs_out_dims() == (0, None)
    assert matmul_spec(2, 2).rhs_out_dims() == (None, 1)
    assert matmul_spec(1, 2).rhs_out_dims() == (None, 0)
    assert matmul_spec(3, 2).lhs_out_dims() == (0, 1, None)
    assert matmul_spec(3, 2).rhs_out_dims() == (None, 2)
    assert elementwise_spec(1, 3).lhs_out_dims() == (2,)
    assert elementwise_spec(1, 3).rhs_out_dims() == (0, 1, 2)


def test_conv2d_spec():
    a = TensorTerm.Tensor("a", [4, 8, 8], True)
    b = TensorTerm.Tensor("b", [8, 4, 3, 3], False)
    term = TensorTerm.conv2d(a, b, 2, "same")
    spec = spec_for_term(term, [[4, 8, 8], [8, 4, 3, 3]])

    assert str(spec) == (
        "c_in h w, c_out c_in kh kw -> c_out h_out w_out"
        " [h_out=2*h+kh] [w_out=2*w+kw]"
    )
    assert spec.accepts([[4, 8, 8], [8, 4, 3, 3]])
    assert not spec.accepts([[4, 8, 8], [8, 4, 3]])
    # Input channel aligns with filter dim 1; the output channel and the taps
    # face replication on the input, the spatial dims face it on the filter.
    assert spec.alignment() == {
        (0, 1),
        (1, None),
        (2, None),
        (None, 0),
        (None, 2),
        (None, 3),
    }
    assert spec.contracted() == ("c_in",)
    assert spec.lhs_sum_dims() == (0,)
    assert spec.rhs_sum_dims() == (1, 2, 3)
    assert spec.lhs_out_dims() == (None, 1, 2)
    assert spec.rhs_out_dims() == (0, None, None, None)
    assert spec.window_of("h") == Window(out="h_out", inp="h", tap="kh", stride=2)


def test_conv3d_spec_and_per_axis_strides():
    a = TensorTerm.Tensor("a", [4, 4, 8, 8], True)
    b = TensorTerm.Tensor("b", [8, 4, 3, 3, 3], False)
    term = TensorTerm.conv3d(a, b, 1, "same")
    spec = spec_for_term(term, [[4, 4, 8, 8], [8, 4, 3, 3, 3]])
    assert spec.lhs == ("c_in", "d", "h", "w")
    assert spec.rhs == ("c_out", "c_in", "kd", "kh", "kw")
    assert spec.out == ("c_out", "d_out", "h_out", "w_out")
    assert spec.lhs_out_dims() == (None, 1, 2, 3)

    # A stride per axis is a different spec, not a different operator.
    spatial_only = conv_spec([1, 2, 2])
    assert [w.stride for w in spatial_only.windows] == [1, 2, 2]


def test_depthwise_conv_spec_carries_the_channel():
    spec = conv_spec([1, 1], depthwise=True)
    assert str(spec) == "c h w, c kh kw -> c h_out w_out [h_out=1*h+kh] [w_out=1*w+kw]"
    assert spec.contracted() == ()
    assert spec.alignment() == {(0, 0), (1, None), (2, None), (None, 1), (None, 2)}
    assert spec.lhs_out_dims() == (0, 1, 2)


@pytest.mark.parametrize(
    "spec,windows",
    [
        ("ik,kj", ()),  # no output
        ("ii,ij->ij", ()),  # repeated label
        ("ik,kj->ijz", ()),  # output label with no source
        ("ik,j->j", ()),  # label summed over one operand
        ("cy,ocp->oY", (Window("Y", "y", "p", 0),)),  # stride 0
        ("cy,ocp->oy", (Window("y", "y", "p"),)),  # window out is not output-only
    ],
)
def test_invalid_specs_are_rejected(spec, windows):
    with pytest.raises(ValueError):
        OperatorSpec.parse(spec, windows)


@pytest.mark.parametrize("rolls", [False, True])
def test_vector_matrix_matmul_matches_eval(rolls):
    args = get_default_args()
    args.n = 16
    args.rolls = rolls
    args.backend = "toy"
    args.benchmark = f"vector_matrix_matmul_{rolls}"

    a = TensorTerm.Tensor("a", [4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    term = a @ b
    rng = np.random.default_rng(0)
    inputs = {"a": rng.normal(size=4), "b": rng.normal(size=(4, 4))}

    expected = term.eval(inputs)
    assert np.allclose(expected, inputs["a"] @ inputs["b"])
    results, kernel = run_compiler_and_backend(term, inputs, args, "toy")
    assert_results_equal(apply_layout(expected, kernel.layout), results, "toy")
