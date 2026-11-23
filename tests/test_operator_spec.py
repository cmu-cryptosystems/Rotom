"""Operator specs: the alignment, sum dims and output dims derived from labels."""

import numpy as np
import pytest

from assignment.alignment import (
    OperatorSpec,
    Relation,
    conv_spec,
    elementwise_spec,
    get_dim_alignment,
    identity_spec,
    index_spec,
    matmul_spec,
    permute_spec,
    reduce_spec,
    spec_for_term,
    split_spec,
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
        " [h=2*h_out+kh] [w=2*w_out+kw]"
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
    window = Relation(label="h", base="h_out", offset="kh", stride=2)
    assert spec.relation_of("h") == window
    assert spec.relation_of("kh") == window
    assert spec.relation_of("c_in") is None


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
    assert [r.stride for r in spatial_only.relations] == [1, 2, 2]


def test_depthwise_conv_spec_carries_the_channel():
    spec = conv_spec([1, 1], depthwise=True)
    assert str(spec) == "c h w, c kh kw -> c h_out w_out [h=h_out+kh] [w=w_out+kw]"
    assert spec.contracted() == ()
    assert spec.alignment() == {(0, 0), (1, None), (2, None), (None, 1), (None, 2)}
    assert spec.lhs_out_dims() == (0, 1, 2)


@pytest.mark.parametrize(
    "spec,relations",
    [
        ("ik,kj", ()),  # no output
        ("ii,ij->ij", ()),  # repeated label
        ("ik,kj->ijz", ()),  # output label with no source
        ("ik,j->j", ()),  # label summed over one operand
        ("cy,ocp->oY", (Relation("y", "Y", "p", stride=0),)),  # stride 0
        ("cy,ocp->oy", (Relation("y", "y", "p"),)),  # related label in the output
        ("cy,ocp->oY", (Relation("c", "Y", "p"),)),  # related label in the rhs
        ("cy,ocp->oY", (Relation("y", "Y", "q"),)),  # term in no operand
        (
            "cyx,ocp->oY",
            (Relation("y", "Y", "p"), Relation("x", "Y")),
        ),  # term read twice
        ("ab->az", ()),  # unary output label with no source
        ("aa->a", ()),  # unary repeated label
    ],
)
def test_invalid_specs_are_rejected(spec, relations):
    with pytest.raises(ValueError):
        OperatorSpec.parse(spec, relations)


def test_unary_spec_parses_and_prints():
    spec = OperatorSpec.parse("ab->a")
    assert spec.is_unary and spec.rhs is None
    assert str(spec) == "ab->a"
    assert spec.accepts([[4, 4]])
    assert not spec.accepts([[4, 4], [4]])
    assert spec.reduced() == ("b",)
    assert spec.lhs_sum_dims() == (1,)
    assert spec.rhs_sum_dims() == ()
    assert spec.rhs_out_dims() == ()
    with pytest.raises(ValueError):
        spec.alignment()
    # A binary spec with an empty rhs (a constant) is not a unary spec.
    assert not OperatorSpec.parse("ab,->ab").is_unary
    # A binary operator still may not sum a label over one operand.
    with pytest.raises(ValueError):
        OperatorSpec.parse("ab,->a")


def _unary(rank=2, extent=4):
    return TensorTerm.Tensor("a", [extent] * rank, True), [[extent] * rank]


def test_slotwise_unary_specs_carry_every_dim():
    a, shapes = _unary(3)
    for term in (a.poly_call("silu", -1.0, 1.0),):
        spec = spec_for_term(term, shapes)
        assert spec == identity_spec(3)
        assert str(spec) == "d0 d1 d2 -> d0 d1 d2"
        assert spec.lhs_out_dims() == (0, 1, 2)
        assert spec.lhs_sum_dims() == ()


def test_permute_and_transpose_specs():
    a, shapes = _unary(3)
    spec = spec_for_term(a.permute({0: 1, 1: 2, 2: 0}), shapes)
    assert spec.out == ("d2", "d0", "d1")
    assert spec.lhs_out_dims() == (1, 2, 0)
    assert spec.lhs_sum_dims() == ()

    m, m_shapes = _unary(2)
    assert spec_for_term(m.T, m_shapes) == permute_spec({0: 1, 1: 0})
    assert spec_for_term(m.T, m_shapes).lhs_out_dims() == (1, 0)
    with pytest.raises(ValueError):
        permute_spec({0: 0, 1: 0})


def test_sum_and_product_specs_reduce_one_dim():
    a, shapes = _unary(3)
    for term in (a.sum(1), a.product(1)):
        spec = spec_for_term(term, shapes)
        assert spec == reduce_spec(3, 1)
        assert str(spec) == "d0 d1 d2 -> d0 d2"
        assert spec.reduced() == ("d1",)
        assert spec.lhs_sum_dims() == (1,)
        assert spec.lhs_out_dims() == (0, None, 1)


def test_reshape_spec_splits_the_last_dim():
    a = TensorTerm.Tensor("a", [4, 32], True)
    spec = spec_for_term(a.reshape(1, {1: 4, 2: 8}), [[4, 32]])
    assert spec == split_spec(2, 1, 4, 8)
    assert str(spec) == "d0 d1 -> d0 d1.0 d1.1 [d1=8*d1.0+d1.1]"
    # The split dim is neither reduced nor carried; it becomes its outer part.
    assert spec.reduced() == ()
    assert spec.lhs_sum_dims() == ()
    assert spec.lhs_out_dims() == (0, 1)
    assert spec.relation_of("d1.1").label == "d1"


def test_index_spec_fixes_ints_and_keeps_slices():
    a = TensorTerm.Tensor("a", [4, 8, 8], True)
    shapes = [[4, 8, 8]]

    spec = spec_for_term(a[2], shapes)
    assert str(spec) == "d0 d1 d2 -> d1 d2 [d0=2]"
    assert spec.reduced() == ()
    assert spec.lhs_out_dims() == (None, 0, 1)

    # bert_attention's head select: an int, then whole-dim slices.
    assert spec_for_term(a[2, :, :], shapes) == spec

    spec = index_spec(3, (slice(None), slice(1, 7, 2), slice(0, 4)))
    assert str(spec) == "d0 d1 d2 -> d0 d1_out d2 [d1=2*d1_out+1]"
    assert spec.lhs_out_dims() == (0, 1, 2)

    with pytest.raises(NotImplementedError):
        index_spec(3, -1)
    with pytest.raises(ValueError):
        index_spec(1, (0, 0))


def test_roll_is_a_relation_with_a_modulus():
    # A matmul whose contraction dim is rolled by j: k = (j + d) mod 4.
    roll = Relation(label="k", base="j", offset="d", modulus=4)
    assert str(roll) == "k=(j+d) mod 4"
    spec = OperatorSpec(("i", "k"), ("d", "j"), ("i", "j"), (roll,))
    assert spec.rhs_sum_dims() == (0,)
    assert spec.lhs_out_dims() == (0, 1)


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


def _bert_attention_small():
    h = TensorTerm.Tensor("h", [4, 8], True)
    w = TensorTerm.Tensor("w", [8, 8], False)
    b = TensorTerm.Tensor("b", [8], False)
    q = (h @ w + b).reshape(1, {1: 2, 2: 4}).permute({0: 1, 1: 0, 2: 2})
    k = (h @ w + b).reshape(1, {1: 2, 2: 4}).permute({0: 2, 1: 0, 2: 1})
    return (q[0, :, :] @ k[0, :, :]).sum(1).poly_call("silu", -1.0, 1.0)


def test_every_term_of_a_graph_has_a_spec_of_the_right_ranks():
    from ir.analysis.shape import Shape

    root = _bert_attention_small()
    shape = Shape(root)
    shape.run()
    seen = set()
    for term in root.post_order():
        if term.op.name in ("TENSOR", "CONST"):
            continue
        operands = [c for c in term.cs if isinstance(c, TensorTerm)]
        shapes = [shape.get_shape(c) for c in operands]
        spec = spec_for_term(term, shapes)
        assert spec.accepts(shapes), f"{term.op}: {spec}"
        assert len(spec.out) == len(shape.get_shape(term)), f"{term.op}: {spec}"
        assert spec.is_unary == (len(operands) == 1)
        seen.add(term.op.name)
    assert seen == {"MATMUL", "ADD", "RESHAPE", "PERMUTE", "INDEX", "SUM", "POLY_CALL"}
