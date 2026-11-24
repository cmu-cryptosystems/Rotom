"""Unary operators take their output layout from their operator spec."""

import numpy as np
import pytest

from assignment.alignment import (
    index_spec,
    permute_spec,
    reduce_spec,
    split_spec,
)
from assignment.assignment import LayoutAssignment
from assignment.gen.gen_unary import output_pieces, unary_layout
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from ir.layout import Layout
from ir.roll import Roll
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results
from util.layout_util import apply_layout, parse_layout


def _pieces(spec, layout, out_shape=()):
    """The output pieces of each input piece that holds a dim."""
    dims = parse_layout(layout, 64, True).get_dims()
    held = output_pieces(spec, dims, out_shape)
    if held is None:
        return None
    return [
        "".join(str(p) for p in ps)
        for piece, ps in zip(dims, held)
        if piece.dim is not None
    ]


def test_a_kept_label_moves_to_its_output_dim():
    spec = permute_spec({0: 2, 1: 0, 2: 1})
    assert _pieces(spec, "[1:8:1][2:2:1][0:4:1]") == ["[0:8:1]", "[1:2:1]", "[2:4:1]"]


def test_a_dropped_label_becomes_a_gap_and_later_dims_move_up():
    spec = reduce_spec(3, 0)
    assert _pieces(spec, "[1:8:1][2:2:1][0:4:1]") == ["[0:8:1]", "[1:2:1]", "[G:4]"]


def test_a_split_sends_each_piece_to_the_outer_or_inner_dim():
    spec = split_spec(2, 1, 2, 4)
    assert _pieces(spec, "[0:4:1][1:2:4][1:4:1]") == ["[0:4:1]", "[1:2:1]", "[2:4:1]"]


def test_a_split_cuts_the_piece_that_crosses_it():
    spec = split_spec(2, 1, 2, 4)
    assert _pieces(spec, "[0:4:1][1:8:1]") == ["[0:4:1]", "[1:2:1][2:4:1]"]


def test_a_split_with_a_padded_inner_dim_is_not_a_relabelling():
    # d = 3*outer + inner sits at digit 3*outer + inner, not 4*outer + inner
    assert _pieces(split_spec(1, 0, 2, 3), "[0:8:1]") is None
    # one outer element: every index is its inner index
    assert _pieces(split_spec(1, 0, 1, 3), "[0:4:1]") == ["[1:4:1]"]


def test_a_slice_shrinks_the_dim_and_keeps_it_dense():
    spec = index_spec(3, (slice(None), slice(None, None, 2)))
    assert _pieces(spec, "[0:4:1][1:8:1][2:2:1]", (4, 4, 2)) == [
        "[0:4:1]",
        "[1:4:1]",
        "[2:2:1]",
    ]
    # the coarsest piece gives up the extent
    assert _pieces(spec, "[1:2:4][0:4:1][1:4:1][2:2:1]", (4, 4, 2)) == [
        "",
        "[0:4:1]",
        "[1:4:1]",
        "[2:2:1]",
    ]


def test_an_int_index_drops_its_dim():
    spec = index_spec(3, (slice(None), 3))
    assert _pieces(spec, "[0:4:1][2:2:1][1:8:1]", (4, 2)) == [
        "[0:4:1]",
        "[1:2:1]",
        "[G:8]",
    ]


def _rolled(layout, rolled, rolled_by):
    dims = parse_layout(layout, 64, True).get_dims()
    return Layout(None, [Roll(dims[rolled], dims[rolled_by])], dims, 64, True)


def test_a_roll_follows_its_pieces_through_a_permute():
    layout, _ = unary_layout(
        permute_spec({0: 1, 1: 0}), None, _rolled("[0:8:1][1:8:1]", 0, 1), (8, 8)
    )
    assert layout.layout_str().endswith("[1:8:1][0:8:1]")
    (roll,) = layout.rolls
    assert (str(roll.dim_to_roll), str(roll.dim_to_roll_by)) == ("[1:8:1]", "[0:8:1]")


def test_a_sum_over_the_rolled_dim_drops_the_roll():
    layout, dropped = unary_layout(
        reduce_spec(2, 0), None, _rolled("[0:8:1][1:8:1]", 0, 1), (8,)
    )
    assert layout.layout_str() == "[G:8][0:8:1]"
    assert [str(piece) for piece in dropped] == ["[0:8:1]"]


def test_a_sum_over_the_dim_another_is_rolled_by_is_not_carried():
    # each position of the kept dim would sum elements of different indices
    assert (
        unary_layout(reduce_spec(2, 1), None, _rolled("[0:8:1][1:8:1]", 0, 1), (8,))
        is None
    )


def _run(build, shapes, n=64, rolls=False):
    rng = np.random.default_rng(0)
    inputs, tensors = {}, []
    for i, shape in enumerate(shapes):
        inputs[f"t{i}"] = rng.integers(-4, 5, size=shape).astype(float)
        tensors.append(TensorTerm.Tensor(f"t{i}", list(shape), i == 0))
    term = build(*tensors)
    args = get_default_args()
    args.n = n
    args.rolls = rolls
    args.backend = "toy"
    kernel = LayoutAssignment(term, args).run()
    results = Toy(Lower(kernel).run(), inputs, args).run()
    check_results(term, inputs, kernel, results, 0, args)

    # the output layout holds every element of the result
    expected = term.eval(inputs)
    numbered = np.arange(1, expected.size + 1, dtype=float).reshape(expected.shape)
    held = {v for ct in apply_layout(numbered, kernel.layout) for v in ct if v}
    assert held == set(numbered.ravel().tolist())
    return kernel


RANK3 = [(4, 8, 2)]
CASES = {
    "sum over the first dim": (lambda x: x.sum(0), RANK3),
    "sum over a middle dim": (lambda x: x.sum(1), RANK3),
    "sum then add": (lambda x, y: x.sum(0) + y, [(4, 8, 2), (8, 2)]),
    "sum twice": (lambda x: x.sum(0).sum(0), RANK3),
    "sum then matmul": (lambda x, y: x.sum(0) @ y, [(4, 8, 2), (2, 4)]),
    "sum then transpose": (lambda x: x.sum(0).transpose(), RANK3),
    "product over the first dim": (lambda x: x.product(0), RANK3),
    "product then add": (lambda x, y: x.product(1) + y, [(4, 8, 2), (4, 2)]),
    "reshape then sum": (lambda x: x.reshape(1, {1: 2, 2: 4}).sum(1), [(4, 8)]),
    "slice with a step": (lambda x: x[:, ::2], RANK3),
    "slice with a start": (lambda x: x[:, 2:6], RANK3),
    "slice then add": (lambda x, y: x[:, ::2] + y, [(4, 8, 2), (4, 4, 2)]),
    "negative index": (lambda x: x[-1], RANK3),
    "index after a matmul": (lambda x, y: (x @ y)[3], [(8, 8), (8, 8)]),
    "sum after a matmul": (lambda x, y: (x @ y).sum(0), [(8, 8), (8, 8)]),
    "permute then add": (
        lambda x, y: x.permute({0: 2, 1: 0, 2: 1}) + y,
        [(4, 8, 2), (8, 2, 4)],
    ),
}


@pytest.mark.parametrize("rolls", [False, True])
@pytest.mark.parametrize("case", list(CASES))
def test_unary_operator_matches_eval(case, rolls):
    build, shapes = CASES[case]
    _run(build, shapes, rolls=rolls)


def test_sum_numbers_its_output_dims_from_zero():
    kernel = _run(lambda x: x.sum(0), RANK3)
    assert kernel.op == KernelOp.SUM
    assert {piece.dim for piece in kernel.layout.get_dims()} == {0, 1, None}
