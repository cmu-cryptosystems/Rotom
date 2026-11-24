"""Elementwise operators whose operands broadcast over axes of extent 1."""

import numpy as np
import pytest

from assignment.alignment import elementwise_spec
from frontends.tensor import TensorTerm
from ir.analysis.shape import Shape
from ir.kernel import KernelOp
from tests.test_conv_direct import _run


def test_a_unit_axis_holds_no_label():
    spec = elementwise_spec([6, 4, 8, 8], [6, 4, 1, 1])
    assert str(spec) == "d0 d1 d2 d3, d0 d1 u2 u3 -> d0 d1 d2 d3"
    assert spec.units == ("u2", "u3")
    assert spec.alignment() == {(0, 0), (1, 1), (2, None), (3, None)}
    assert spec.accepts([[6, 4, 8, 8], [6, 4, 1, 1]])


def test_a_unit_axis_on_the_left():
    spec = elementwise_spec([1, 4, 1], [6, 4, 8])
    assert str(spec) == "u0 d1 u2, d0 d1 d2 -> d0 d1 d2"
    assert spec.alignment() == {(1, 1), (None, 0), (None, 2)}


def test_units_and_rank_broadcasting():
    spec = elementwise_spec([6, 4, 8], [1, 8])
    assert str(spec) == "d0 d1 d2, u1 d2 -> d0 d1 d2"
    assert spec.alignment() == {(0, None), (1, None), (2, 1)}


def test_a_unit_axis_in_both_operands_is_shared():
    spec = elementwise_spec([6, 1, 8], [6, 1, 8])
    assert spec == elementwise_spec(3, 3)
    assert spec.units == ()


def test_shape_of_a_broadcast():
    a = TensorTerm.Tensor("a", [6, 4, 1, 1], True)
    b = TensorTerm.Tensor("b", [6, 4, 8, 8], True)
    c = TensorTerm.Tensor("c", [1, 8], True)
    for term, shape, padded in [
        (a * b, [6, 4, 8, 8], [8, 4, 8, 8]),
        (b * a, [6, 4, 8, 8], [8, 4, 8, 8]),
        (b + c, [6, 4, 8, 8], [8, 4, 8, 8]),
    ]:
        analysis = Shape(term)
        analysis.run()
        assert analysis.get_shape(term) == shape
        assert analysis.padded_shapes[term] == padded


CASES = [
    # lhs shape, rhs shape, rhs secret, op
    ((6, 4, 8, 8), (6, 4, 1, 1), True, "mul"),  # a squeeze-excitation gate
    ((6, 4, 1, 1), (6, 4, 8, 8), True, "mul"),
    ((4, 8, 8), (4, 1, 1), True, "mul"),
    ((4, 8, 8), (4, 1, 1), True, "add"),
    ((4, 8, 8), (4, 1, 1), True, "sub"),
    ((6, 4, 8, 8), (6, 1, 1, 1), False, "add"),  # a bias per channel
    ((6, 4, 8, 8), (6, 1, 1, 1), False, "mul"),
    ((6, 4, 4, 4), (1, 4, 1, 1), False, "mul"),  # a scale per frame
    ((4, 8, 8), (1, 8, 8), True, "mul"),
    ((4, 8, 8), (4, 1, 8), True, "add"),
    ((4, 8, 8), (1, 8), True, "mul"),
    ((5, 8), (5, 1), True, "mul"),
]


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("n", [256, 2048])
def test_broadcast_matches_eval(case, n):
    lhs_shape, rhs_shape, rhs_secret, op = CASES[case]
    rng = np.random.default_rng(case)
    a = TensorTerm.Tensor("a", list(lhs_shape), True)
    b = TensorTerm.Tensor("b", list(rhs_shape), rhs_secret)
    inputs = {"a": rng.normal(size=lhs_shape), "b": rng.normal(size=rhs_shape)}
    term = {"mul": a * b, "add": a + b, "sub": a - b}[op]
    expected = {"mul": np.multiply, "add": np.add, "sub": np.subtract}[op](
        inputs["a"], inputs["b"]
    )
    # the evaluator pads a matrix to powers of two
    got = term.eval(inputs)[tuple(slice(extent) for extent in expected.shape)]
    np.testing.assert_allclose(got, expected)
    _run(term, inputs, n)


def test_a_gate_from_a_mean_scales_its_features():
    """Squeeze and excite: features times a function of their own mean."""
    rng = np.random.default_rng(3)
    x = TensorTerm.Tensor("x", [4, 4, 8, 8], True)
    inputs = {"x": rng.normal(size=(4, 4, 8, 8))}
    term = x * x.mean((2, 3))
    kernel = _run(term, inputs, 2048)
    assert any(k.op == KernelOp.REPLICATE for k in kernel.post_order())
