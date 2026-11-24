"""Shape analysis computes each term's shape once, however often the term is
used."""

import pytest

from frontends.tensor import TensorTerm
from ir.analysis.shape import Shape


def _residual_chain(depth):
    """`x = x + x * x`, `depth` times: every step uses its input three times."""
    x = TensorTerm.Tensor("x", [4, 4], True)
    for _ in range(depth):
        x = x + x * x
    return x


@pytest.mark.parametrize("depth", [1, 4, 8])
def test_each_shape_is_computed_once(depth, monkeypatch):
    term = _residual_chain(depth)
    computed = []
    compute = Shape._shape

    def counted(self, t):
        computed.append(t)
        return compute(self, t)

    monkeypatch.setattr(Shape, "_shape", counted)
    shape = Shape(term)
    shape.run()

    # the input, and a product and a sum per step
    assert len(computed) == 1 + 2 * depth
    assert shape.shapes[term] == [4, 4]
    assert shape.padded_shapes[term] == [4, 4]


def test_a_returned_shape_can_be_changed_without_changing_the_analysis():
    term = _residual_chain(1)
    shape = Shape(term)
    shape.run()
    shape.get_shape(term).append(7)
    assert shape.get_shape(term) == [4, 4]


def test_a_shape_asked_for_before_the_run():
    a = TensorTerm.Tensor("a", [4, 8], True)
    b = TensorTerm.Tensor("b", [8, 2], False)
    assert Shape(a @ b).get_shape(a @ b) == [4, 2]
