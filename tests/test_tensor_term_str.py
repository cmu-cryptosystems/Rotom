"""A term's string stays short when subterms are shared."""

import numpy as np

from frontends.tensor import TensorTerm
from tests.test_conv_direct import _run


def _residual(depth):
    x = TensorTerm.Tensor("x", [4, 4], True)
    h = x
    for i in range(depth):
        h = (h @ TensorTerm.Tensor(f"w{i}", [4, 4], False)) + h
    return h


def test_a_small_term_is_written_in_full():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    assert str((a @ b) + a) == "(+ (@ a b) a)"


def test_a_shared_subterm_does_not_double_the_string():
    # written in full, 64 residual blocks are 2**64 copies of x
    term = _residual(64)
    assert len(str(term)) < 400
    assert len(term.post_order()) == 1 + 3 * 64


def test_terms_that_differ_deep_down_are_not_equal():
    x = TensorTerm.Tensor("x", [4, 4], True)
    y = TensorTerm.Tensor("y", [4, 4], True)

    def stack(h):
        for i in range(12):
            h = (h @ TensorTerm.Tensor(f"w{i}", [4, 4], False)) + h
        return h

    assert stack(x) == stack(x)
    assert stack(x) != stack(y)
    assert str(stack(x)) != str(stack(y))


def test_a_deep_residual_program_compiles_and_matches_eval():
    rng = np.random.default_rng(0)
    term = _residual(6)
    inputs = {"x": rng.normal(size=(4, 4))}
    inputs.update({f"w{i}": rng.normal(size=(4, 4)) * 0.3 for i in range(6)})
    _run(term, inputs, 64)
