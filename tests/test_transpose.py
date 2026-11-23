"""Transpose: the layout is the input's with the two dims renamed."""

import numpy as np
import pytest

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def _tensor(name, shape, secret=True):
    return TensorTerm.Tensor(name, shape, secret)


def _graphs():
    return {
        "transpose": (lambda: _tensor("a", [4, 8]).T, (4, 8), (4, 8)),
        "transpose_of_matmul": (
            lambda: (_tensor("a", [4, 4]) @ _tensor("b", [4, 4])).T,
            (4, 4),
            (4, 4),
        ),
        "matmul_of_transpose": (
            lambda: _tensor("a", [4, 4]).T @ _tensor("b", [4, 4], False),
            (4, 4),
            (4, 4),
        ),
        "add_of_transpose": (
            lambda: _tensor("a", [4, 8]).T + _tensor("b", [8, 4]),
            (4, 8),
            (8, 4),
        ),
    }


@pytest.mark.parametrize("rolls", [False, True])
@pytest.mark.parametrize("n", [16, 32])
@pytest.mark.parametrize("name", list(_graphs()))
def test_transpose_matches_eval(name, n, rolls):
    build, a_shape, b_shape = _graphs()[name]
    term = build()
    rng = np.random.default_rng(0)
    inputs = {"a": rng.normal(size=a_shape), "b": rng.normal(size=b_shape)}

    args = get_default_args()
    args.n = n
    args.rolls = rolls
    args.backend = "toy"
    args.benchmark = f"{name}_{n}_{rolls}"

    results, kernel = run_compiler_and_backend(term, inputs, args, "toy")
    expected = apply_layout(term.eval(inputs), kernel.layout)
    assert_results_equal(expected, results, "toy")
