"""A rank-3 tensor times a matrix, where the result spans ciphertexts."""

from copy import deepcopy

import numpy as np
import pytest

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.kernel import KernelOp
from lower.lower import Lower
from tests.test_util import get_default_args
from util.checker import check_results


@pytest.mark.parametrize(
    "shape,n,rolls",
    [
        ((8, 8, 8), 64, False),
        ((8, 8, 8), 128, False),
        ((8, 8, 8), 128, True),
        ((16, 8, 8), 256, True),
        ((8, 16, 16), 512, True),
    ],
)
def test_compaction_keeps_the_dims_that_stay_on_ciphertexts(shape, n, rolls):
    rng = np.random.default_rng(0)
    inputs = {
        "t": rng.integers(0, 3, size=shape),
        "m": rng.integers(0, 3, size=(shape[2], shape[2])),
    }
    t = TensorTerm.Tensor("t", list(shape), True)
    m = TensorTerm.Tensor("m", [shape[2], shape[2]], True)
    term = t @ m

    args = get_default_args()
    args.n = n
    args.rolls = rolls
    args.backend = "toy"
    kernel = LayoutAssignment(term, args).run()
    results = Toy(Lower(kernel).run(), inputs, args).run()
    check_results(term, inputs, kernel, results, 0, args)

    # the plan compacts onto fewer ciphertexts and still leaves some dims there
    assert kernel.op == KernelOp.COMPACT
    assert kernel.layout.ct_dims


def _program(shape):
    rng = np.random.default_rng(0)
    inputs = {
        "t": rng.integers(0, 3, size=shape),
        "m": rng.integers(0, 3, size=(shape[2], shape[2])),
    }
    t = TensorTerm.Tensor("t", list(shape), True)
    m = TensorTerm.Tensor("m", [shape[2], shape[2]], True)
    return t @ m, inputs


@pytest.mark.parametrize(
    "shape,n", [((4, 8, 8), 64), ((2, 8, 8), 64), ((2, 4, 4), 16), ((8, 8, 8), 64)]
)
def test_every_rolled_plan_computes_the_product(shape, n):
    """Not only the cheapest plan: every plan the search keeps for the matmul.

    With fewer slots than the tensor needs, t holds a dim in two pieces where
    m holds one replicated piece, and a roll by a replicated piece has to be
    by the piece at the same slots on both operands.
    """
    term, inputs = _program(shape)
    args = get_default_args()
    args.n = n
    args.rolls = True
    args.backend = "toy"
    search = LayoutAssignment(term, args)
    search.run()

    plans = list(search.kernels[term].values())
    assert len(plans) > 1
    for plan in plans:
        kernel = search.combine_kernels(deepcopy(plan))
        results = Toy(Lower(kernel).run(), inputs, args).run()
        check_results(term, inputs, kernel, results, 0, args)


def test_rolled_search_with_a_piece_that_would_face_two_dims():
    # n = 4: a replicated piece of m spans two dims of t.
    term, inputs = _program((2, 4, 4))
    args = get_default_args()
    args.n = 4
    args.rolls = True
    args.backend = "toy"
    kernel = LayoutAssignment(term, args).run()
    results = Toy(Lower(kernel).run(), inputs, args).run()
    check_results(term, inputs, kernel, results, 0, args)
