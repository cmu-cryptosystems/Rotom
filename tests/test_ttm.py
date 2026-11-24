"""A rank-3 tensor times a matrix, where the result spans ciphertexts."""

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
