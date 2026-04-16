import numpy as np

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_cumsum_toy_e2e_axis1() -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = 128
    args.skip_toy_eval_checks = True

    x = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [2, 4], True).cumsum(axis=1)
    results, kernel = run_compiler_and_backend(t, inputs, args, "toy")

    expected_dense = np.cumsum(x, axis=1)
    expected_cts = apply_layout(expected_dense, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
