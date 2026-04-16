import numpy as np

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_concat_toy_e2e_axis1() -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = 128
    args.skip_toy_eval_checks = True

    x0 = np.arange(2 * 2 * 8, dtype=np.float64).reshape(2, 2, 8)
    x1 = (1000.0 + np.arange(2 * 2 * 8, dtype=np.float64)).reshape(2, 2, 8)
    inputs = {"x0": x0, "x1": x1}

    t0 = TensorTerm.Tensor("x0", [2, 2, 8], True)
    t1 = TensorTerm.Tensor("x1", [2, 2, 8], True)
    t = TensorTerm.concat([t0, t1], axis=1)
    results, kernel = run_compiler_and_backend(t, inputs, args, "toy")

    expected_dense = np.concatenate([x0, x1], axis=1)
    expected_cts = apply_layout(expected_dense, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
