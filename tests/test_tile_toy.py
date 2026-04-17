import numpy as np

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_tile_toy_e2e_axis1_repeat4() -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = 256
    args.skip_toy_eval_checks = True

    x = np.arange(2 * 1 * 8, dtype=np.float64).reshape(2, 1, 8)
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [2, 1, 8], True).tile([1, 4, 1])
    results, kernel = run_compiler_and_backend(t, inputs, args, "toy")

    expected_dense = np.tile(x, (1, 4, 1))
    expected_cts = apply_layout(expected_dense, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
