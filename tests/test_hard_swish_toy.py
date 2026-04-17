import numpy as np

from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout
from frontends.tensor import TensorTerm


def _hard_swish(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * np.clip(x + 3.0, 0.0, 6.0) / 6.0


def test_hard_swish_toy_e2e() -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = 64
    args.skip_toy_eval_checks = True

    x = np.array(
        [
            [-4.0, -3.0, -2.0, -1.0],
            [0.0, 1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0, 7.0],
            [-7.0, -6.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [4, 4], True).hard_swish()
    results, kernel = run_compiler_and_backend(t, inputs, args, "toy")

    expected_dense = _hard_swish(x)
    expected_cts = apply_layout(expected_dense, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
