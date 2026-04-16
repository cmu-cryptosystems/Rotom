import numpy as np

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_avg_pool2d_toy_e2e_k2_s2_valid() -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = 256
    args.skip_toy_eval_checks = True

    x = np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8)
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [2, 8, 8], True).avg_pool2d(
        kernel=2, stride=2, padding="valid"
    )
    results, kernel = run_compiler_and_backend(t, inputs, args, "toy")

    expected_dense = (
        x[:, 0::2, 0::2] + x[:, 0::2, 1::2] + x[:, 1::2, 0::2] + x[:, 1::2, 1::2]
    ) / 4.0
    expected_cts = apply_layout(expected_dense, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
