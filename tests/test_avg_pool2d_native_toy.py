import numpy as np

from frontends.tensor import TensorTerm
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_avg_pool2d_native_toy_k2_s2_valid_with_layout() -> None:
    """Native ``AVG_POOL2D`` path (layout forces op; area 4 → ``RESCALE`` by 2^2)."""
    args = get_default_args()
    args.backend = "toy"
    args.n = 256
    args.skip_toy_eval_checks = True

    x = np.arange(2 * 8 * 8, dtype=np.float64).reshape(2, 8, 8)
    inputs = {"x": x}

    t = TensorTerm.Tensor("x", [2, 8, 8], True).avg_pool2d(
        kernel=2, stride=2, padding="valid", layout="[0:2:1][1:8:1][2:8:1]"
    )
    results, kernel = run_compiler_and_backend(t, inputs, args, "toy")

    # Reference: channel-wise avg pool (matches ``TensorEvaluator``).
    c, h, w = x.shape
    h_o = (h - 2) // 2 + 1
    w_o = (w - 2) // 2 + 1
    expected_dense = np.zeros((c, h_o, w_o), dtype=np.float64)
    for ch in range(c):
        for i in range(h_o):
            for j in range(w_o):
                hs, ws = i * 2, j * 2
                expected_dense[ch, i, j] = float(
                    np.mean(x[ch, hs : hs + 2, ws : ws + 2])
                )

    expected_cts = apply_layout(expected_dense, kernel.layout)
    assert_results_equal(expected_cts, results, "toy")
