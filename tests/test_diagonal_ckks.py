import numpy as np

from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_backend
from tests.diagonal_helpers import direct_conv2d_kernel, direct_matmul_kernel
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_direct_diagonal_matmul_ckks_smoke_ct_pt():
    inputs = {
        "a": np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]),
        "b": np.array([[1, 2, 0, 1], [0, 1, 1, 0], [2, 0, 1, 1], [1, 1, 0, 2]]),
    }
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    term = a @ b
    kernel = direct_matmul_kernel(a, b, term)

    _assert_direct_kernel_ckks_execution(kernel, term, inputs, "diagonal_matmul_ckks")


def test_direct_diagonal_conv2d_ckks_smoke_pointwise():
    inputs = {
        "x": np.arange(16).reshape(1, 4, 4),
        "w": np.array([[[[2]]]]),
    }
    x = TensorTerm.Tensor("x", [1, 4, 4], True)
    w = TensorTerm.Tensor("w", [1, 1, 1, 1], False)
    term = TensorTerm.conv2d(x, w, 1, "same")
    kernel = direct_conv2d_kernel(x, w, term)

    _assert_direct_kernel_ckks_execution(kernel, term, inputs, "diagonal_conv2d_ckks")


def _assert_direct_kernel_ckks_execution(kernel, term, inputs, benchmark):
    args = get_default_args()
    args.n = 16
    args.benchmark = benchmark
    circuit_ir = Lower(kernel).run()
    results = run_backend("ckks", circuit_ir, inputs, args)

    expected_cts = apply_layout(term.eval(inputs), kernel.layout)
    assert_results_equal(expected_cts, results, "ckks")
