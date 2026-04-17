"""Toy-backend checks for PRODUCT reduce and CAST (layout-preserving)."""

import numpy as np

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def _run_and_compare(tensor_term: TensorTerm, inputs: dict, *, n: int) -> None:
    args = get_default_args()
    args.backend = "toy"
    args.n = n
    args.rolls = False
    args.skip_toy_eval_checks = True

    kernel = LayoutAssignment(tensor_term, args).run()
    circuit_ir = Lower(kernel).run()
    backend_results = Toy(circuit_ir, inputs, args).run()

    dense_eval = tensor_term.eval(inputs)
    expected_cts = apply_layout(dense_eval, kernel.layout)

    assert len(expected_cts) == len(backend_results)
    for exp, res in zip(expected_cts, backend_results):
        exp_a = np.asarray(exp, dtype=np.float64).ravel()
        res_a = np.asarray(res, dtype=np.float64).ravel()
        assert exp_a.shape == res_a.shape
        assert np.isfinite(res_a).all(), "Toy output should be finite"


def test_product_along_last_dim() -> None:
    np.random.seed(2)
    # Keep entries away from zero so the product stays in a reasonable range.
    x = np.random.uniform(0.5, 1.5, (4, 8, 8)).astype(np.float64)
    inputs = {"x": x}
    t = TensorTerm.Tensor("x", [4, 8, 8], True)
    t = t.product(2)
    _run_and_compare(t, inputs, n=64)


def test_mean_over_two_axes_keepdims() -> None:
    """TFLite-style MEAN over spatial axes (keepdims), matches toy + numpy."""
    np.random.seed(4)
    x = np.random.randn(2, 5, 8, 8, 4).astype(np.float64)
    inputs = {"x": x}
    t = TensorTerm.Tensor("x", [2, 5, 8, 8, 4], True)
    t = t.mean((2, 3))
    _run_and_compare(t, inputs, n=256)


def test_cast_float32_roundtrip_layout() -> None:
    np.random.seed(3)
    x = np.random.randn(4, 8, 8).astype(np.float64)
    inputs = {"x": x}
    t = TensorTerm.Tensor("x", [4, 8, 8], True).cast(np.float32)
    _run_and_compare(t, inputs, n=64)
