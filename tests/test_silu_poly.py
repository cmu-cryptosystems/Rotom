"""SiLU-related plaintext tests.

- **PolyCall(\"silu\", ...)**: Rotom's tensor eval and Toy backend use *exact* SiLU
  (clipped sigmoid), not a truncated polynomial. Bounds on `poly_call` are for the
  HE / lowering contract; they are not enforced in plaintext eval.

- **Coefficient polynomial** (`TensorEvaluator._eval_poly` with a list of coeffs):
  Evaluates ``sum_i c_i x^i`` in the **native** coordinate `x` (same ascending
  convention as the ReLU coeffs in `mlp_mnist_square.py`). That approximation is
  only valid on the interval the coefficients were fitted for; tests below fit
  SiLU on ``[lo, hi]`` and check in-range error vs. divergence outside.
"""

import numpy as np
import pytest

from backends.toy import Toy
from frontends.tensor import TensorTerm
from frontends.tensor_evaluator import TensorEvaluator


def _reference_silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (8,),
        (4, 4),  # dims must be powers of 2: eval pads each axis to ceil_pow2
        (2, 4, 4),
    ],
)
def test_tensor_eval_poly_call_silu_matches_reference(shape):
    rng = np.random.RandomState(42)
    x = rng.standard_normal(shape).astype(np.float64) * 2.0
    name = "x"
    term = TensorTerm.Tensor(name, list(x.shape), True).poly_call(
        "silu", lower_bound=-8.0, upper_bound=8.0
    )
    out = term.eval({name: x})
    expected = _reference_silu(x)
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    "value",
    [-50.0, -40.0, -1.0, 0.0, 1.0, 40.0, 50.0],
)
def test_tensor_eval_silu_edge_scalars(value):
    name = "x"
    x = np.array(value, dtype=np.float64)
    term = TensorTerm.Tensor(name, [], True).poly_call(
        "silu", lower_bound=-10.0, upper_bound=10.0
    )
    out = term.eval({name: x})
    expected = _reference_silu(x)
    np.testing.assert_allclose(out, expected, rtol=1e-14, atol=1e-14)


def test_toy_poly_silu_matches_reference():
    # _apply_poly_to_vector only needs inputs for batchnorm; use bare instance.
    toy = object.__new__(Toy)
    toy.inputs = {}
    vec = np.array([-3.0, 0.0, 2.5, -40.0, 40.0], dtype=np.float64)
    got = toy._apply_poly_to_vector(vec, "silu")
    np.testing.assert_allclose(got, _reference_silu(vec), rtol=1e-14, atol=1e-14)


def _chebyshev_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    j = np.arange(n, dtype=np.float64) + 0.5
    t = np.cos(np.pi * j / n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo)


def _silu_poly_ascending_coeffs(
    lo: float, hi: float, degree: int, n_nodes: int
) -> list:
    """Least-squares polynomial for SiLU on [lo, hi]; coeffs for c0 + c1*x + ... + c_d*x^d."""
    xs = _chebyshev_nodes(lo, hi, n_nodes)
    ys = _reference_silu(xs)
    high_to_low = np.polyfit(xs, ys, degree)
    return [float(c) for c in high_to_low[::-1]]


def test_silu_coefficient_poly_small_error_inside_fitted_range():
    """Fitted poly should track SiLU on [lo, hi] where it was trained."""
    lo, hi = -4.0, 4.0
    degree = 11
    coeffs = _silu_poly_ascending_coeffs(lo, hi, degree, n_nodes=80)
    ev = TensorEvaluator()
    xv = np.linspace(lo, hi, 501)
    approx = ev._eval_poly(xv, coeffs, {})
    true = _reference_silu(xv)
    max_err = float(np.max(np.abs(approx - true)))
    assert max_err < 1e-3, f"max abs err inside [{lo}, {hi}] = {max_err}"


def test_silu_coefficient_poly_assumption_inputs_in_range():
    """Outside the fitted interval the same coeffs are unreliable vs. true SiLU."""
    lo, hi = -3.0, 3.0
    degree = 9
    coeffs = _silu_poly_ascending_coeffs(lo, hi, degree, n_nodes=64)
    ev = TensorEvaluator()

    xv = np.linspace(lo, hi, 401)
    err_in = np.abs(ev._eval_poly(xv, coeffs, {}) - _reference_silu(xv))
    max_in = float(np.max(err_in))

    x_out = np.array([-8.0, -6.0, 6.0, 8.0], dtype=np.float64)
    err_out = np.abs(ev._eval_poly(x_out, coeffs, {}) - _reference_silu(x_out))
    max_out = float(np.max(err_out))

    assert max_in < 1e-3
    assert max_out > 25 * max_in, (
        "expected much larger error outside fitted range; "
        f"max_in={max_in}, max_out={max_out}"
    )
