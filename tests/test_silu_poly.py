"""SiLU-related plaintext tests.

- **PolyCall(\"silu\", ...)**: Rotom's tensor eval and Toy backend use *exact* SiLU
  (clipped sigmoid), not a truncated polynomial. Bounds on `poly_call` are for the
  HE / lowering contract; they are not enforced in plaintext eval.
  ``test_tensor_eval_silu_polycall_matches_toy_no_drift`` checks that the two
  paths agree to float64 tolerance (guards silent divergence if one side changes).

- **Coefficient polynomial** (`TensorEvaluator._eval_poly` with a list of coeffs):
  Evaluates ``sum_i c_i x^i`` in the **native** coordinate `x` (same ascending
  convention as the ReLU coeffs in `mlp_mnist_square.py`). That approximation is
  only valid on the interval the coefficients were fitted for; tests below fit
  SiLU on ``[lo, hi]`` and check in-range error vs. divergence outside.
  ``test_poly_silu_composition_max_error_vs_depth`` chains that poly ``N`` times
  against ``N`` exact SiLUs to see how approximation error behaves with depth.
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


@pytest.mark.parametrize(
    "shape,lower_bound,upper_bound",
    [
        ((), -8.0, 8.0),
        ((8,), -20.0, 20.0),
        ((4, 4), -8.0, 8.0),
        ((2, 4, 4), -10.0, 15.0),
    ],
)
def test_tensor_eval_silu_polycall_matches_toy_no_drift(
    shape, lower_bound, upper_bound
):
    """``TensorTerm.eval`` and Toy's element-wise SiLU should match (no implementation drift)."""
    rng = np.random.RandomState(7)
    x = rng.standard_normal(shape).astype(np.float64) * 2.5
    name = "x"
    term = TensorTerm.Tensor(name, list(shape), True).poly_call(
        "silu", lower_bound=lower_bound, upper_bound=upper_bound
    )
    eval_out = np.asarray(term.eval({name: x}))

    toy = object.__new__(Toy)
    toy.inputs = {}
    toy_out = toy._apply_poly_to_vector(x.ravel(), "silu").reshape(shape)

    max_abs_drift = float(np.max(np.abs(eval_out - toy_out)))
    assert (
        max_abs_drift < 1e-14
    ), f"SiLU drift tensor.eval vs toy: max_abs={max_abs_drift}"
    np.testing.assert_allclose(eval_out, toy_out, rtol=1e-14, atol=1e-14)


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


@pytest.mark.parametrize("n_layers", [1, 2, 5, 10, 20, 50])
def test_poly_silu_composition_max_error_vs_depth(n_layers: int) -> None:
    """Chained poly-SiLU vs chained exact SiLU: max |Δ| over a fixed random batch.

    Uses one least-squares fit on ``[lo, hi]`` (same as other tests here). Inputs
    are drawn strictly inside that interval so each step stays in-range; even then,
    depth can grow the gap vs exact SiLU. For this toy setup the max error quickly
    plateaus (SiLU maps activations into a region where the poly tracks well).
    """
    lo, hi = -4.0, 4.0
    degree = 11
    n_nodes = 80
    coeffs = _silu_poly_ascending_coeffs(lo, hi, degree, n_nodes=n_nodes)
    ev = TensorEvaluator()

    rng = np.random.RandomState(0)
    x = rng.uniform(lo * 0.5, hi * 0.5, size=2048).astype(np.float64)

    y_poly = x.copy()
    y_exact = x.copy()
    for _ in range(n_layers):
        y_poly = ev._eval_poly(y_poly, coeffs, {})
        y_exact = _reference_silu(y_exact)

    max_abs_err = float(np.max(np.abs(y_poly - y_exact)))
    assert np.all(np.isfinite(y_poly)), "poly composition produced non-finite values"

    # Empirically ~1.13e-3 asymptote for this seed / interval / fit; leave headroom.
    assert (
        max_abs_err < 1.3e-3
    ), f"n_layers={n_layers}: max_abs_err={max_abs_err} vs exact SiLU chain"


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
