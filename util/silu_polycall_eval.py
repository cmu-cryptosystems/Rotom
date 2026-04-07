"""Shared SiLU ``PolyCall`` evaluation for :class:`~frontends.tensor_evaluator.TensorEvaluator` and :class:`~backends.toy.Toy`.

PolyCall(``"silu"``, *lo*, *hi*) can be evaluated in two ways:

- **``poly``** (default): least-squares approximation ``silu(x) ≈ x * q(x)`` on
  ``[lo, hi]`` with ``q`` fitted on Chebyshev nodes (same recipe in eval and Toy).
  This matches the polynomial used when exercising the lowered ``POLY_CALL`` path.

- **``exact``**: plaintext SiLU ``x * sigmoid(x)`` with clipped exponent (same as
  PyTorch-style numerics for large |x|).

Tests that compare ``tensor_ir.eval`` to PyTorch should set
``inputs["__rotom_silu_eval_mode"] = "exact"``.  Tests that compare eval to Toy
on lowered SiLU can omit the key (both default to ``poly``) or set
``"poly"`` explicitly.

**Poly fit hyperparameters** (degree, Chebyshev node count) are read from
``SILU_POLY_DEGREE_KEY`` / ``SILU_POLY_NODES_KEY`` in ``inputs``; defaults are
``DEFAULT_SILU_POLY_DEGREE`` and ``DEFAULT_SILU_POLY_NODES``. CIFAR ResNet
builders call :func:`benchmarks.e2e.resnet.resnet20_tensor_ir.populate_resnet20_inputs`,
which sets those keys so Toy and ``tensor_ir.eval`` stay locked to the same fit.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

# Inputs dict key (same as historical Toy/TensorEvaluator convention).
SILU_EVAL_MODE_KEY = "__rotom_silu_eval_mode"

# Default matches FHE-oriented polynomial approximation for PolyCall("silu", ...).
DEFAULT_SILU_EVAL_MODE = "poly"

# Optional ``inputs`` keys; :func:`populate_resnet20_inputs` sets these for ResNet e2e.
SILU_POLY_DEGREE_KEY = "__rotom_silu_poly_degree"
SILU_POLY_NODES_KEY = "__rotom_silu_poly_nodes"
DEFAULT_SILU_POLY_DEGREE = 11
DEFAULT_SILU_POLY_NODES = 80

_COEFF_CACHE: dict[tuple[float, float, int, int], list[float]] = {}


def get_silu_eval_mode(inputs: Mapping[str, Any] | None) -> str:
    """Return ``\"exact\"`` or ``\"poly\"`` (default ``DEFAULT_SILU_EVAL_MODE``)."""
    if not inputs:
        return DEFAULT_SILU_EVAL_MODE
    m = inputs.get(SILU_EVAL_MODE_KEY, DEFAULT_SILU_EVAL_MODE)
    if m == "exact" or m == "poly":
        return m
    return DEFAULT_SILU_EVAL_MODE


def silu_exact(x: np.ndarray) -> np.ndarray:
    """Element-wise SiLU with stable sigmoid (matches Toy / reference tests)."""
    x = np.asarray(x, dtype=np.float64)
    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))))


def chebyshev_nodes(lo: float, hi: float, n: int) -> np.ndarray:
    j = np.arange(n, dtype=np.float64) + 0.5
    t = np.cos(np.pi * j / n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo)


def silu_poly_ascending_coeffs(
    lo: float, hi: float, degree: int, n_nodes: int
) -> list[float]:
    """Ascending coeffs for ``q`` in ``silu(x) ≈ x * q(x)`` (total degree ``degree``).

    Factoring out ``x`` keeps ``f(0)=0`` exactly (unused HE slots stay zero under
    ``apply_layout``).
    """
    key = (float(lo), float(hi), int(degree), int(n_nodes))
    if key in _COEFF_CACHE:
        return _COEFF_CACHE[key]
    if degree < 1:
        raise ValueError("silu poly degree must be >= 1")
    xs = chebyshev_nodes(lo, hi, n_nodes)
    ys = xs * (1.0 / (1.0 + np.exp(-np.clip(xs, -40.0, 40.0))))
    q_deg = degree - 1
    ratio = np.divide(
        ys,
        xs,
        out=np.full_like(ys, 0.5),
        where=np.abs(xs) > 1e-15,
    )
    high_to_low = np.polyfit(xs, ratio, q_deg)
    coeffs = [float(c) for c in high_to_low[::-1]]
    _COEFF_CACHE[key] = coeffs
    return coeffs


def apply_silu_poly_approx(
    x: np.ndarray, lo: float, hi: float, degree: int, n_nodes: int
) -> np.ndarray:
    """Evaluate ``x * q(x)`` with ``x`` clipped to ``[lo, hi]`` (Toy / eval poly path)."""
    q = silu_poly_ascending_coeffs(lo, hi, degree, n_nodes)
    xv = np.asarray(x, dtype=np.float64)
    xv = np.clip(xv, float(lo), float(hi))
    qx = np.zeros_like(xv, dtype=np.float64)
    for i, c in enumerate(q):
        qx = qx + c * (xv**i)
    return xv * qx


def eval_silu_polycall(
    x: np.ndarray, lo: float, hi: float, inputs: Mapping[str, Any] | None
) -> np.ndarray:
    """Dispatch SiLU PolyCall by ``SILU_EVAL_MODE_KEY`` in ``inputs``."""
    if get_silu_eval_mode(inputs) == "exact":
        return silu_exact(x)
    degree = int((inputs or {}).get(SILU_POLY_DEGREE_KEY, DEFAULT_SILU_POLY_DEGREE))
    n_nodes = int((inputs or {}).get(SILU_POLY_NODES_KEY, DEFAULT_SILU_POLY_NODES))
    return apply_silu_poly_approx(x, lo, hi, degree, n_nodes)
