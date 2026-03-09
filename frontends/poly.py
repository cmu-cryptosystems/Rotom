"""
Polynomial approximation helpers for frontend-level activations.

Currently this module provides a Chebyshev-based approximation to ReLU
that can be consumed by the generic `TensorTerm.poly` operator.
"""

from __future__ import annotations

import numpy as np


def build_relu_chebyshev_coeffs(
    degree: int = 21, domain: tuple[float, float] = (-3.0, 3.0)
) -> list[float]:
    """Construct a Chebyshev-based polynomial approximation to ReLU on a domain.

    We approximate ReLU(x) = max(0, x) in least-squares sense on the given
    domain using numpy's Chebyshev utilities, then convert the Chebyshev
    series to a standard monomial basis so it can be evaluated by the generic
    Poly evaluator.
    """
    lower, upper = domain
    # Sample ReLU on a dense grid in [lower, upper].
    xs = np.linspace(lower, upper, 4097)
    ys = np.maximum(xs, 0.0)
    # Fit a Chebyshev series on [lower, upper], letting the Chebyshev helper
    # handle domain scaling. This yields a numerically stable approximation
    # that is reasonably accurate (on the order of a few percent) across the
    # domain for a moderate polynomial degree.
    cheb = np.polynomial.chebyshev.Chebyshev.fit(
        xs, ys, degree, domain=[lower, upper]
    )
    mono = cheb.convert(kind=np.polynomial.Polynomial)
    return [float(c) for c in mono.coef]


APPROX_RELU_CHEBYSHEV_COEFFS = build_relu_chebyshev_coeffs()

