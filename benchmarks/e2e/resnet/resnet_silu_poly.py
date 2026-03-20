"""
ResNet-style SiLU activation for benchmarks (same polynomial as ``frontends.silu_poly``).
"""

from frontends.silu_poly import CALCULATION_COEF, calculation, silu_approx


def silu(x):
    return silu_approx(x)


__all__ = ["CALCULATION_COEF", "calculation", "silu", "silu_approx"]
