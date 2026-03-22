"""
Homomorphic-style SiLU approximation (odd polynomial × x).

  HE_SiLU(x) = x * (calculation(x) + 0.5)

SiLU is x * sigmoid(x). The odd polynomial ``calculation`` approximates
sigmoid(x) - 0.5 on a bounded domain; only odd powers x, x^3, …, x^95 appear.

  HE_SiLU(x) = x * (0.5 + calculation(x)) ≈ x * sigmoid(x)
"""

from __future__ import annotations

import numpy as np

# Guard rail for the high-degree SiLU polynomial.
# The fitted polynomial is numerically unstable for large |x|, so clip first.
SILU_APPROX_CLIP_ABS = 3.0

# Odd polynomial, degree 95 (nonzero coefficients at indices 1, 3, …, 95).
CALCULATION_COEF = np.array(
    [
        0.0,
        7.996710431764373794,
        0.0,
        -675.3369309867523498,
        0.0,
        64899.63259606801876,
        0.0,
        -5541863.243898005225,
        0.0,
        388624530.7195351720,
        0.0,
        -21608048515.00339890,
        0.0,
        946201888033.5170898,
        0.0,
        -32874354420731.14062,
        0.0,
        917904633513872.2500,
        0.0,
        -20887030138692156.00,
        0.0,
        392565229968949760.0,
        0.0,
        -6168668708088421376.0,
        0.0,
        81924194928832151550.0,
        0.0,
        -928326340879375794200.0,
        0.0,
        9.050348341649228169e21,
        0.0,
        -7.646354195756242436e22,
        0.0,
        5.633932837045944545e23,
        0.0,
        -3.640197394417100977e24,
        0.0,
        2.072374353246384543e25,
        0.0,
        -1.043846647792529524e26,
        0.0,
        4.668483874066138716e26,
        0.0,
        -1.859538001263114637e27,
        0.0,
        6.613579024094195375e27,
        0.0,
        -2.104700769475414176e28,
        0.0,
        6.003513230087685687e28,
        0.0,
        -1.536907037645646570e29,
        0.0,
        3.534384365301353460e29,
        0.0,
        -7.305228057596438300e29,
        0.0,
        1.357276312861418564e30,
        0.0,
        -2.266227300947486863e30,
        0.0,
        3.398144488542107102e30,
        0.0,
        -4.570757645154817978e30,
        0.0,
        5.505922490531496169e30,
        0.0,
        -5.926675879089307926e30,
        0.0,
        5.684538233293153623e30,
        0.0,
        -4.840805008574164386e30,
        0.0,
        3.643509877742666477e30,
        0.0,
        -2.410238330504335108e30,
        0.0,
        1.391507836238266000e30,
        0.0,
        -6.949516905159151207e29,
        0.0,
        2.968704174558214159e29,
        0.0,
        -1.068938899589015000e29,
        0.0,
        3.181294273579338889e28,
        0.0,
        -7.615697335562942964e27,
        0.0,
        1.409112193760684038e27,
        0.0,
        -1.890896638417253165e26,
        0.0,
        1.637159740395940573e25,
        0.0,
        -6.864790934219505900e23,
    ]
)


def calculation(x):
    """Odd polynomial (degree 95) used inside HE_SiLU."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    x_pow = x.copy()
    for k in range(1, 96, 2):
        out = out + CALCULATION_COEF[k] * x_pow
        if k < 95:
            x_pow = x_pow * x * x
    return out


def silu_approx(x):
    """Polynomial approximation to SiLU(x) = x * sigmoid(x)."""
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -SILU_APPROX_CLIP_ABS, SILU_APPROX_CLIP_ABS)
    return x * (0.5 + calculation(x))
