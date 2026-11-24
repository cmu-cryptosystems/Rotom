"""
MoViNet-A0 (stream) benchmarks: random weights on a random clip.

The program and its PyTorch reference are in
``benchmarks/e2e/movinet/movinet_a0.py``.
"""

import numpy as np

from benchmarks.e2e.movinet.movinet_a0 import A0, TINY, init_weights, tensor_ir


def _clip_program(config, frames, size, n):
    clip = np.random.default_rng(0).normal(size=(3, frames, size, size))
    term, inputs = tensor_ir(config, init_weights(config), clip)
    return term, inputs, n


def movinet_tiny():
    """Three MoViNet layers at reduced widths on a 2-frame 16x16 clip.

    Returns:
        (tensor_ir, inputs, n)
    """
    return _clip_program(TINY, frames=2, size=16, n=4096)


def movinet_a0(frames=2, size=32, n=8192):
    """MoViNet-A0 (stream), all 15 layers at the official widths.

    The official clip is 172x172; the default here is a small one.

    Returns:
        (tensor_ir, inputs, n)
    """
    return _clip_program(A0, frames, size, n)
