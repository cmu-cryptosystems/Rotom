"""
MLP inference for MNIST (HEIR-style, linear layers only).

Based on https://github.com/google/heir/issues/1232: MLP inference on MNIST
using HEIR CKKS. The HEIR write-up uses:
  - FC1: 1024x1024 (784x512 padded for square matmul)
  - Approx-RELU (polynomial approximation)
  - FC2: 1024x1024 (512x10 padded)

Rotom does not implement ReLU in the tensor frontend for HE, so this benchmark
expresses the linear part only: input @ fc1 @ fc2. This matches the "cleartext
computation" structure from the issue and validates layout assignment and
lowering for two consecutive matrix multiplications (ciphertext @ plaintext).

Default sizes match the HEIR example (1x1024 input, 1024x1024 weights).
"""

import numpy as np

from frontends.tensor import TensorTerm


def mlp_mnist(hidden_size=1024):
    """
    Two-layer MLP (linear only): output = (input @ fc1) @ fc2.

    Args:
        hidden_size: Size of hidden dimension (default 1024 for MNIST-style).
                     Input shape [1, hidden_size], both weights [hidden_size, hidden_size].

    Returns:
        (tensor_ir, inputs) for use with main.py --benchmark mlp_mnist.
    """
    inputs = {}
    inputs["input"] = np.array([[np.random.randn() * 0.1 for _ in range(hidden_size)]])
    inputs["fc1"] = np.array(
        [
            [np.random.randn() * 0.1 for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
    )
    inputs["fc2"] = np.array(
        [
            [np.random.randn() * 0.1 for _ in range(hidden_size)]
            for _ in range(hidden_size)
        ]
    )

    inp = TensorTerm.Tensor("input", [1, hidden_size], True)
    fc1 = TensorTerm.Tensor("fc1", [hidden_size, hidden_size], False)
    fc2 = TensorTerm.Tensor("fc2", [hidden_size, hidden_size], False)
    tensor_ir = (inp @ fc1) @ fc2
    return tensor_ir, inputs
