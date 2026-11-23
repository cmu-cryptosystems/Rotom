"""
MLP inference for MNIST (HEIR-style, with ReLU as a PolyCall).

Based on https://github.com/google/heir/issues/1232: MLP inference on MNIST
using HEIR CKKS. The HEIR write-up uses:
  - FC1: 1024x1024 (784x512 padded for square matmul)
  - Approx-RELU (polynomial approximation)
  - FC2: 1024x1024 (512x10 padded)

This benchmark mirrors that structure, with a ``PolyCall("relu", -8, 8)``
between the two linear layers:
  output = relu(input @ fc1) @ fc2

Default sizes match the HEIR example (1x1024 input, 1024x1024 weights).
"""

import numpy as np

from frontends.tensor import TensorTerm


def mlp_mnist_square(hidden_size=1024):
    """
    Two-layer MLP with a ReLU PolyCall:

        hidden = input @ fc1
        hidden_relu = relu(hidden)
        output = hidden_relu @ fc2

    Args:
        hidden_size: Size of hidden dimension (default 1024 for MNIST-style).
                     Input shape [1, hidden_size], both weights [hidden_size, hidden_size].

    Returns:
        (tensor_ir, inputs) for use with main.py --benchmark mlp_mnist_square.
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
    hidden = inp @ fc1
    hidden_relu = hidden.poly_call("relu", -8.0, 8.0)
    tensor_ir = hidden_relu @ fc2
    return tensor_ir, inputs
