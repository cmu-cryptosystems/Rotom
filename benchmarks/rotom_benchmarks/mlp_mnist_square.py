"""
MLP inference for MNIST (HEIR-style, with approx-ReLU via Poly).

Based on https://github.com/google/heir/issues/1232: MLP inference on MNIST
using HEIR CKKS. The HEIR write-up uses:
  - FC1: 1024x1024 (784x512 padded for square matmul)
  - Approx-RELU (polynomial approximation)
  - FC2: 1024x1024 (512x10 padded)

This benchmark mirrors that structure using Rotom's `Poly` operator to
approximate ReLU between the two linear layers:
  output ≈ Poly_ReLU(input @ fc1) @ fc2

We use a simple degree-4 polynomial approximation to ReLU on [-1, 1]:
  ReLU(x) ≈ 0.5*x + 0.75*x^2 - 0.25*x^4
which can be encoded as Poly coefficients [0.0, 0.5, 0.75, 0.0, -0.25].

Default sizes match the HEIR example (1x1024 input, 1024x1024 weights).
"""

import numpy as np

from frontends.tensor import TensorTerm


def mlp_mnist_square(hidden_size=1024):
    """
    Two-layer MLP with polynomial approx-ReLU:

        hidden = input @ fc1
        hidden_relu ≈ Poly_ReLU(hidden)
        output = hidden_relu @ fc2

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
    # Approximate ReLU using a fixed degree-4 polynomial on [-1, 1]:
    #   ReLU(x) ≈ 0.5*x + 0.75*x^2 - 0.25*x^4
    # Encoded as Poly coefficients [c0, c1, c2, c3, c4].
    relu_poly_coeffs = [0.0, 0.5, 0.75, 0.0, -0.25]
    hidden = inp @ fc1
    hidden_relu = hidden.poly(relu_poly_coeffs)
    tensor_ir = hidden_relu @ fc2
    return tensor_ir, inputs
