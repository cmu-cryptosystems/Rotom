"""
MLP inference for MNIST (HEIR-style, with approx-ReLU via Poly).

Based on https://github.com/google/heir/issues/1232: MLP inference on MNIST
using HEIR CKKS.
  - FC1: 784x512
  - Approx-RELU (polynomial approximation)
  - FC2: 512x10
"""

import numpy as np

from frontends.tensor import TensorTerm


def mlp_mnist_heir():
    """
    Two-layer MLP with polynomial approx-ReLU:

        hidden = input @ fc1
        hidden_relu ≈ Poly_ReLU(hidden)
        output = hidden_relu @ fc2

    Returns:
        (tensor_ir, inputs) for use with main.py --benchmark mlp_mnist.
    """
    inputs = {}
    inputs["input"] = np.array([[np.random.randn() * 0.1 for _ in range(784)]])
    inputs["fc1"] = np.array(
        [[np.random.randn() * 0.1 for _ in range(512)] for _ in range(784)]
    )
    inputs["fc2"] = np.array(
        [[np.random.randn() * 0.1 for _ in range(10)] for _ in range(512)]
    )

    inp = TensorTerm.Tensor("input", [1, 784], True)
    fc1 = TensorTerm.Tensor("fc1", [784, 512], False)
    fc2 = TensorTerm.Tensor("fc2", [512, 10], False)
    hidden = inp @ fc1
    hidden_relu = hidden.poly_call("relu", 20, -20)
    tensor_ir = hidden_relu @ fc2
    return tensor_ir, inputs
