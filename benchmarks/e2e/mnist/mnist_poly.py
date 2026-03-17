"""
MLP inference for MNIST (HEIR-style, with approx-ReLU via Poly).

Based on https://github.com/google/heir/issues/1232: MLP inference on MNIST
using HEIR CKKS.
  - FC1: 784x512
  - Approx-RELU (polynomial approximation)
  - FC2: 512x10
"""

from frontends.tensor import TensorTerm
from tests.mnist.test_mlp_mnist import (
    MODEL_FILE,
    _build_rotom_mnist_ir,
    _extract_traced_mnist_linears,
    _load_mnist_test_set,
)


def relu(x):
    x2 = x * x  # x^2,  depth 1
    x4 = x2 * x2  # x^4,  depth 2
    x6 = x4 * x2  # x^6,  depth 3
    x8 = x4 * x4  # x^8,  depth 3
    x10 = x8 * x2  # x^10, depth 4
    x12 = x8 * x4  # x^12, depth 4
    return (
        TensorTerm.const(4.113641024556607e-01)
        + TensorTerm.const(5.000000000000002e-01) * x
        + TensorTerm.const(1.223805757222573e-01) * x2
        - TensorTerm.const(1.937688573683916e-03) * x4
        + TensorTerm.const(2.034568933371524e-05) * x6
        - TensorTerm.const(1.193749792878656e-07) * x8
        + TensorTerm.const(3.868625851050720e-10) * x10
        - TensorTerm.const(6.474340232242984e-13) * x12
    )


def mnist_poly(idx):
    # Load a single MNIST test sample.
    images, labels = _load_mnist_test_set()
    assert images.shape[0] == labels.shape[0] and images.shape[0] > 0
    x = images[idx : idx + 1]  # [1, 1, 28, 28]
    y = int(labels[idx].item())

    # Flatten to [1, 784] to match the traced model's expected input.
    x_flat = x.view(1, -1).numpy()

    # Extract trained weights/biases from the TorchScript model.
    params = _extract_traced_mnist_linears(MODEL_FILE)
    in_dim = params["in_dim"]
    hidden_dim = params["hidden_dim"]
    out_dim = params["out_dim"]
    tensor_ir = _build_rotom_mnist_ir(in_dim, hidden_dim, out_dim)

    # Convert from PyTorch's [out_features, in_features] convention to the
    # Tensor frontend's [in_dim, hidden_dim] / [hidden_dim, out_dim].
    fc1_w = params["fc1_w"].T  # [in_dim, hidden_dim]
    fc1_b = params["fc1_b"].reshape(1, hidden_dim)
    fc2_w = params["fc2_w"].T  # [hidden_dim, out_dim]
    fc2_b = params["fc2_b"].reshape(1, out_dim)

    inputs = {
        "input": x_flat,
        "fc1": fc1_w,
        "b1": fc1_b,
        "fc2": fc2_w,
        "b2": fc2_b,
    }

    inp = TensorTerm.Tensor("input", [1, in_dim], True)
    fc1 = TensorTerm.Tensor("fc1", [in_dim, hidden_dim], False)
    b1 = TensorTerm.Tensor("b1", [1, hidden_dim], False)
    fc2 = TensorTerm.Tensor("fc2", [hidden_dim, out_dim], False)
    b2 = TensorTerm.Tensor("b2", [1, out_dim], False)

    hidden = inp @ fc1 + b1
    hidden_relu = relu(hidden)
    tensor_ir = hidden_relu @ fc2 + b2
    return tensor_ir, inputs, y
