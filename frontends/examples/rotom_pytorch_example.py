"""
Rotom PyTorch Frontend Example

This example demonstrates how to use the Rotom PyTorch frontend to build
neural network computations with homomorphic encryption support.
"""

import numpy as np

from frontends.rotom_pytorch import torch


def basic_tensor_operations():
    """Demonstrate basic tensor operations."""
    print("=== Basic Tensor Operations ===")

    # Create tensors (similar to PyTorch)
    a = torch.tensor([[1, 2], [3, 4]], requires_grad=True)  # Ciphertext
    b = torch.tensor([[5, 6], [7, 8]], requires_grad=False)  # Plaintext

    print(f"Tensor a: {a}")
    print(f"Tensor b: {b}")

    # Arithmetic operations
    c = a + b
    print(f"a + b: {c}")

    d = a * b
    print(f"a * b: {d}")

    e = torch.matmul(a, b)
    print(f"matmul(a, b): {e}")

    return a, b, c, d, e


def shape_operations():
    """Demonstrate shape manipulation operations."""
    print("\n=== Shape Operations ===")

    # Create a tensor
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    print(f"Original tensor: {x}")
    print(f"Shape: {x.shape}")

    # Transpose
    x_t = x.transpose()
    print(f"Transpose: {x_t}")

    # Reshape
    x_reshaped = x.reshape(3, 2)
    print(f"Reshaped to (3, 2): {x_reshaped}")

    # Sum operations
    sum_dim0 = x.sum(0)
    print(f"Sum along dim 0: {sum_dim0}")

    sum_dim1 = x.sum(1)
    print(f"Sum along dim 1: {sum_dim1}")

    sum_all = x.sum()
    print(f"Sum all elements: {sum_all}")

    return x, x_t, x_reshaped


def layout_specification():
    """Demonstrate layout specification for HE optimization."""
    print("\n=== Layout Specification ===")

    # Create tensors with specific layouts
    a = torch.tensor([[1, 2], [3, 4]], requires_grad=True, layout="[0:2:1][1:2:1]")
    b = torch.tensor([[5, 6], [7, 8]], requires_grad=False, layout="[1:2:1][0:2:1]")

    print(f"Tensor a with row-major layout: {a}")
    print(f"Tensor b with column-major layout: {b}")

    # Operations with layout specification
    c = torch.matmul(a, b, layout="[0:2:1][1:2:1]")
    print(f"Matrix multiplication with specific output layout: {c}")

    # Sum with layout
    d = a.sum(0, layout="[0:1:1]")
    print(f"Sum with layout: {d}")

    return a, b, c, d


def neural_network_example():
    """Demonstrate a simple neural network computation."""
    print("\n=== Neural Network Example ===")

    # Input data (batch_size=2, features=3)
    x = torch.tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    print(f"Input x: {x}")

    # Weight matrix (features=3, hidden=4)
    W1 = torch.tensor(
        [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1.0, 1.1, 1.2]],
        requires_grad=True,
    )
    print(f"Weight matrix W1: {W1}")

    # Bias vector (hidden=4)
    b1 = torch.tensor([0.1, 0.2, 0.3, 0.4], requires_grad=True)
    print(f"Bias b1: {b1}")

    # Linear layer 1: x @ W1 + b1
    linear1 = torch.matmul(x, W1) + b1
    print(f"Linear layer 1 output: {linear1}")

    # Activation: ReLU (using polynomial approximation)
    activated1 = torch.relu(linear1)
    print(f"After ReLU activation: {activated1}")

    # Second weight matrix (hidden=4, output=2)
    W2 = torch.tensor(
        [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0], [1.1, 1.2]], requires_grad=True
    )
    b2 = torch.tensor([0.2, 0.3], requires_grad=True)

    # Linear layer 2
    linear2 = torch.matmul(activated1, W2) + b2
    print(f"Final output: {linear2}")

    # Final activation
    output = torch.sigmoid(linear2)
    print(f"Final sigmoid output: {output}")

    return x, W1, b1, W2, b2, output


def convolution_example():
    """Demonstrate 2D convolution operation."""
    print("\n=== Convolution Example ===")

    # Input tensor (batch=1, channels=1, height=4, width=4)
    input_tensor = torch.tensor(
        [[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]]],
        requires_grad=True,
    )
    print(f"Input tensor shape: {input_tensor.shape}")

    # Filter tensor (out_channels=1, in_channels=1, height=3, width=3)
    filter_tensor = torch.tensor(
        [[[[1, 0, -1], [2, 0, -2], [1, 0, -1]]]], requires_grad=True
    )
    print(f"Filter tensor shape: {filter_tensor.shape}")

    # Convolution
    output = torch.conv2d(input_tensor, filter_tensor, stride=1, padding="valid")
    print(f"Convolution output shape: {output.shape}")
    print(f"Convolution output: {output}")

    return input_tensor, filter_tensor, output


def evaluation_example():
    """Demonstrate tensor evaluation with actual values."""
    print("\n=== Evaluation Example ===")

    # Create a simple computation
    a = torch.tensor([[1, 2], [3, 4]], requires_grad=True)
    b = torch.tensor([[5, 6], [7, 8]], requires_grad=False)

    # Complex computation
    c = (a + b) * a
    d = torch.sum(c)

    print(f"Computation: sum((a + b) * a)")
    print(f"Where a = {a.data}")
    print(f"  and b = {b.data}")

    # Evaluate the computation
    inputs = {a.name: a.data, b.name: b.data}
    result = d.eval(inputs)

    print(f"Result: {result}")

    # Verify with numpy
    expected = np.sum((a.data + b.data) * a.data)
    print(f"Expected (numpy): {expected}")
    print(f"Match: {np.allclose(result, expected)}")


def layout_optimization_example():
    """Demonstrate layout optimization for HE efficiency."""
    print("\n=== Layout Optimization Example ===")

    # Create matrices with different layouts for optimal HE operations
    # Row-major layout for matrix A
    A = torch.tensor([[1, 2], [3, 4]], requires_grad=True, layout="[0:2:1][1:2:1]")

    # Column-major layout for matrix B (better for certain HE operations)
    B = torch.tensor([[5, 6], [7, 8]], requires_grad=False, layout="[1:2:1][0:2:1]")

    # Matrix multiplication with optimized output layout
    C = torch.matmul(A, B, layout="[0:2:1][1:2:1]")

    print(f"Matrix A (row-major): {A}")
    print(f"Matrix B (column-major): {B}")
    print(f"Result C (row-major output): {C}")

    # Demonstrate roll operations in layout strings
    D = torch.tensor(
        [[1, 2, 3, 4]], requires_grad=True, layout="roll(0,1) [1:4:1][0:1:1]"
    )
    print(f"Tensor D with roll layout: {D}")

    return A, B, C, D


def main():
    """Run all examples."""
    print("Rotom PyTorch Frontend Examples")
    print("=" * 50)

    # Run examples
    basic_tensor_operations()
    shape_operations()
    layout_specification()
    neural_network_example()
    convolution_example()
    evaluation_example()
    layout_optimization_example()

    print("\n" + "=" * 50)
    print("Examples completed successfully!")
    print("\nKey Features Demonstrated:")
    print("- PyTorch-like tensor creation and operations")
    print("- Layout specification for HE optimization")
    print("- Neural network computations")
    print("- 2D convolution operations")
    print("- Tensor evaluation with actual values")
    print("- Layout optimization strategies")


if __name__ == "__main__":
    main()
