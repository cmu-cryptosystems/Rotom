"""
Layout Visualizer

This module provides utilities for visualizing tensor layouts and their effects
on data packing in homomorphic encryption vectors.

Key Functions:
    visualize_layout: Create test tensors and visualize how layouts pack them
"""

import numpy as np

from ir.layout import Layout
from util.layout_util import apply_layout


def visualize_layout(layout_str, n, tensor_shape=None, secret=False):
    """
    Visualize a layout by creating a test tensor and showing the packed result.

    Args:
        layout_str: String representation of the layout (e.g., "roll(0,1) [1:4:1][0:4:1]")
        n: Number of slots in the HE vector
        tensor_shape: Shape of test tensor to create (optional)
        secret: Whether this is ciphertext (True) or plaintext (False)

    Returns:
        tuple: (layout object, packed tensor data)

    Examples:
        >>> layout, packed = visualize_layout("[0:4:1][1:4:1]", 16, (4, 4))
        >>> layout, packed = visualize_layout("roll(0,1) [1:4:1][0:4:1]", 16, (4, 4))
        >>> layout, packed = visualize_layout("[R:4:1];[0:4:1][1:4:1]", 16, (4, 4))
    """
    # Create layout from string using the Layout.from_string method
    layout = Layout.from_string(layout_str, n, secret)

    # Create test tensor if shape provided
    if tensor_shape:
        # Create a tensor with sequential values
        total_elements = 1
        for dim_size in tensor_shape:
            total_elements *= dim_size

        # Create flat array and reshape
        flat_tensor = np.array([i for i in range(total_elements)])
        tensor = flat_tensor.reshape(tensor_shape)

        print(f"=== Layout: {layout_str} ===")
        print(f"Test tensor shape: {tensor_shape}")
        print("Original tensor:")
        print(tensor)
        print(f"Layout: {layout.layout_str()}")

        # Apply layout and show result
        layout_tensor = apply_layout(tensor, layout)
        print("Packed vector:")
        if isinstance(layout_tensor, list) and len(layout_tensor) > 1:
            for i, l in enumerate(layout_tensor):
                print(f"Ciphertext {i}: {l}")
        else:
            print(layout_tensor)
        print()

        return layout, layout_tensor

    return layout, None


def demo_layout_examples():
    """
    Demonstrate various layout examples with visualization.

    This function shows how different layouts affect the packing of tensor data
    into homomorphic encryption vectors.
    """
    print("=== Layout Visualization Examples ===\n")

    # Example 1: Row-major layout
    print("1. Row-major layout (standard matrix storage):")
    visualize_layout("[0:4:1][1:4:1]", 16, (4, 4))

    # Example 2: Column-major layout
    print("2. Column-major layout:")
    visualize_layout("[1:4:1][0:4:1]", 16, (4, 4))

    # Example 3: Layout with roll operation
    print("3. Layout with roll operation:")
    visualize_layout("roll(0,1) [0:4:1][1:4:1]", 16, (4, 4))

    # Example 4: Ciphertext distribution
    print("4. Ciphertext distribution:")
    visualize_layout("[R:4:1];[0:4:1][1:4:1]", 16, (4, 4))

    # Example 5: 3D tensor layout
    print("5. 3D tensor layout:")
    visualize_layout("[0:2:1][1:4:1][2:4:1]", 32, (2, 4, 4))

    # Example 6: Complex layout with multiple rolls
    print("6. Complex layout with multiple rolls:")
    visualize_layout(
        "roll(0,1) roll(2,3) [0:2:1][1:2:1][2:4:1][3:4:1]", 32, (2, 2, 4, 4)
    )


def compare_layouts(layout_strings, n, tensor_shape):
    """
    Compare multiple layouts side by side.

    Args:
        layout_strings: List of layout strings to compare
        n: Number of slots in the HE vector
        tensor_shape: Shape of test tensor to create

    Returns:
        list: List of (layout_str, layout_obj, packed_data) tuples
    """
    print(f"=== Comparing Layouts for Tensor Shape {tensor_shape} ===\n")

    results = []
    for i, layout_str in enumerate(layout_strings, 1):
        print(f"Layout {i}: {layout_str}")
        layout, packed = visualize_layout(layout_str, n, tensor_shape)
        results.append((layout_str, layout, packed))
        print("-" * 50)

    return results


if __name__ == "__main__":
    # Run demonstration examples
    demo_layout_examples()

    print("\n" + "=" * 60)
    print("COMPARISON EXAMPLE")
    print("=" * 60)

    # Compare different layouts for the same tensor
    layouts_to_compare = [
        "[0:4:1][1:4:1]",  # Row-major
        "[1:4:1][0:4:1]",  # Column-major
        "roll(0,1) [0:4:1][1:4:1]",  # Rolled
        "[R:4:1];[0:4:1][1:4:1]",  # With ciphertext distribution
    ]

    compare_layouts(layouts_to_compare, 16, (4, 4))
