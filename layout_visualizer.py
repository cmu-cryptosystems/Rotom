"""
Layout Visualizer

Utilities for visualizing tensor layouts and their effects on data packing in
homomorphic encryption vectors.

Key Functions:
    visualize_layout: Create test tensors and visualize how layouts pack them
"""

import numpy as np

from util.layout_util import apply_layout, parse_layout


def _visualize_core(layout_str, n=None, tensor_shape=None, secret=False):
    """Parse layout, optionally build test tensor and packed ciphertexts."""
    # `n` can be omitted: `parse_layout` will infer it from the layout string.
    layout = parse_layout(layout_str, n=n, secret=secret)
    if not tensor_shape:
        return layout, None, None
    total_elements = 1
    for dim_size in tensor_shape:
        total_elements *= dim_size
    flat_tensor = np.array([i for i in range(total_elements)])
    tensor = flat_tensor.reshape(tensor_shape)
    packed = apply_layout(tensor, layout)
    return layout, tensor, packed


def _emit_visualize_prints(layout_str, tensor_shape, layout, tensor, packed):
    print(f"=== Layout: {layout_str} ===")
    print(f"Test tensor shape: {tensor_shape}")
    print("Original tensor:")
    print(tensor)
    print(f"Layout: {layout.layout_str()}")
    print("Packed vector:")
    if isinstance(packed, list) and len(packed) > 1:
        for i, ct in enumerate(packed):
            print(f"Ciphertext {i}: {ct}")
    else:
        print(packed)
    print()


def visualize_layout(layout_str, n=None, tensor_shape=None, secret=False):
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
    layout, tensor, packed = _visualize_core(layout_str, n, tensor_shape, secret)

    if tensor_shape and tensor is not None:
        _emit_visualize_prints(layout_str, tensor_shape, layout, tensor, packed)
        return layout, packed

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
