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


def _separator(width: int = 60, char: str = "=") -> str:
    return char * width


def _truncate_str(s: str, max_len: int) -> str:
    if len(s) <= max_len:
        return s
    return s[: max_len - 3] + "..."


def _format_array(arr: np.ndarray, *, max_elements: int, precision: int) -> str:
    # threshold controls whether NumPy uses ellipses for large arrays.
    return np.array2string(
        arr,
        precision=precision,
        separator=", ",
        threshold=max_elements,
        max_line_width=120,
        suppress_small=False,
    )


def _format_packed_value(
    value, *, max_elements: int, precision: int, max_repr_len: int
):
    # Try to render numpy arrays and numeric lists nicely; otherwise fall back to repr().
    if isinstance(value, np.ndarray):
        return _format_array(value, max_elements=max_elements, precision=precision)
    if isinstance(value, (list, tuple)):
        # Heuristic: only attempt array rendering if it looks numeric.
        sample = value[: min(16, len(value))]
        if all(isinstance(x, (int, float, np.number)) for x in sample):
            return _format_array(
                np.asarray(value), max_elements=max_elements, precision=precision
            )

    return _truncate_str(repr(value), max_repr_len=max_repr_len)


def visualize_layout(
    layout_str,
    n,
    tensor_shape=None,
    secret=False,
    *,
    max_tensor_elements: int = 64,
    max_packed_elements: int = 64,
    max_repr_len: int = 220,
    precision: int = 3,
):
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

        tensor = np.arange(total_elements, dtype=float).reshape(tensor_shape)

        print("\n" + _separator())
        print(f"LAYOUT: {layout_str}")
        print(_separator())

        print(f"HE slots (n): {n}")
        print(f"Secret/plaintext: {secret}")
        print(f"Tensor shape: {tensor_shape} (elements={total_elements})")
        print("\nOriginal tensor:")
        print(
            _format_array(
                np.asarray(tensor),
                max_elements=max_tensor_elements,
                precision=precision,
            )
        )
        print("\nLayout (canonical):")
        print(layout.layout_str())

        # Apply layout and show result
        layout_tensor = apply_layout(tensor, layout)
        print("\nPacked vector(s):")

        if isinstance(layout_tensor, list):
            print(f"  ciphertexts: {len(layout_tensor)}")
            if len(layout_tensor) == 0:
                print("  (empty)")
            else:
                for i, ct in enumerate(layout_tensor):
                    # Keep per-ciphertext output readable even for large packed values.
                    formatted = _format_packed_value(
                        ct,
                        max_elements=max_packed_elements,
                        precision=precision,
                        max_repr_len=max_repr_len,
                    )
                    print(f"  [{i}] {formatted}")
        else:
            formatted = _format_packed_value(
                layout_tensor,
                max_elements=max_packed_elements,
                precision=precision,
                max_repr_len=max_repr_len,
            )
            print(formatted)

        print(_separator())
        print()

        return layout, layout_tensor

    return layout, None


def demo_layout_examples():
    """
    Demonstrate various layout examples with visualization.

    This function shows how different layouts affect the packing of tensor data
    into homomorphic encryption vectors.
    """
    print("Layout Visualization Examples\n" + _separator())

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
    print(f"Comparing Layouts for Tensor Shape {tensor_shape}\n" + _separator())

    results = []
    for i, layout_str in enumerate(layout_strings, 1):
        print(f"\n[{i}/{len(layout_strings)}] {layout_str}")
        layout, packed = visualize_layout(layout_str, n, tensor_shape)
        results.append((layout_str, layout, packed))
        print(_separator(char="-", width=50))

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
