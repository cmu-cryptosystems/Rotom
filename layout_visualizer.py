"""
Layout Visualizer

This module provides utilities for visualizing tensor layouts and their effects
on data packing in homomorphic encryption vectors.

Key Functions:
    visualize_layout: Create test tensors and visualize how layouts pack them
    visualize_for_web: Same evaluation as a single run, plus a JSON payload for the web UI
"""

import numpy as np

from util.layout_util import apply_layout, parse_layout


def _pythonize_scalar(x):
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return x.item()
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _visualize_core(layout_str, n=None, tensor_shape=None, secret=False):
    """Parse layout, optionally build test tensor and packed ciphertexts."""
    # `n` can be omitted: `parse_layout` will infer it from the layout string.
    layout = parse_layout(layout_str, n=n, secret=secret)
    if not tensor_shape:
        return layout, None, None, None
    total_elements = 1
    for dim_size in tensor_shape:
        total_elements *= dim_size
    flat_tensor = np.array([i for i in range(total_elements)])
    tensor = flat_tensor.reshape(tensor_shape)
    packed, slot_maps = apply_layout(tensor, layout, return_slot_mapping=True)
    return layout, tensor, packed, slot_maps


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


def _build_ui_payload_from_data(
    layout_str, n, secret, tensor_shape, layout, tensor, packed, slot_maps
):
    tensor_list = tensor.tolist()
    packed_out = []
    for i, ct in enumerate(packed):
        entries = []
        for j, val in enumerate(ct):
            m = slot_maps[i][j]
            entries.append(
                {
                    "value": _pythonize_scalar(val),
                    "kind": m["kind"],
                    "label": m["label"],
                    "coords": m["coords"],
                    "linear": m["linear"],
                    "slot": m["slot"],
                }
            )
        packed_out.append(
            {
                "id": i,
                "slots": [_pythonize_scalar(v) for v in ct],
                "entries": entries,
            }
        )

    flat_vals = []
    if isinstance(tensor_list, list):
        flat_vals.extend(_flatten_numbers(tensor_list))
    else:
        flat_vals.append(float(tensor_list))
    for pv in packed_out:
        flat_vals.extend(float(s) for s in pv["slots"])

    vmin = min(flat_vals) if flat_vals else 0.0
    vmax = max(flat_vals) if flat_vals else 1.0
    if vmin == vmax:
        vmax = vmin + 1.0

    return {
        "layout_str_input": layout_str,
        "layout_str_resolved": layout.layout_str(),
        "layout_order_hint": _infer_layout_order_hint(
            layout.layout_str(), tuple(tensor_shape)
        ),
        "color_grouping": _infer_color_grouping(slot_maps, tuple(tensor_shape)),
        "n": n,
        "secret": secret,
        "tensor_shape": list(tensor_shape),
        "tensor": tensor_list,
        "tensor_ndim": int(tensor.ndim),
        "tensor_index_order": "C",
        "num_ciphertexts": len(packed_out),
        "packed": packed_out,
        "value_range": {"min": vmin, "max": vmax},
    }


def _infer_layout_order_hint(layout_str_resolved, tensor_shape):
    """
    Return a lightweight hint for common 2D layouts:
    - "row-major" for [0:*:*][1:*:*]
    - "column-major" for [1:*:*][0:*:*]
    - "other" for anything else
    """
    if len(tensor_shape) != 2:
        return "other"

    compact = layout_str_resolved.replace(" ", "")
    if "[0:" in compact and "[1:" in compact:
        idx0 = compact.find("[0:")
        idx1 = compact.find("[1:")
        if idx0 != -1 and idx1 != -1:
            return "row-major" if idx0 < idx1 else "column-major"
    return "other"


def _infer_color_grouping(slot_maps, tensor_shape):
    """
    Infer a grouping strategy from apply_layout's actual traversal sequence.

    For 2D tensors:
    - row-major-like traversal -> {"mode": "row"}
    - column-major-like traversal -> {"mode": "column"}
    - rolled diagonal traversal -> {"mode": "diagonal-diff"} or {"mode": "diagonal-sum"}
    """
    if len(tensor_shape) != 2:
        return {"mode": "value"}

    coords_seq = []
    for ct in slot_maps:
        for entry in ct:
            if entry.get("kind") != "tensor":
                continue
            coords = entry.get("coords")
            if not isinstance(coords, list) or len(coords) < 2:
                continue
            coords_seq.append((int(coords[0]), int(coords[1])))

    if len(coords_seq) < 2:
        return {"mode": "row"}

    deltas = []
    prev = coords_seq[0]
    for curr in coords_seq[1:]:
        dr = int(curr[0] - prev[0])
        dc = int(curr[1] - prev[1])
        if dr != 0 or dc != 0:
            deltas.append((dr, dc))
        prev = curr

    if not deltas:
        return {"mode": "row"}

    single_dim_counts = [0, 0]
    multi_dim_deltas = []
    for dr, dc in deltas:
        changed = int(dr != 0) + int(dc != 0)
        if changed == 1:
            if dr != 0:
                single_dim_counts[0] += 1
            else:
                single_dim_counts[1] += 1
        elif changed >= 2:
            multi_dim_deltas.append((dr, dc))

    # If simultaneous 2D movement dominates, treat as diagonal grouping.
    if len(multi_dim_deltas) >= max(single_dim_counts):
        diag_step = next(
            ((dr, dc) for dr, dc in multi_dim_deltas if abs(dr) == 1 and abs(dc) == 1),
            multi_dim_deltas[0],
        )
        return {
            "mode": "diagonal-sum"
            if diag_step[0] * diag_step[1] >= 0
            else "diagonal-diff"
        }

    # Otherwise infer fastest-changing dimension from single-dim transitions.
    fastest_dim = 0 if single_dim_counts[0] >= single_dim_counts[1] else 1
    # If dim 1 changes fastest -> row groups. If dim 0 changes fastest -> column groups.
    return {"mode": "row" if fastest_dim == 1 else "column"}


def visualize_for_web(layout_str, n=None, tensor_shape=None, secret=False):
    """
    Single ``apply_layout`` evaluation: prints to stdout (capture externally) and
    returns a JSON-ready payload, or ``None`` if ``tensor_shape`` is omitted.
    """
    layout, tensor, packed, slot_maps = _visualize_core(
        layout_str, n, tensor_shape, secret
    )
    if tensor is None:
        return None
    _emit_visualize_prints(layout_str, tensor_shape, layout, tensor, packed)
    return _build_ui_payload_from_data(
        layout_str, n, secret, tensor_shape, layout, tensor, packed, slot_maps
    )


def _flatten_numbers(nested):
    """Flatten nested lists to floats for min/max."""
    if isinstance(nested, list):
        out = []
        for x in nested:
            out.extend(_flatten_numbers(x))
        return out
    return [float(nested)]


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
    layout, tensor, packed, _slot_maps = _visualize_core(
        layout_str, n, tensor_shape, secret
    )

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
