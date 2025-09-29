"""
Strassen's algorithm optimization for matrix multiplication.

This module implements Strassen's algorithm optimization for matrix multiplication
in homomorphic encryption. Strassen's algorithm reduces the number of
multiplications required for matrix multiplication from O(n³) to approximately
O(n^2.807), providing significant computational savings for large matrices.

Key Concepts:
- Strassen's Algorithm: Divide-and-conquer approach to matrix multiplication
- Tiled Layouts: Requires specific tiled layout patterns for operands
- Multiplication Reduction: Reduces the number of expensive multiplications
- Recursive Decomposition: Breaks large matrices into smaller subproblems
"""

from ir.kernel import KernelOp
from util.kernel_util import get_cs_op_kernels


def check_tiled_layout(kernel, sequence):
    """
    Check that the kernel layout matches the required tiled layout pattern.

    For Strassen's algorithm, we need specific tiled layout patterns:
    - cs[0]: row-major and tiled
    - cs[1]: col-major and tiled

    Args:
        kernel: Kernel to check for tiled layout
        sequence: Expected dimension sequence for the tiled layout

    Returns:
        Boolean indicating if the layout matches the tiled pattern
    """
    ct_dims = kernel.layout.ct_dims
    slot_dims = kernel.layout.slot_dims
    return (
        [ct_dim.dim for ct_dim in ct_dims] == sequence
        and [slot_dim.dim for slot_dim in slot_dims] == sequence
        and all(
            [
                a > b
                for a, b in zip(
                    [ct_dim.stride for ct_dim in ct_dims],
                    [slot_dim.stride for slot_dim in slot_dims],
                )
            ]
        )
    )


def check_strassen_tiles(kernel):
    """
    Check if Strassen's algorithm can be applied to the kernel.

    This function verifies that both operands have the required tiled
    layout patterns for Strassen's algorithm to be applicable.

    Args:
        kernel: Kernel to check for Strassen applicability

    Returns:
        Boolean indicating if Strassen's algorithm can be applied
    """
    # checks to see that strassen's can be run
    cs_kernels = get_cs_op_kernels(kernel)
    cs_kernels = sorted(cs_kernels, key=lambda x: x.cs[0])
    return check_tiled_layout(cs_kernels[0], [0, 1]) and check_tiled_layout(
        cs_kernels[1], [1, 0]
    )


def run_strassens(candidate):
    """
    Apply Strassen's algorithm optimization to matrix multiplication kernels.

    This function applies Strassen's algorithm to reduce the number of
    multiplications required for matrix multiplication operations.
    It requires both operands to be secret and have compatible tiled layouts.

    Args:
        candidate: Kernel to apply Strassen optimization to

    Returns:
        Updated kernel with Strassen optimization applied where applicable
    """
    update_map = {}
    for kernel in candidate.post_order():
        # update kernel cs
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                kernel.cs[i] = update_map[cs]

        # update kernel with rewrites
        match kernel.op:
            case KernelOp.MATMUL:
                # check that both operands are secret and
                # check that the tile shapes match
                if (
                    kernel.cs[0].layout.secret
                    and kernel.cs[0].layout.secret
                    and check_strassen_tiles(kernel)
                ):
                    kernel.op = KernelOp.STRASSEN_MATMUL
                update_map[kernel] = kernel
            case _:
                update_map[kernel] = kernel
    return candidate
