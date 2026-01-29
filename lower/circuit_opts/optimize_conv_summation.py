"""
Optimization pass to optimize convolution summation order.

This optimization ensures that filter position CTs are summed first (using only
additions, no rotations) before channel summation (which requires rotations).

Key insight:
- After convolution multiplication, we have multiple CTs (one per filter position)
- These should be summed across filter dimensions (ciphertext dimensions) first
- This reduces the number of CTs before channel summation, minimizing rotations
- Result: Only 3 rotations needed for channel summation instead of 9*3=27
"""

from ir.dim import DimType
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from util.layout_util import get_cts_by_dim


def optimize_conv_summation(layout_cts):
    """
    Optimize convolution summation by summing filter position CTs first.

    This function takes a LayoutCiphertexts object and optimizes the summation
    order by first summing across filter dimensions (None dimensions in ct_dims),
    which correspond to filter positions. This reduces the number of CTs before
    channel summation, minimizing the total number of rotations needed.

    Args:
        layout_cts: LayoutCiphertexts object containing layout and ciphertexts

    Returns:
        Optimized LayoutCiphertexts with filter dimensions summed first
    """
    # Sum all None dimensions (filter dimensions) in ciphertext dimensions
    # These correspond to filter positions and should be summed together
    # This reduces from 9 CTs (one per filter position) to 1 CT using only additions
    filter_ct_dims = [
        dim
        for dim in layout_cts.layout.ct_dims
        if dim.dim is None and dim.dim_type == DimType.FILL
    ]

    for ct_sum_dim in filter_ct_dims:
        if ct_sum_dim not in layout_cts.layout.ct_dims:
            continue

        ct_groups = get_cts_by_dim(layout_cts, ct_sum_dim)

        # Sum within group (just additions, no rotations)
        sum_cts = {}
        for i, group in enumerate(ct_groups):
            base = group[0]
            for j in range(1, len(group)):
                base = base + group[j]
            sum_cts[i] = base

        # Create new layout without the summed dimension
        new_layout = create_layout_without_dims(layout_cts.layout, [ct_sum_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

    return layout_cts
