from copy import deepcopy as copy

from ir.dim import DimType
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import rotate_and_sum
from util.layout_util import convert_layout_to_mask, get_cts_by_dim, get_segment


def lower_sum(env, kernel):
    kernel = copy(kernel)
    input_cts = env[kernel.cs[0]]  # LayoutCiphertexts
    layout_cts = input_cts
    sum_dims = kernel.cs[1]

    # Categorize sum dimensions
    ct_sum_dims = []
    slot_sum_dims = []
    for sum_dim in sum_dims:
        if sum_dim in layout_cts.layout.ct_dims:
            ct_sum_dims.append(sum_dim)
        elif sum_dim in layout_cts.layout.slot_dims:
            slot_sum_dims.append(sum_dim)

    # Sum together ciphertext dimensions
    for ct_sum_dim in ct_sum_dims:
        ct_groups = get_cts_by_dim(layout_cts, ct_sum_dim)
        # Sum within group
        sum_cts = {}
        for i, group in enumerate(ct_groups):
            base = group[0]
            for j in range(1, len(group)):
                base += group[j]
            sum_cts[i] = base

        # Create new layout without the summed dimension
        new_layout = create_layout_without_dims(layout_cts.layout, [ct_sum_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

    # Sum together slot dimensions per ciphertext
    for slot_sum_dim in slot_sum_dims:
        segment = get_segment(slot_sum_dim, layout_cts.layout.slot_dims)
        extent = segment[1]
        mul_offset = segment[2]
        summed_cts = {}
        for index, term in layout_cts.cts.items():
            summed_cts[index] = rotate_and_sum(term, extent, mul_offset)

        # Create new layout without the summed dimension
        new_layout = create_layout_without_dims(layout_cts.layout, [slot_sum_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=summed_cts)

    # Mask out gap dimensions
    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm(HEOp.MASK, [convert_layout_to_mask(kernel.layout)], False, "mask")
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            mask_term = mask * term
            masked_cts[index] = mask_term
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=masked_cts)
    else:
        # Update layout to output layout if no mask needed
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)

    return layout_cts
