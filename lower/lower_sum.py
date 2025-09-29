from copy import deepcopy as copy

from ir.dim import DimType
from ir.he import HEOp, HETerm
from lower.lower_util import rotate_and_sum
from util.layout_util import convert_layout_to_mask, get_cts_by_dim, get_segment


def lower_sum(env, kernel):
    kernel = copy(kernel)
    cts = env[kernel.cs[0]]
    ct_dims = kernel.cs[0].layout.ct_dims
    slot_dims = kernel.cs[0].layout.slot_dims
    sum_dims = kernel.cs[1]

    ct_sum_dims = []
    slot_sum_dims = []
    for sum_dim in sum_dims:
        if sum_dim in ct_dims:
            ct_sum_dims.append(sum_dim)
        elif sum_dim in slot_dims:
            slot_sum_dims.append(sum_dim)

    sum_cts = {}
    # sum together ciphertexts
    if ct_sum_dims:
        for ct_sum_dim in ct_sum_dims:
            ct_groups = get_cts_by_dim(cts, ct_dims, ct_sum_dim)
            # sum within group
            sum_cts = {}
            for i, group in enumerate(ct_groups):
                base = group[0]
                for j in range(1, len(group)):
                    base += group[j]
                sum_cts[i] = base

            # update layout and cts
            ct_dims.remove(ct_sum_dim)
            cts = sum_cts

    # sum together slot dimensions per ciphertext
    if slot_sum_dims:
        summed_cts = {}
        for index, term in cts.items():
            for slot_sum_dim in slot_sum_dims:
                segment = get_segment(slot_sum_dim, slot_dims)
                extent = segment[1]
                mul_offset = segment[2]
                if index not in summed_cts:
                    summed_cts[index] = rotate_and_sum(term, extent, mul_offset)
                else:
                    summed_cts[index] = rotate_and_sum(
                        summed_cts[index], extent, mul_offset
                    )
        cts = summed_cts

    # mask out gap dimensions
    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm(HEOp.MASK, [convert_layout_to_mask(kernel.layout)], False, "mask")
        masked_cts = {}
        for index, term in cts.items():
            mask_term = mask * term
            masked_cts[index] = mask_term
        cts = masked_cts
    return cts
