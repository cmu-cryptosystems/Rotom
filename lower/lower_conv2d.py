import math

import numpy as np

from ir.analysis.shape import Shape
from ir.dim import DimType
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import find_sum_dim, rotate_and_sum
from util.layout_util import convert_layout_to_mask, get_cts_by_dim, get_segment
from util.shape_util import get_term_shape


def lower_conv2d(env, kernel):
    """Lower CONV2D kernel to circuit IR."""
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    shape = Shape(kernel.layout.term)
    shape.run()

    # This should be a matrix-vector multiplication

    # create cs pointers
    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in a_cts.values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    # rotate a based a_shape
    a_shape = get_term_shape(kernel.cs[0].layout.term)
    b_shape = get_term_shape(kernel.cs[1].layout.term)

    pad_top = kernel.layout.term.cs[4][0]
    pad_bottom = kernel.layout.term.cs[4][1]
    pad_left = kernel.layout.term.cs[4][2]
    pad_right = kernel.layout.term.cs[4][3]

    # calculate rotation amounts for input ciphertexts
    # assumes dim 1 and 2 are continuous and in the slot dimensions
    rot_amts = []
    dim_map = []
    offset = 1
    for dim in kernel.cs[0].layout.get_dims()[::-1]:
        if dim.dim == 1 or dim.dim == 2:
            dim_map.append((dim.dim, dim.extent, offset))
        offset *= dim.extent

    dim_map = dim_map[::-1]

    rot_amts = []

    # find rotational offset from padding
    rot_offset = -pad_top * dim_map[0][2] - pad_left

    for i in range(b_shape[2]):
        for j in range(b_shape[3]):
            rot_0 = i * dim_map[0][2]
            rot_1 = j * dim_map[1][2]
            rot_amts.append(rot_0 + rot_1 + rot_offset)

    masks = []
    for i in range(b_shape[2]):
        for j in range(b_shape[3]):
            rot_0 = i * dim_map[0][2]
            rot_1 = j * dim_map[1][2]
            segment_0 = dim_map[0][1] * dim_map[0][2]
            segment_1 = dim_map[1][1] * dim_map[1][2]
            mask = [1] * kernel.cs[0].layout.n
            for k in range(kernel.cs[0].layout.n // segment_0):
                for l in range(segment_0):
                    if not 0 <= l + rot_0 - (pad_top * dim_map[0][2]) < segment_0:
                        mask[k * segment_0 + l] = 0

            for k in range(kernel.cs[0].layout.n // segment_1):
                for l in range(segment_1):
                    if not 0 <= l + rot_1 - pad_left < segment_1:
                        mask[k * segment_1 + l] = 0
            masks.append(mask)

    kernel.cs[1].layout.term.cs.append(masks)
    kernel.cs[1].layout.term.cs.append(rot_amts)

    # rotate a_cs by rot_amt
    a_rot_cs = []
    for i in range(len(rot_amts)):
        a_rot_cs.append(a_cs[i] << rot_amts[i])

    # calculate the multiplications between ct
    cts = {}
    for i, (a, b) in enumerate(zip(a_rot_cs, b_cs[: len(a_rot_cs)])):
        mul_term = a * b
        cts[i] = mul_term

    # Create initial layout_cts with input layout
    layout_cts = LayoutCiphertexts(layout=kernel.cs[0].layout, cts=cts)

    # find summing dimension
    # the sum dimension should be the 0th dimension of kernel.cs[0]
    # the sum dimension also should be the 0th and 1st dimension of kernel.cs[1]
    sum_dims = [(0, 0), (0, None)]
    for sum_dim in sum_dims:
        ct_sum_dims, slot_sum_dims = find_sum_dim(
            kernel.cs[sum_dim[0]].layout, sum_dim[1]
        )
        if not ct_sum_dims and not slot_sum_dims:
            continue

        # sum together ciphertexts
        for ct_sum_dim in ct_sum_dims:
            ct_groups = get_cts_by_dim(layout_cts, ct_sum_dim)

            # sum within group
            sum_cts = {}
            for i, group in enumerate(ct_groups):
                base = group[0]
                for j in range(1, len(group)):
                    base = base + group[j]
                sum_cts[i] = base

            # Create new layout without the summed dimension
            new_layout = create_layout_without_dims(layout_cts.layout, [ct_sum_dim])
            layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

        # sum together slot dimensions per ciphertext
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

    # mask out gap dimensions
    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(kernel.layout)])
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            mask_term = term * mask
            masked_cts[index] = mask_term
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)
    else:
        # Update layout to output layout if no mask needed
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=layout_cts.cts)

    return layout_cts
