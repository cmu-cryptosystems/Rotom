import math

import numpy as np

from ir.analysis.shape import Shape
from ir.dim import DimType
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from lower.lower_util import rotate_and_sum
from util.layout_util import convert_layout_to_mask, get_segment
from util.util import split_lists


def sum_slot_dim(kernel, ct, slot_sum_dims):
    # sum together slot dimensions per ciphertext
    slot_dims = kernel.cs[0].layout.slot_dims
    if slot_sum_dims:
        for slot_sum_dim in slot_sum_dims:
            segment = get_segment(slot_sum_dim, slot_dims)
            extent = segment[1]
            mul_offset = segment[2]
            ct = rotate_and_sum(ct, extent, mul_offset)

    # mask out gap dimensions
    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm(
            HEOp.MASK, [convert_layout_to_mask(kernel.layout)], False, "output mask"
        )
        mask_term = ct * mask
        return mask_term
    return ct


def lower_conv2d_roll(env, kernel):
    """same as a normal mul, but with filtering rules"""
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    shape = Shape(kernel.layout.term)
    shape.run()

    # figure out filters
    padding = kernel.layout.term.cs[4]
    a_shape = shape.shapes[kernel.layout.term.cs[0]]
    b_shape = shape.shapes[kernel.layout.term.cs[1]]
    i_c = a_shape[0]
    i_h = a_shape[1]
    i_w = a_shape[2]
    assert i_h == i_w
    f_o = b_shape[0]
    f_i = b_shape[1]
    f_h = b_shape[2]
    f_w = b_shape[3]

    # a_cs is a list of cts
    # Note: split_roll returns dict with list values, but LayoutCiphertexts has single HETerm values
    a_cs = []
    for cts_item in a_cts.values():
        # Handle both list (from split_roll) and single HETerm (from LayoutCiphertexts)
        if isinstance(cts_item, list):
            cts_list = cts_item
        else:
            cts_list = [cts_item]
        a_cts_list = []
        for ct in cts_list:
            a_cts_list.append(HETerm(HEOp.CS, [ct], ct.secret))
        a_cs.append(a_cts_list)
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    split_a_cs = split_lists(a_cs, i_h)
    split_b_cs = split_lists(b_cs, i_h)

    # cts are split rolls
    # each ct can have either a length of 1, 2, or 4
    # if 1: then there's no rotation applied
    # if 2: then there's a left and right rotation applied
    # if 4: then there's a ll, lr, rl, rr rotation applied
    total = []
    cts = []
    for i in range(i_h):
        a_splits = split_a_cs[i]
        group = []
        for j in range(len(a_splits)):
            split = []
            for k in range(len(a_splits[j])):
                total.append(a_splits[j][k] * split_b_cs[i][j])
                split.append(a_splits[j][k] * split_b_cs[i][j])
            group.append(split)
        cts.append(group)

    if padding == [0, 0, 0, 0]:
        # 1x1 convolution - no padding, single element multiplication
        # For 1x1 filter, we only need the first element (no rotation)
        output_cts = {}
        output_cts[0] = cts[0][0][0]

    elif padding == [0, 1, 0, 1]:
        filter_cts = cts[:f_h]
        for i in range(len(filter_cts)):
            filter_cts[i] = filter_cts[i][:f_w]

        # keep only left masks
        left_rots = []
        for ct in filter_cts:
            for c in ct:
                left_rots.append(c[0])

        # flatten filtered list
        output_cts = {}
        for i in range(len(left_rots)):
            output_cts[i] = left_rots[i]

    elif padding == [1, 1, 1, 1]:
        # keep left masks
        left_rots = []
        for group in cts[: f_h - 1]:
            for split in group[: f_w - 1]:
                left_rots.append(split[0])

        # keep right masks
        right_rots = []
        for group in cts[: f_h - 1]:
            split = group[-1]
            if len(split) == 2:
                right_rots.append(split[-1])
            elif len(split) == 4:
                right_rots.append(split[-2])

        group = cts[-1]
        for i in range(f_h - 1):
            right_rots.append(group[i][1])
        right_rots.append(group[-1][-1])

        both = left_rots + right_rots

        # flatten filtered list
        output_cts = {}
        for i in range(len(both)):
            output_cts[i] = both[i]
    else:
        raise NotImplementedError("different padding")

    # sum all cts
    sum_together = output_cts[0]
    for i in range(1, len(output_cts)):
        sum_together += output_cts[i]

    # sum together input channels
    slot_dims = kernel.cs[0].layout.slot_dims
    slot_sum_dims = []
    for dim in slot_dims:
        if dim.dim == 0:
            slot_sum_dims.append(dim)
    ct = sum_slot_dim(kernel, sum_together, slot_sum_dims)
    return LayoutCiphertexts(layout=kernel.layout, cts={0: ct})
