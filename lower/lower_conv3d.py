from ir.analysis.shape import Shape
from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import find_sum_dim, rotate_and_sum
from util.layout_util import (
    convert_layout_to_mask,
    get_cts_by_dim,
    get_dim_indices,
    get_dim_map,
    get_segment,
)
from util.shape_util import get_term_shape


def lower_conv3d(env, kernel):
    """Lower CONV3D kernel to circuit IR (secret input, public filter)."""
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    shape = Shape(kernel.layout.term)
    shape.run()

    a_cs = [
        HETerm(HEOp.CS, [a_cts[ct_idx]], a_cts[ct_idx].secret)
        for ct_idx in sorted(a_cts.keys())
    ]
    b_cs = [
        HETerm(HEOp.CS, [b_cts[ct_idx]], b_cts[ct_idx].secret)
        for ct_idx in sorted(b_cts.keys())
    ]

    b_shape = get_term_shape(kernel.cs[1].layout.term)
    # Padding: [pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]
    pad_front = kernel.layout.term.cs[4][0]
    _pad_back = kernel.layout.term.cs[4][1]
    pad_top = kernel.layout.term.cs[4][2]
    _pad_bottom = kernel.layout.term.cs[4][3]
    pad_left = kernel.layout.term.cs[4][4]
    _pad_right = kernel.layout.term.cs[4][5]

    spatial = {}
    offset = 1
    for dim in kernel.cs[0].layout.get_dims()[::-1]:
        if dim.dim in (1, 2, 3) and dim.dim_type != DimType.EMPTY:
            spatial[dim.dim] = (dim.extent, offset)
        offset *= dim.extent
    assert 1 in spatial and 2 in spatial and 3 in spatial
    d_extent, d_stride = spatial[1]
    h_extent, h_stride = spatial[2]
    w_extent, w_stride = spatial[3]

    rot_offset = -pad_front * d_stride - pad_top * h_stride - pad_left * w_stride

    k_d, k_h, k_w = b_shape[2], b_shape[3], b_shape[4]
    rot_amts = []
    for i in range(k_d):
        for j in range(k_h):
            for k in range(k_w):
                rot_amts.append(i * d_stride + j * h_stride + k * w_stride + rot_offset)

    filter_masks = []
    for i in range(k_d):
        for j in range(k_h):
            for k in range(k_w):
                rot_d = i * d_stride
                rot_h = j * h_stride
                rot_w = k * w_stride
                mask = [1] * kernel.cs[0].layout.n

                segment_d = d_extent * d_stride
                for loop in range(kernel.cs[0].layout.n // segment_d):
                    for idx in range(segment_d):
                        if not 0 <= idx + rot_d - (pad_front * d_stride) < segment_d:
                            mask[loop * segment_d + idx] = 0

                segment_h = h_extent * h_stride
                for loop in range(kernel.cs[0].layout.n // segment_h):
                    for idx in range(segment_h):
                        if not 0 <= idx + rot_h - (pad_top * h_stride) < segment_h:
                            mask[loop * segment_h + idx] = 0

                segment_w = w_extent * w_stride
                for loop in range(kernel.cs[0].layout.n // segment_w):
                    for idx in range(segment_w):
                        if not 0 <= idx + rot_w - (pad_left * w_stride) < segment_w:
                            mask[loop * segment_w + idx] = 0

                filter_masks.append(mask)

    b_num_cts = kernel.cs[1].layout.num_ct()
    b_ct_dims = kernel.cs[1].layout.ct_dims
    dim_indices_masks = get_dim_indices(b_ct_dims)
    dim_map_masks = get_dim_map(b_ct_dims)
    dim_to_pos_masks = {
        dim.dim: dim_map_masks[dim] for dim in b_ct_dims if dim.dim is not None
    }
    flat_w = k_w
    flat_hw = k_h * k_w
    masks = []
    for ct_idx in range(b_num_cts):
        f_d = (
            dim_indices_masks[dim_to_pos_masks[2]][ct_idx]
            if 2 in dim_to_pos_masks
            else 0
        )
        f_h = (
            dim_indices_masks[dim_to_pos_masks[3]][ct_idx]
            if 3 in dim_to_pos_masks
            else 0
        )
        f_w = (
            dim_indices_masks[dim_to_pos_masks[4]][ct_idx]
            if 4 in dim_to_pos_masks
            else 0
        )
        filter_idx = f_d * flat_hw + f_h * flat_w + f_w
        masks.append(filter_masks[filter_idx])

    kernel.cs[1].layout.term.cs.append(masks)
    kernel.cs[1].layout.term.cs.append(rot_amts)

    dim_indices = get_dim_indices(b_ct_dims)
    dim_map = get_dim_map(b_ct_dims)
    dim_to_pos = {dim.dim: dim_map[dim] for dim in b_ct_dims if dim.dim is not None}

    cts = {}
    for b_idx, b_ct in enumerate(b_cs):
        f_d = dim_indices[dim_to_pos[2]][b_idx] if 2 in dim_to_pos else 0
        f_h = dim_indices[dim_to_pos[3]][b_idx] if 3 in dim_to_pos else 0
        f_w = dim_indices[dim_to_pos[4]][b_idx] if 4 in dim_to_pos else 0
        filter_idx = f_d * flat_hw + f_h * flat_w + f_w
        rot_amt = rot_amts[filter_idx]
        a_idx = min(b_idx, len(a_cs) - 1)
        cts[b_idx] = (a_cs[a_idx] << rot_amt) * b_ct

    b_layout_cts = LayoutCiphertexts(layout=kernel.cs[1].layout, cts=cts)
    filter_ct_dims = [
        dim
        for dim in b_layout_cts.layout.ct_dims
        if (dim.dim is not None and dim.dim != 0) and dim.dim_type == DimType.FILL
    ]

    filter_a_dims = []
    for a_dim, b_dim in zip(
        kernel.cs[0].layout.get_dims(), kernel.cs[1].layout.get_dims()
    ):
        if b_dim not in filter_ct_dims:
            filter_a_dims.append(a_dim)

    for ct_sum_dim in filter_ct_dims:
        if ct_sum_dim not in b_layout_cts.layout.ct_dims:
            continue
        ct_groups = get_cts_by_dim(b_layout_cts, ct_sum_dim)
        sum_cts = {}
        for i, group in enumerate(ct_groups):
            base = group[0]
            for j in range(1, len(group)):
                base = base + group[j]
            sum_cts[i] = base
        new_layout = create_layout_without_dims(b_layout_cts.layout, [ct_sum_dim])
        b_layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

    a_layout = Layout(
        kernel.cs[0].layout.term,
        [],
        filter_a_dims,
        kernel.cs[0].layout.n,
        kernel.cs[0].layout.secret,
    )
    a_layout_cts = LayoutCiphertexts(layout=a_layout, cts=b_layout_cts.cts)

    while True:
        _, slot_sum_dims = find_sum_dim(a_layout_cts.layout, 0)
        if not slot_sum_dims:
            break
        slot_dims_cur = a_layout_cts.layout.slot_dims
        slot_sum_dim = min(
            slot_sum_dims,
            key=lambda d: get_segment(d, slot_dims_cur)[2],
        )
        segment = get_segment(slot_sum_dim, slot_dims_cur)
        extent = segment[1]
        mul_offset = segment[2]
        summed_cts = {}
        for index, term in a_layout_cts.cts.items():
            summed_cts[index] = rotate_and_sum(term, extent, mul_offset)
        new_layout = create_layout_without_dims(a_layout_cts.layout, [slot_sum_dim])
        a_layout_cts = LayoutCiphertexts(layout=new_layout, cts=summed_cts)

    layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=a_layout_cts.cts)

    # Valid: zero slots outside the logical output box (d>=D_out, h>=H_out, w>=W_out).
    logical_out_shape = shape.get_shape(kernel.layout.term)
    if kernel.layout.term.cs[3] == "valid" and logical_out_shape is not None:
        _c_o, d_o, h_o, w_o = logical_out_shape
        slot_dims = kernel.layout.slot_dims
        slot_dim_indices = get_dim_indices(slot_dims)
        dim_to_indices = {}
        for dim, indices in zip(slot_dims, slot_dim_indices):
            if dim.dim_type == DimType.EMPTY:
                continue
            if dim.dim in (1, 2, 3):
                dim_to_indices[dim.dim] = indices

        if 1 in dim_to_indices and 2 in dim_to_indices and 3 in dim_to_indices:
            mask = [1] * kernel.layout.n
            d_idx = dim_to_indices[1]
            h_idx = dim_to_indices[2]
            w_idx = dim_to_indices[3]
            for i in range(kernel.layout.n):
                if d_idx[i] >= d_o or h_idx[i] >= h_o or w_idx[i] >= w_o:
                    mask[i] = 0
            mask_term = HETerm.mask([mask])
            masked_cts = {}
            for index, term in layout_cts.cts.items():
                masked_cts[index] = term * mask_term
            layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)

    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(kernel.layout)])
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            masked_cts[index] = term * mask
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)

    return layout_cts
