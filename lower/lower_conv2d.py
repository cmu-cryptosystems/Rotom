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


def lower_conv2d(env, kernel):
    """Lower CONV2D kernel to circuit IR."""
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    shape = Shape(kernel.layout.term)
    shape.run()

    # This should be a matrix-vector multiplication

    # create cs pointers
    # Keep ciphertext index ordering stable: downstream mapping assumes ct index alignment.
    a_cs = [
        HETerm(HEOp.CS, [a_cts[ct_idx]], a_cts[ct_idx].secret)
        for ct_idx in sorted(a_cts.keys())
    ]
    b_cs = [
        HETerm(HEOp.CS, [b_cts[ct_idx]], b_cts[ct_idx].secret)
        for ct_idx in sorted(b_cts.keys())
    ]

    # rotate a based a_shape
    _a_shape = get_term_shape(kernel.cs[0].layout.term)
    b_shape = get_term_shape(kernel.cs[1].layout.term)

    pad_top = kernel.layout.term.cs[4][0]
    _pad_bottom = kernel.layout.term.cs[4][1]
    pad_left = kernel.layout.term.cs[4][2]
    _pad_right = kernel.layout.term.cs[4][3]
    _stride = kernel.layout.term.cs[2]

    # calculate rotation amounts for input ciphertexts
    # derive spatial multipliers from actual layout order (supports arbitrary dim ordering)
    # while preserving the historical rotation basis used in lowering.
    rot_amts = []
    spatial = {}
    offset = 1
    for dim in kernel.cs[0].layout.get_dims()[::-1]:
        if dim.dim in (1, 2) and dim.dim_type != DimType.EMPTY:
            spatial[dim.dim] = (dim.extent, offset)
        offset *= dim.extent
    assert 1 in spatial and 2 in spatial
    h_extent, h_stride = spatial[1]
    w_extent, w_stride = spatial[2]

    rot_offset = -pad_top * h_stride - pad_left * w_stride

    for i in range(b_shape[2]):
        for j in range(b_shape[3]):
            rot_0 = i * h_stride
            rot_1 = j * w_stride
            rot_amts.append(rot_0 + rot_1 + rot_offset)

    filter_masks = []
    for i in range(b_shape[2]):
        for j in range(b_shape[3]):
            rot_0 = i * h_stride
            rot_1 = j * w_stride
            segment_0 = h_extent * h_stride
            segment_1 = w_extent * w_stride
            mask = [1] * kernel.cs[0].layout.n
            for k in range(kernel.cs[0].layout.n // segment_0):
                for idx in range(segment_0):
                    if not 0 <= idx + rot_0 - (pad_top * h_stride) < segment_0:
                        mask[k * segment_0 + idx] = 0

            for k in range(kernel.cs[0].layout.n // segment_1):
                for idx in range(segment_1):
                    if not 0 <= idx + rot_1 - (pad_left * w_stride) < segment_1:
                        mask[k * segment_1 + idx] = 0
            filter_masks.append(mask)

    # Replicate filter masks for each CT in b layout - use layout to get filter position per ct_idx
    # num_filter_positions = b_shape[2] * b_shape[3]
    b_num_cts = kernel.cs[1].layout.num_ct()
    b_ct_dims = kernel.cs[1].layout.ct_dims
    _inner_stride = 1
    if b_ct_dims and b_ct_dims[-1].dim is None:
        _inner_stride = b_ct_dims[-1].extent
    dim_indices_masks = get_dim_indices(b_ct_dims)
    dim_map_masks = get_dim_map(b_ct_dims)
    dim_to_pos_masks = {
        dim.dim: dim_map_masks[dim] for dim in b_ct_dims if dim.dim is not None
    }
    f_w_extent_masks = b_shape[3]
    masks = []
    for ct_idx in range(b_num_cts):
        f_h = (
            dim_indices_masks[dim_to_pos_masks[2]][ct_idx]
            if 2 in dim_to_pos_masks
            else 0
        )
        f_w = (
            dim_indices_masks[dim_to_pos_masks[3]][ct_idx]
            if 3 in dim_to_pos_masks
            else 0
        )
        filter_idx = f_h * f_w_extent_masks + f_w
        masks.append(filter_masks[filter_idx])

    kernel.cs[1].layout.term.cs.append(masks)
    kernel.cs[1].layout.term.cs.append(rot_amts)

    # calculate the multiplications between ct
    # For each b_idx, use the already-aligned a_idx (same ct index) and rotate only
    # by filter spatial position. This avoids assuming channel contiguity in a_ct dims
    # and naturally supports split channel dims with gap slots.
    b_ct_dims = kernel.cs[1].layout.ct_dims
    dim_indices = get_dim_indices(b_ct_dims)
    dim_map = get_dim_map(b_ct_dims)
    # Tensor dims: 0=c_out, 1=c_in, 2=f_h, 3=f_w. Find which ct_dim has each.
    dim_to_pos = {dim.dim: dim_map[dim] for dim in b_ct_dims if dim.dim is not None}
    f_w_extent = b_shape[3]
    cts = {}
    for b_idx, b_ct in enumerate(b_cs):
        f_h = dim_indices[dim_to_pos[2]][b_idx] if 2 in dim_to_pos else 0
        f_w = dim_indices[dim_to_pos[3]][b_idx] if 3 in dim_to_pos else 0
        filter_idx = f_h * f_w_extent + f_w
        rot_amt = rot_amts[filter_idx]
        a_idx = min(b_idx, len(a_cs) - 1)
        cts[b_idx] = (a_cs[a_idx] << rot_amt) * b_ct

    # find summation dimensions from b layout
    b_layout_cts = LayoutCiphertexts(layout=kernel.cs[1].layout, cts=cts)
    filter_ct_dims = [
        dim
        for dim in b_layout_cts.layout.ct_dims
        if (dim.dim is not None and dim.dim != 0) and dim.dim_type == DimType.FILL
    ]

    # get the output layout according to a_dims
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

        # sum within group
        sum_cts = {}
        for i, group in enumerate(ct_groups):
            base = group[0]
            for j in range(1, len(group)):
                base = base + group[j]
            sum_cts[i] = base

        new_layout = create_layout_without_dims(b_layout_cts.layout, [ct_sum_dim])
        b_layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

    # rename b_layout_cts to a_layout_cts, and use a's layout
    a_layout = Layout(
        kernel.cs[0].layout.term,
        [],
        filter_a_dims,
        kernel.cs[0].layout.n,
        kernel.cs[0].layout.secret,
    )
    a_layout_cts = LayoutCiphertexts(layout=a_layout, cts=b_layout_cts.cts)

    # Sum input-channel slot dimensions. Channel may be split across several slot
    # dims (same tensor dim index). Re-query find_sum_dim on the layout we update
    # after each rotate_and_sum so Dim references stay valid; pick the remaining
    # dim with the smallest slot stride (inner radix) first so get_segment agrees
    # with the current slot_dims list (see dim_list_index / duplicate [G:n] gaps).
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

    return layout_cts
