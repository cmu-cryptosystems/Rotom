from ir.analysis.shape import Shape
from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.kernel import KernelOp
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
    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in a_cts.values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    # rotate a based a_shape
    _a_shape = get_term_shape(kernel.cs[0].layout.term)
    b_shape = get_term_shape(kernel.cs[1].layout.term)

    pad_top = kernel.layout.term.cs[4][0]
    _pad_bottom = kernel.layout.term.cs[4][1]
    pad_left = kernel.layout.term.cs[4][2]
    _pad_right = kernel.layout.term.cs[4][3]
    _stride = kernel.layout.term.cs[2]

    # calculate rotation amounts for input ciphertexts
    # assumes dim 1 and 2 are continuous and in the slot dimensions
    # filter position (i,j) reads input at (h+i, w+j) for output (h,w); stride does not change these offsets
    rot_amts = []
    dim_map = []
    offset = 1
    for dim in kernel.cs[0].layout.get_dims()[::-1]:
        if dim.dim == 1 or dim.dim == 2:
            dim_map.append((dim.dim, dim.extent, offset))
        offset *= dim.extent

    dim_map = dim_map[::-1]

    rot_offset = -pad_top * dim_map[0][2] - pad_left

    for i in range(b_shape[2]):
        for j in range(b_shape[3]):
            rot_0 = i * dim_map[0][2]
            rot_1 = j * dim_map[1][2]
            rot_amts.append(rot_0 + rot_1 + rot_offset)

    filter_masks = []
    for i in range(b_shape[2]):
        for j in range(b_shape[3]):
            rot_0 = i * dim_map[0][2]
            rot_1 = j * dim_map[1][2]
            segment_0 = dim_map[0][1] * dim_map[0][2]
            segment_1 = dim_map[1][1] * dim_map[1][2]
            mask = [1] * kernel.cs[0].layout.n
            for k in range(kernel.cs[0].layout.n // segment_0):
                for idx in range(segment_0):
                    if not 0 <= idx + rot_0 - (pad_top * dim_map[0][2]) < segment_0:
                        mask[k * segment_0 + idx] = 0

            for k in range(kernel.cs[0].layout.n // segment_1):
                for idx in range(segment_1):
                    if not 0 <= idx + rot_1 - pad_left < segment_1:
                        mask[k * segment_1 + idx] = 0
            filter_masks.append(mask)

    # Replicate filter masks for each CT in b layout - use layout to get filter position per ct_idx
    num_filter_positions = b_shape[2] * b_shape[3]
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

    # OPTIMIZATION: Lift filter alignment rotations to packing
    # Instead of creating rotations on CS terms, create pre-rotated PACK operations
    # This requires finding the original input layout (before replication)

    # Find original input layout by tracing back through kernel.cs[0]
    # kernel.cs[0] might be a REPLICATE kernel, so we need to find the original TENSOR
    original_input_layout = None
    input_kernel = kernel.cs[0]
    while input_kernel.op == KernelOp.REPLICATE and len(input_kernel.cs) > 0:
        input_kernel = input_kernel.cs[0]
    if input_kernel.op == KernelOp.TENSOR:
        original_input_layout = input_kernel.layout

    # If we found the original layout, we can create pre-rotated PACK operations
    # Otherwise, fall back to creating rotations (which will be lifted by optimization pass)
    f_h, f_w = b_shape[2], b_shape[3]
    num_filter_positions = f_h * f_w
    c_in_count = b_shape[1]
    b_ct_dims = kernel.cs[1].layout.ct_dims
    # Stride to skip within a filter position (e.g. R dim extent)
    _inner_stride = 1
    if b_ct_dims and b_ct_dims[-1].dim is None:
        _inner_stride = b_ct_dims[-1].extent

    a_rot_cs = []
    if original_input_layout is not None:
        # Create pre-rotated PACK operations for each (c_in, filter_position) pair.
        # Product (c_out, c_in, f_h, f_w) needs input from channel c_in, rotated for filter (f_h, f_w).
        # packing_idx = c_in (which input CT), rot_amt = rot_amts[filter_idx].
        for c_in in range(c_in_count):
            for filter_idx, rot_amt in enumerate(rot_amts):
                packing_idx = (
                    c_in % original_input_layout.num_ct()
                    if original_input_layout.num_ct() > 0
                    else 0
                )
                metadata = f"{packing_idx} {kernel.cs[0]} rot:{rot_amt}"
                pre_rotated_pack = HETerm(
                    HEOp.PACK,
                    [original_input_layout],
                    original_input_layout.secret,
                    metadata,
                )
                a_rot_cs.append(
                    HETerm(HEOp.CS, [pre_rotated_pack], pre_rotated_pack.secret)
                )
    else:
        # Fallback: create rotations (will be lifted by optimization pass if possible)
        for i in range(len(rot_amts)):
            a_rot_cs.append(a_cs[i] << rot_amts[i])

    # calculate the multiplications between ct
    # Create products for all b positions; a_rot depends on (c_in, filter_position).
    # Decode b_idx from b's layout so we get correct (c_in, f_h, f_w) regardless of ct_dims order.
    b_ct_dims = kernel.cs[1].layout.ct_dims
    dim_indices = get_dim_indices(b_ct_dims)
    dim_map = get_dim_map(b_ct_dims)
    # Tensor dims: 0=c_out, 1=c_in, 2=f_h, 3=f_w. Find which ct_dim has each.
    dim_to_pos = {dim.dim: dim_map[dim] for dim in b_ct_dims if dim.dim is not None}
    _f_h_extent = b_shape[2]
    f_w_extent = b_shape[3]
    cts = {}
    for b_idx, b_ct in enumerate(b_cs):
        c_in = dim_indices[dim_to_pos[1]][b_idx] if 1 in dim_to_pos else 0
        f_h = dim_indices[dim_to_pos[2]][b_idx] if 2 in dim_to_pos else 0
        f_w = dim_indices[dim_to_pos[3]][b_idx] if 3 in dim_to_pos else 0
        filter_idx = f_h * f_w_extent + f_w
        rot_idx = c_in * num_filter_positions + filter_idx
        rot_idx = min(rot_idx, len(a_rot_cs) - 1)
        cts[b_idx] = a_rot_cs[rot_idx] * b_ct

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

    # find summing dimension for input channels
    _, slot_sum_dims = find_sum_dim(kernel.cs[0].layout, 0)

    # sum together slot dimensions per ciphertext
    for slot_sum_dim in slot_sum_dims:
        # slot_sum_dim is from input layout; use input if not in current layout
        slot_dims = (
            a_layout_cts.layout.slot_dims
            if slot_sum_dim in a_layout_cts.layout.slot_dims
            else kernel.cs[0].layout.slot_dims
        )
        if slot_sum_dim not in slot_dims:
            continue
        segment = get_segment(slot_sum_dim, slot_dims)
        extent = segment[1]
        mul_offset = segment[2]
        summed_cts = {}
        for index, term in a_layout_cts.cts.items():
            summed_cts[index] = rotate_and_sum(term, extent, mul_offset)
        # Create new layout without the summed dimension
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
