from frontends.tensor_args import Conv2dArgs
from ir.analysis.shape import Shape
from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import find_sum_dim, rotate_and_sum
import numpy as np
from util.layout_util import (
    _get_apply_layout_plan,
    convert_layout_to_mask,
    dim_list_index,
    get_cts_by_dim,
    get_dim_indices,
    get_dim_map,
    get_segment,
)
from util.shape_util import get_term_shape


def _slot_mask_conv2d_same_stride1(
    *,
    base_indices_ct,
    rot_amt: int,
    fh: int,
    fw: int,
    pad_top: int,
    pad_left: int,
    hin: int,
    win: int,
    n: int,
) -> list[int]:
    """One ciphertext: mask post-rotation slots for ``same`` conv, stride 1, 3×3.

    Toy rotation matches ``np.roll(vec, -rot_amt)`` → ``out[s] = vec[(s + rot_amt) % n]``.
    Zero slots where the source slot would supply an out-of-bounds (implicit) neighbor
    for some output cell, including across gap dims where packing wraps incorrectly.
    """
    # Fast path: compact (n, ndims) int32 plan rows (common for ResNet-scale n).
    if (
        isinstance(base_indices_ct, np.ndarray)
        and base_indices_ct.ndim == 2
        and base_indices_ct.shape[0] == n
        and base_indices_ct.shape[1] >= 3
        and base_indices_ct.dtype == np.int32
    ):
        idx = np.arange(n, dtype=np.int64)
        src = (idx + int(rot_amt)) % n
        rows = base_indices_ct[src].astype(np.int64, copy=False)
        c0, hu, wu = rows[:, 0], rows[:, 1], rows[:, 2]
        bad = (c0 < 0) | (hu < 0) | (wu < 0)
        bad |= (hu >= hin) | (wu >= win)
        o_h = hu + int(pad_top) - int(fh)
        o_w = wu + int(pad_left) - int(fw)
        bad |= (o_h < 0) | (o_h >= hin) | (o_w < 0) | (o_w >= win)
        return np.logical_not(bad).astype(np.int32).tolist()

    mask: list[int] = [1] * n
    for s in range(n):
        src = (s + int(rot_amt)) % n
        if src < 0 or src >= len(base_indices_ct):
            mask[s] = 0
            continue
        idx = base_indices_ct[src]
        # Compact plan: int32 with -1 sentinel.
        if isinstance(idx, np.ndarray):
            if idx.shape[0] < 3:
                mask[s] = 0
                continue
            if int(idx[0]) < 0 or int(idx[1]) < 0 or int(idx[2]) < 0:
                mask[s] = 0
                continue
            hu, wu = int(idx[1]), int(idx[2])
        else:
            if not isinstance(idx, (list, tuple)) or len(idx) < 3:
                mask[s] = 0
                continue
            if any(idx[i] is None for i in range(3)):
                mask[s] = 0
                continue
            hu, wu = int(idx[1]), int(idx[2])
        if hu < 0 or hu >= hin or wu < 0 or wu >= win:
            mask[s] = 0
            continue
        o_h = hu + pad_top - fh
        o_w = wu + pad_left - fw
        if o_h < 0 or o_h >= hin or o_w < 0 or o_w >= win:
            mask[s] = 0
        else:
            mask[s] = 1
    return mask


def lower_conv2d(env, kernel):
    """Lower CONV2D kernel to circuit IR."""
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    conv_term = kernel.layout.term
    pad_list = Conv2dArgs.get_computed_padding(conv_term)
    if pad_list is None or len(pad_list) != 4:
        raise RuntimeError(
            "conv2d kernel term is missing precomputed padding from layout gen"
        )

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

    pad_top = int(pad_list[0])
    _pad_bottom = int(pad_list[1])
    pad_left = int(pad_list[2])
    _pad_right = int(pad_list[3])
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

    # Replicate filter masks for each CT in b layout - use layout to get filter position per ct_idx
    b_num_cts = len(b_cs)
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

    filter_masks: list[list[int]] = []
    base_by_ct: list | None = None
    hin = win = 0
    if int(_stride) != 1:
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
    else:
        layout_a = kernel.cs[0].layout
        layout_len = max(len(layout_a), layout_a.n)
        plan = _get_apply_layout_plan(layout_a, 3, layout_len=layout_len)
        base_by_ct = plan["base_indices_by_cts"]
        hin, win = int(_a_shape[1]), int(_a_shape[2])

    masks: list[list[int]] = []
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
        rot_amt = rot_amts[filter_idx]
        a_idx = ct_idx
        if int(_stride) == 1:
            assert base_by_ct is not None
            m = _slot_mask_conv2d_same_stride1(
                base_indices_ct=base_by_ct[a_idx],
                rot_amt=rot_amt,
                fh=int(f_h),
                fw=int(f_w),
                pad_top=int(pad_top),
                pad_left=int(pad_left),
                hin=hin,
                win=win,
                n=kernel.cs[0].layout.n,
            )
        else:
            m = filter_masks[filter_idx]
        masks.append(m)

    # Some downstream helpers (e.g. punctured layout application in Toy checks) read
    # precomputed conv masks from `layout.term.cs[4]`. Preserve that convention.
    # Only attach if not already present to avoid unbounded growth when lowering
    # is invoked multiple times in the same process.
    b_term_cs = kernel.cs[1].layout.term.cs
    while len(b_term_cs) <= 5:
        b_term_cs.append(None)
    b_term_cs[4] = masks
    b_term_cs[5] = rot_amts

    # Apply per-filter padding masks during conv lowering.
    # Without this, rotations wrap around in the packed HE vector and leak values
    # from outside the valid padded region, which becomes visible for some layouts
    # (notably when H/W are separated by gap slot dims).
    # Share HETerm.mask nodes when slot mask vectors repeat (e.g. same filter tap
    # across many ciphertexts). Key by packed bytes — masks are 0/1 per slot.
    _mask_term_by_key: dict[bytes, HETerm] = {}
    mask_terms: list[HETerm] = []
    for m in masks:
        k = bytes(m)
        t = _mask_term_by_key.get(k)
        if t is None:
            t = HETerm.mask([m])
            _mask_term_by_key[k] = t
        mask_terms.append(t)

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
        a_idx = b_idx
        a_rot = a_cs[a_idx] << rot_amt
        a_masked = a_rot * mask_terms[b_idx]
        cts[b_idx] = a_masked * b_ct

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
    # after each rotate_and_sum so Dim references stay valid.
    #
    # Sum **inner** slot axes (larger index in ``slot_dims``) before **outer**
    # ones: removing the outer fragment first coalesces gaps and changes
    # ``get_segment`` / ``mul_offset`` for the inner axis.
    while True:
        _, slot_sum_dims = find_sum_dim(a_layout_cts.layout, 0)
        if not slot_sum_dims:
            break
        slot_dims_cur = a_layout_cts.layout.slot_dims
        slot_sum_dim = max(
            slot_sum_dims,
            key=lambda d: dim_list_index(d, slot_dims_cur),
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
