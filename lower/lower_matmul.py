from copy import deepcopy as copy

from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import bsgs, find_sum_dim, rotate_and_sum
from util.layout_util import (
    convert_layout_to_mask,
    get_ct_idxs_by_dim,
    get_cts_by_dim,
    get_dim_indices,
    get_segment,
)
from util.util import prod


def lower_matmul(env, kernel):
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    # create cs pointers
    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in a_cts.values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    # calculate the multiplications between ct
    cts = {}
    for i, (a, b) in enumerate(zip(a_cs, b_cs)):
        mul_term = a * b
        cts[i] = mul_term

    # Create initial layout_cts with input layout
    layout_cts = LayoutCiphertexts(layout=kernel.cs[0].layout, cts=cts)

    # find summing dimension
    sum_dim = max(
        dim.dim for dim in kernel.cs[0].layout.get_dims() if dim.dim is not None
    )
    ct_sum_dims, slot_sum_dims = find_sum_dim(kernel.cs[0].layout, sum_dim)

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
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=masked_cts)
    else:
        # Update layout to output layout if no mask needed
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)

    return layout_cts


def lower_bsgs_matmul(env, kernel):
    if kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL:
        replicated_kernel = kernel.cs[1].cs[1]
        rot_rolled_kernel = kernel.cs[1]
        rolled_kernel = kernel.cs[2]
    else:
        replicated_kernel = kernel.cs[2].cs[1]
        rot_rolled_kernel = kernel.cs[2]
        rolled_kernel = kernel.cs[1]

    replicated_cts = env[replicated_kernel]
    rolled_cts = env[rolled_kernel]
    assert replicated_cts.keys() == rolled_cts.keys()

    repeated_ct_dims = [d for d in replicated_kernel.layout.ct_dims if d.dim is None]
    assert len(repeated_ct_dims) == 1

    dims = rot_rolled_kernel.layout.get_dims()
    roll = kernel.cs[0]
    dim_size = dims[roll[0]].extent
    stride = prod(dim.extent for dim in rot_rolled_kernel.layout.slot_dims[1:])

    num_output_cts = kernel.layout.num_ct()
    rep_dim = repeated_ct_dims[0]

    rep_ct_dims = replicated_kernel.layout.ct_dims
    rolled_ct_dims = rolled_kernel.layout.ct_dims
    rep_dim_indices = get_dim_indices(rep_ct_dims)
    rolled_dim_indices = get_dim_indices(rolled_ct_dims)
    num_rep = len(rep_dim_indices[0])
    num_rolled = len(rolled_dim_indices[0])

    # Find the inner-product group dim in replicated ct_dims (non-replicated dim).
    # This groups a's columns (shared dimension) — each inner chunk uses a different group.
    rep_inner_dim_idx = None
    for i, d in enumerate(rep_ct_dims):
        if d.dim is not None:
            rep_inner_dim_idx = i
            break

    # In rolled ct_dims, identify:
    #   - diagonal dim (extent == dim_size, used by BSGS baby/giant steps)
    #   - output dim (matches the output layout's ct_dims by dim and extent)
    #   - inner chunk dims (remaining — need to accumulate over these)
    output_ct_dims = kernel.layout.ct_dims
    diag_dim_idx = None
    rolled_output_dim_idx = None
    inner_dim_indices = []
    for idx, d in enumerate(rolled_ct_dims):
        if diag_dim_idx is None and d.extent == dim_size:
            diag_dim_idx = idx
        elif (
            rolled_output_dim_idx is None
            and d.dim is not None
            and any(od.dim == d.dim and od.extent == d.extent for od in output_ct_dims)
        ):
            rolled_output_dim_idx = idx
        else:
            inner_dim_indices.append(idx)

    # Replicated ct lookup: inner_chunk_index -> first replicated ct index for that group.
    # Each inner chunk k uses a different column group of the input.
    rep_lookup = {}
    if rep_inner_dim_idx is not None:
        for i in range(num_rep):
            k = rep_dim_indices[rep_inner_dim_idx][i]
            if k not in rep_lookup:
                rep_lookup[k] = i
    else:
        rep_lookup[0] = 0

    # Rolled ct lookup: (output_group, diag_val, inner_chunk_vals) -> ct index
    rolled_lookup = {}
    for i in range(num_rolled):
        d_val = rolled_dim_indices[diag_dim_idx][i]
        g_val = (
            rolled_dim_indices[rolled_output_dim_idx][i]
            if rolled_output_dim_idx is not None
            else 0
        )
        inner_val = (
            tuple(rolled_dim_indices[idx][i] for idx in inner_dim_indices)
            if inner_dim_indices
            else ()
        )
        rolled_lookup[(g_val, d_val, inner_val)] = i

    inner_chunk_values = sorted(set(key[2] for key in rolled_lookup))

    # Run BSGS for each output group, accumulating over inner chunks.
    # Each inner chunk uses a different replicated ct (different input column group).
    cts = {}
    print("num_output_cts", num_output_cts)
    for g in range(num_output_cts):
        accumulated = None
        for chunk_idx, inner_val in enumerate(inner_chunk_values):
            rep_ct = replicated_cts[rep_lookup.get(chunk_idx, 0)]
            pts = {}
            for d in range(dim_size):
                pts[d] = rolled_cts[rolled_lookup[(g, d, inner_val)]]
            print("bsgs: g", g)
            print("bsgs: inner_val", inner_val)
            print("bsgs: dim_size", dim_size)
            print("bsgs: stride", stride)
            print()
            partial = bsgs(rep_ct, pts, dim_size, stride, True)
            if accumulated is None:
                accumulated = partial
            else:
                accumulated = accumulated + partial
        cts[g] = accumulated

    # When the shared dimension spans both ct_dims (handled by BSGS rotation)
    # and slot_dims, apply rotate_and_sum for the remaining slot-level portion.
    slot_sum_dims = []
    for a_dim, b_dim in zip(
        kernel.cs[1].layout.slot_dims, kernel.cs[2].layout.slot_dims
    ):
        if a_dim.dim is not None and b_dim.dim is not None:
            slot_sum_dims.append(a_dim)

    if slot_sum_dims:
        kernel_cs = [cs for cs in kernel.cs if isinstance(cs, Kernel)]
        sum_dim = max(
            dim.dim for dim in kernel_cs[0].layout.get_dims() if dim.dim is not None
        )
        intermediate_dims = []
        for ct_dim in kernel.cs[1].layout.ct_dims:
            if ct_dim.dim != sum_dim:
                intermediate_dims.append(ct_dim)
        intermediate_dims.extend(kernel.cs[1].layout.slot_dims)
        intermediate_layout = Layout(
            kernel.layout.term,
            [],
            intermediate_dims,
            kernel.layout.n,
            kernel.layout.secret,
        )
        layout_cts = LayoutCiphertexts(layout=intermediate_layout, cts=cts)

        for slot_sum_dim in slot_sum_dims:
            segment = get_segment(slot_sum_dim, layout_cts.layout.slot_dims)
            extent = segment[1]
            mul_offset = segment[2]
            summed_cts = {}
            for index, term in layout_cts.cts.items():
                summed_cts[index] = rotate_and_sum(term, extent, mul_offset)
            new_layout = create_layout_without_dims(layout_cts.layout, [slot_sum_dim])
            layout_cts = LayoutCiphertexts(layout=new_layout, cts=summed_cts)

        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)
    else:
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=cts)

    # mask out gap dimensions
    needs_mask = False
    for dim in layout_cts.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(layout_cts.layout)])
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            mask_term = term * mask
            masked_cts[index] = mask_term
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)

    return layout_cts
