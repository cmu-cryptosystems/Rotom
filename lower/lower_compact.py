from copy import deepcopy as copy

from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import (
    align_dimension_extents_compact,
    get_cts_by_dim,
    get_dim_map,
    get_segments,
)
from util.util import separate_dims


def lower_compact(env, kernel):
    n = kernel.layout.n

    dims, cs_dims = align_dimension_extents_compact(
        copy(kernel.layout.dims), copy(kernel.cs[0].layout.dims)
    )

    _, slot_dims = separate_dims(dims, n)
    _, cs_slot_dims = separate_dims(cs_dims, n)
    expanded_layout = Layout(
        kernel.cs[0].layout.term,
        kernel.cs[0].layout.rolls,
        cs_dims,
        kernel.cs[0].layout.offset,
        n,
        kernel.cs[0].layout.secret,
    )

    swaps = []
    for cs_slot_dim, slot_dim in zip(cs_slot_dims, slot_dims):
        if cs_slot_dim != slot_dim:
            swaps.append([slot_dim, cs_slot_dim])

    input_cts = env[kernel.cs[0]]
    layout_cts = LayoutCiphertexts(layout=expanded_layout, cts=input_cts.cts)
    split_slot_dim_map = get_dim_map(slot_dims)
    slot_segments = get_segments(slot_dims)

    for swap_slot in swaps:
        # perform compaction
        swap_dim = swap_slot[0]
        if swap_dim.dim is None:
            continue
        swap_dim_index = split_slot_dim_map[swap_dim]
        new_relevant_cts = {}
        ct_groups = get_cts_by_dim(layout_cts, swap_slot[0])
        for i, ct_group in enumerate(ct_groups):
            base = ct_group[0]
            for j in range(1, len(ct_group)):
                offset = j * slot_segments[swap_dim_index][2]
                rot_term = ct_group[j] << -offset
                base = base + rot_term
            new_relevant_cts[i] = base

        # update layout
        new_cs_dims = expanded_layout.dims.copy()
        new_ct_dims, new_slot_dims = separate_dims(new_cs_dims, n)
        new_ct_dims.remove(swap_dim)
        new_slot_dims[swap_dim_index] = swap_dim
        expanded_layout = Layout(
            expanded_layout.term,
            expanded_layout.rolls,
            new_ct_dims + new_slot_dims,
            expanded_layout.offset,
            n,
            expanded_layout.secret,
        )

        # update layout ciphertexts
        layout_cts = LayoutCiphertexts(layout=expanded_layout, cts=new_relevant_cts)

    return LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)
