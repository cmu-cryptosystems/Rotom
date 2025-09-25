from copy import deepcopy as copy

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
    cs_ct_dims, cs_slot_dims = separate_dims(cs_dims, n)

    relevant_slot_dims = cs_slot_dims
    relevant_ct_dims = cs_ct_dims

    swaps = []
    for cs_slot_dim, slot_dim in zip(relevant_slot_dims, slot_dims):
        if cs_slot_dim != slot_dim:
            swaps.append([slot_dim, cs_slot_dim])

    relevant_cts = env[kernel.cs[0]]
    split_slot_dim_map = get_dim_map(slot_dims)
    slot_segments = get_segments(slot_dims)

    for swap_slot in swaps:
        swap_dim = swap_slot[0]
        if swap_dim.dim is None:
            continue
        swap_dim_index = split_slot_dim_map[swap_dim]
        new_relevant_cts = {}
        ct_groups = get_cts_by_dim(relevant_cts, relevant_ct_dims, swap_slot[0])
        for i, ct_group in enumerate(ct_groups):
            base = ct_group[0]
            for j in range(1, len(ct_group)):
                offset = j * slot_segments[swap_dim_index][2]
                rot_term = ct_group[j] << -offset
                base = base + rot_term
            new_relevant_cts[i] = base

        relevant_cts = new_relevant_cts
        relevant_slot_dims[swap_dim_index] = swap_dim
        relevant_ct_dims.remove(swap_dim)
    return relevant_cts
