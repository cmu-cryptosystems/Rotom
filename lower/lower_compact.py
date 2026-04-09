from copy import deepcopy as copy

from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import (
    align_dimension_extents_compact,
    align_dimension_extents_compact_skip_empty_gaps,
    dim_list_index,
    get_cts_by_dim,
    get_dim_map,
    get_segments,
)
from util.util import separate_dims


def lower_compact(env, kernel):
    """Lower a COMPACT kernel by packing non-empty slots into a smaller layout.

    Each step rotates ciphertexts so slices that differ only along a ciphertext
    axis line up in-slot, then sums them—folding that axis from ``ct_dims`` into
    ``slot_dims`` to match the target layout.

    Tabular layouts (small mismatch between target slot rank and child slot rank)
    use the original zip-based pairing on :attr:`Layout.dims`.  When internal
    gap fragmentation is coalesced in :meth:`Layout.get_dims`, aligned lists can
    have very different lengths; we then align with ``get_dims``, skip splitting
    empty (gap) extents in :func:`align_dimension_extents_compact`, and merge
    ciphertext axes in an order chosen from ``len(target_slots) - len(child_slots)``
    vs initial ciphertext rank (see below).
    """
    n = kernel.layout.n

    dims, cs_dims = align_dimension_extents_compact_skip_empty_gaps(
        copy(kernel.layout.get_dims()),
        copy(kernel.cs[0].layout.get_dims()),
    )
    _, slot_dims_probe = separate_dims(dims, n)
    _, cs_slot_probe = separate_dims(cs_dims, n)
    slot_ct_mismatch = len(slot_dims_probe) - len(cs_slot_probe)

    if slot_ct_mismatch <= 1:
        return _lower_compact_zip(env, kernel, n)

    slot_dims = slot_dims_probe
    expanded_layout = Layout(
        kernel.cs[0].layout.term,
        kernel.cs[0].layout.rolls,
        cs_dims,
        n,
        kernel.cs[0].layout.secret,
    )

    input_cts = env[kernel.cs[0]]
    layout_cts = LayoutCiphertexts(layout=expanded_layout, cts=input_cts.cts)
    slot_segments = get_segments(slot_dims)
    split_slot_dim_map = get_dim_map(slot_dims)

    merge_outer_first = slot_ct_mismatch == len(expanded_layout.ct_dims)

    while layout_cts.layout.ct_dims:
        pending = [d for d in layout_cts.layout.ct_dims if d.dim is not None]
        if not pending:
            break
        if merge_outer_first:
            swap_dim = max(pending, key=lambda d: dim_list_index(d, slot_dims))
        else:
            swap_dim = min(pending, key=lambda d: dim_list_index(d, slot_dims))

        swap_dim_index = split_slot_dim_map[swap_dim]
        new_relevant_cts = {}
        ct_groups = get_cts_by_dim(layout_cts, swap_dim)
        for i, ct_group in enumerate(ct_groups):
            base = ct_group[0]
            for j in range(1, len(ct_group)):
                offset = j * slot_segments[swap_dim_index][2]
                rot_term = ct_group[j] << -offset
                base = base + rot_term
            new_relevant_cts[i] = base

        new_cs_dims = expanded_layout.dims.copy()
        new_ct_dims, new_slot_dims = separate_dims(new_cs_dims, n)
        new_ct_dims.remove(swap_dim)
        new_slot_dims[swap_dim_index] = swap_dim
        expanded_layout = Layout(
            expanded_layout.term,
            expanded_layout.rolls,
            new_ct_dims + new_slot_dims,
            n,
            expanded_layout.secret,
        )

        layout_cts = LayoutCiphertexts(layout=expanded_layout, cts=new_relevant_cts)

    return LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)


def _lower_compact_zip(env, kernel, n):
    """Original zip-based compaction (small slot-rank mismatch)."""
    dims, cs_dims = align_dimension_extents_compact(
        copy(kernel.layout.dims), copy(kernel.cs[0].layout.dims)
    )

    _, slot_dims = separate_dims(dims, n)
    _, cs_slot_dims = separate_dims(cs_dims, n)
    expanded_layout = Layout(
        kernel.cs[0].layout.term,
        kernel.cs[0].layout.rolls,
        cs_dims,
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
        swap_dim = swap_slot[0]
        if swap_dim.dim is None:
            continue
        swap_dim_index = split_slot_dim_map[swap_dim]
        new_relevant_cts = {}
        ct_groups = get_cts_by_dim(layout_cts, swap_dim)
        for i, ct_group in enumerate(ct_groups):
            base = ct_group[0]
            for j in range(1, len(ct_group)):
                offset = j * slot_segments[swap_dim_index][2]
                rot_term = ct_group[j] << -offset
                base = base + rot_term
            new_relevant_cts[i] = base

        new_cs_dims = expanded_layout.dims.copy()
        new_ct_dims, new_slot_dims = separate_dims(new_cs_dims, n)
        new_ct_dims.remove(swap_dim)
        new_slot_dims[swap_dim_index] = swap_dim
        expanded_layout = Layout(
            expanded_layout.term,
            expanded_layout.rolls,
            new_ct_dims + new_slot_dims,
            n,
            expanded_layout.secret,
        )

        layout_cts = LayoutCiphertexts(layout=expanded_layout, cts=new_relevant_cts)

    return LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)
