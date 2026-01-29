from copy import deepcopy as copy

from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts
from lower.lower_util import rotate_and_sum
from util.layout_util import get_cts_by_dim, get_dim_indices, get_segments, mul_vec
from util.util import get_mask_from_segment


def lower_conversion(env, kernel):
    n = kernel.layout.n

    # create CS cts
    input_cts = env[kernel.cs[2]]
    cts = [
        HETerm(HEOp.CS, [ct], kernel.cs[2].layout.secret) for ct in input_cts.values()
    ]

    # get conversion metadata
    split_dims = kernel.cs[0]

    # figure out which split dimensions are ct vs slot dimensions
    split_slot_dims = []
    split_ct_dims = []
    total_extent = 1
    for dim in split_dims[::-1]:
        total_extent *= dim.extent
        if total_extent <= n:
            split_slot_dims.insert(0, dim)
        else:
            split_ct_dims.insert(0, dim)

    # figure out target dimensions to convert to
    target_dims = kernel.cs[1]
    target_slot_dims = []
    target_ct_dims = []
    total_extent = 1
    for dim in target_dims[::-1]:
        total_extent *= dim.extent
        if total_extent <= n:
            target_slot_dims.insert(0, dim)
        else:
            target_ct_dims.insert(0, dim)

    # categorize the types of conversions
    slot_slot = []
    slot_ct = []
    ct_slot = []
    ct_ct = []
    for i, split_ct_dim in enumerate(split_ct_dims):
        if split_ct_dim in target_ct_dims and i != target_ct_dims.index(split_ct_dim):
            ct_ct.append((split_ct_dim, target_ct_dims.index(split_ct_dim)))
        elif split_ct_dim in target_slot_dims:
            ct_slot.append((split_ct_dim, target_slot_dims.index(split_ct_dim)))

    for i, split_slot_dim in enumerate(split_slot_dims):
        if split_slot_dim in target_slot_dims and i != target_slot_dims.index(
            split_slot_dim
        ):
            slot_slot.append((split_slot_dim, target_slot_dims.index(split_slot_dim)))
        elif split_slot_dim in target_ct_dims:
            slot_ct.append((split_slot_dim, target_ct_dims.index(split_slot_dim)))

    # get ct and slot segments
    ct_segments = get_segments(split_ct_dims)
    slot_segments = get_segments(split_slot_dims)

    relevant_slot_dims = []
    l = 1
    for dim in kernel.cs[0][::-1]:
        if l < kernel.layout.n:
            l *= dim.extent
            relevant_slot_dims.insert(0, copy(dim))
        else:
            break

    # get non-repeated ct_dims
    num_ct = kernel.layout.num_ct()
    relevant_ct_indices = [0] * num_ct
    relevant_ct_dims = []
    split_ct_dim_map = {}
    for i, split_dim in enumerate(split_ct_dims):
        split_ct_dim_map[split_dim] = i
    for ct_dim in split_ct_dims:
        if ct_dim.dim is None:
            continue
        relevant_ct_dims.append(ct_dim)
        i_len = ct_segments[split_ct_dim_map[ct_dim]][0]
        j_len = ct_segments[split_ct_dim_map[ct_dim]][1]
        k_len = ct_segments[split_ct_dim_map[ct_dim]][2]
        indices = []
        for i in range(i_len):
            for j in range(j_len):
                for _ in range(k_len):
                    indices.append(j * k_len)
        relevant_ct_indices = [a + b for a, b in zip(relevant_ct_indices, indices)]

    relevant_ct_indices = list(set(relevant_ct_indices))
    relevant_ct_indices.sort()
    # Track only the relevant ciphertexts (those that participate in the conversion)
    relevant_cts = [cts[index] for index in relevant_ct_indices]

    # decompaction, slot to ct transformations
    if slot_ct or slot_slot:
        # decompaction
        swaps = slot_ct + slot_slot
        for swap_slot in swaps:
            swap_dim = swap_slot[0]
            swap_dim_index = relevant_slot_dims.index(swap_dim)

            # update layout tracker
            relevant_ct_dims.append(swap_dim)
            relevant_slot_dims[swap_dim_index].dim = None
            relevant_slot_dims[swap_dim_index].dim_type = DimType.EMPTY
            mask = get_mask_from_segment(slot_segments[swap_dim_index])
            masked_cts = []
            for ct in relevant_cts:
                for i in range(slot_segments[swap_dim_index][1]):
                    masked_cts.append(
                        HETerm(
                            HEOp.ROT,
                            [
                                ct,
                                i * slot_segments[swap_dim_index][2],
                            ],
                            ct.secret,
                            f"rot: {i * slot_segments[swap_dim_index][2]}\ndecompaction: {swap_slot} {kernel.cs[2].layout.layout_str()} => {relevant_ct_dims};{relevant_slot_dims}",
                        )
                        * HETerm.mask([mask]),
                    )
            # update relevant_cts
            relevant_cts = masked_cts

    # optimize decompaction chain
    updated_cts = []
    for ct in relevant_cts:
        cs_term = None
        rot_amt = 0
        mask = [1] * n
        for c in ct.post_order():
            if c.op == HEOp.CS:
                cs_term = c
            elif c.op == HEOp.ROT:
                rot_amt += c.cs[1]
            elif c.op == HEOp.MASK:
                mask = mul_vec(mask, c.cs[0])
        updated_cts.append((cs_term << rot_amt) * HETerm.mask([mask]))
    relevant_cts = updated_cts

    if ct_slot or slot_slot:
        # compaction
        swaps = ct_slot + slot_slot

        # Build an initial layout that matches the relevant ciphertexts we are tracking:
        # ct dimensions are the non-repeated relevant_ct_dims, followed by the slot dims.
        expanded_layout = Layout(
            kernel.cs[2].layout.term,
            kernel.cs[2].layout.rolls,
            relevant_ct_dims + relevant_slot_dims,
            n,
            kernel.cs[2].layout.secret,
        )
        layout_cts = LayoutCiphertexts(
            layout=expanded_layout,
            cts={i: ct for i, ct in enumerate(relevant_cts)},
        )

        for swap_slot in swaps:
            swap_dim = swap_slot[0]
            if swap_dim.dim is None:
                # Nothing to compact for purely empty dimensions
                continue

            swap_to_dim_index = swap_slot[1]

            # Use the current LayoutCiphertexts to group ciphertexts by the swap dimension
            ct_groups = get_cts_by_dim(layout_cts, swap_dim)

            # Perform compaction by rotating and adding ciphertexts within each group
            new_relevant_cts_list = []
            for ct_group in ct_groups:
                base = ct_group[0]
                for i in range(1, len(ct_group)):
                    offset = i * slot_segments[swap_to_dim_index][2]
                    rot_term = HETerm(
                        HEOp.ROT,
                        [ct_group[i], -offset],
                        ct_group[i].secret,
                        "compaction",
                    )
                    base = base + rot_term
                new_relevant_cts_list.append(base)

            # Update tracked ciphertexts
            relevant_cts = new_relevant_cts_list
            new_cts_dict = {i: ct for i, ct in enumerate(relevant_cts)}

            # Update layout bookkeeping to reflect the move of swap_dim from ct to slot
            if relevant_slot_dims[swap_to_dim_index].extent == swap_dim.extent:
                relevant_slot_dims[swap_to_dim_index] = swap_dim
                relevant_ct_dims.remove(swap_dim)
            elif relevant_slot_dims[swap_to_dim_index].extent > swap_dim.extent:
                split_dim = copy(relevant_slot_dims[swap_to_dim_index])
                split_dim.extent //= swap_dim.extent
                relevant_slot_dims[swap_to_dim_index] = split_dim
                relevant_slot_dims.insert(swap_to_dim_index + 1, swap_dim)
                relevant_ct_dims.remove(swap_dim)
            else:
                raise NotImplementedError("sizes mismatch")

            # Keep the LayoutCiphertexts in sync with these dimension/layout changes
            expanded_layout = Layout(
                layout_cts.layout.term,
                layout_cts.layout.rolls,
                relevant_ct_dims + relevant_slot_dims,
                n,
                layout_cts.layout.secret,
            )
            layout_cts = LayoutCiphertexts(layout=expanded_layout, cts=new_cts_dict)

    # find places for slot replication
    slot_dims = kernel.layout.slot_dims
    for i, slot_dim in enumerate(relevant_slot_dims):
        if slot_dim.dim_type == DimType.EMPTY and slot_dims[i].dim_type == DimType.FILL:
            new_relevant_cts = []
            for ct in relevant_cts:
                base_term = ct
                extent = slot_segments[i][1]
                mul_offset = slot_segments[i][2]
                base_term = rotate_and_sum(base_term, extent, mul_offset, True)
                new_relevant_cts.append(base_term)
            relevant_cts = new_relevant_cts
            relevant_slot_dims[i].dim_type = DimType.FILL

    relevant_ct_dim_extent_map = {}
    relevant_ct_segments = get_segments(relevant_ct_dims)
    for i, ct_dim in enumerate(relevant_ct_dims):
        relevant_ct_dim_extent_map[ct_dim] = relevant_ct_segments[i][2]

    ct_dims = kernel.layout.ct_dims
    ct_dim_indices = get_dim_indices(ct_dims)
    ct_indices = [0] * kernel.layout.num_ct()
    for ct_dim, ct_dim_index in zip(ct_dims, ct_dim_indices):
        if ct_dim not in relevant_ct_dim_extent_map:
            continue
        extent = relevant_ct_dim_extent_map[ct_dim]

        for i in range(len(ct_dim_index)):
            ct_indices[i] += ct_dim_index[i] * extent

    cts = {}
    for i, ct_index in enumerate(ct_indices):
        cts[i] = relevant_cts[ct_index]

    assert kernel.layout.num_ct() == len(cts)
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
