import math
from copy import deepcopy as copy

import numpy as np

from frontends.tensor import TensorOp
from ir.dim import Dim, DimType


def swap_rolls(layout, roll):
    # swap rolls
    new_rolls = []
    for base_roll in layout.rolls:
        if base_roll == roll:
            from ir.roll import Roll

            new_rolls.append(Roll(roll.dim_to_roll_by, roll.dim_to_roll))
        elif base_roll.dim_to_roll == roll.dim_to_roll:
            from ir.roll import Roll

            new_rolls.append(Roll(roll.dim_to_roll_by, base_roll.dim_to_roll_by))
        else:
            new_rolls.append(base_roll.copy())

    # swap dimensions
    dims = layout.dims
    new_dims = copy(dims)
    new_dims[roll.dim_to_roll] = dims[roll.dim_to_roll_by].copy()
    new_dims[roll.dim_to_roll_by] = dims[roll.dim_to_roll].copy()

    # create new layout
    from ir.layout import Layout

    eq_layout = Layout(layout.term, new_rolls, new_dims, layout.n, layout.secret)
    return eq_layout


def prod(lst):
    i = 1
    for num in lst:
        i *= num
    return i


def find_unique_dims(layout):
    dims = set()
    for dim in layout.dims:
        if dim.dim is not None:
            dims.add(dim.dim)
    return len(dims)


def find_n(layout):
    return prod([dim.extent for dim in layout.slot_dims])


def zero_mask(n):
    return [0] * n


def equivalent_layouts(kernel):
    layout = kernel.layout
    dims = layout.get_dims()

    # 1. create roll map
    roll_map = {}
    for roll in layout.rolls:
        if roll.dim_to_roll not in roll_map:
            roll_map[roll.dim_to_roll] = set()
        roll_map[roll.dim_to_roll].add(roll.dim_to_roll_by)

    eq_layouts = set()
    queue = [layout]
    while queue:
        next_queue = []
        for layout in queue:
            if layout in eq_layouts:
                continue
            eq_layouts.add(layout)
            for roll in layout.rolls:
                if dims[roll.dim_to_roll_by].dim is None:
                    next_queue.append(swap_rolls(layout, roll))

        queue = next_queue
    return eq_layouts


def convert_layout_to_indices(layout):
    # HACK assume that tensors are only 2 dimensions
    n = find_n(layout)
    unique_dims = find_unique_dims(layout)
    if unique_dims == 1:
        slot_indices = [[0] for _ in range(n)]
        slot_offset = 1
        dim_offset = {
            0: 1,
        }
    elif unique_dims == 2:
        slot_indices = [[0, 0] for _ in range(n)]
        slot_offset = 1
        dim_offset = {
            0: 1,
            1: 1,
        }
    else:
        raise NotImplementedError("other tensor dimensions are not supported")

    for dim in layout.slot_dims[::-1]:
        for i in range(n // (dim.extent * slot_offset)):
            for j in range(dim.extent):
                for k in range(slot_offset):
                    if dim.dim is not None and dim.dim_type == DimType.FILL:
                        index = i * (dim.extent * slot_offset) + j * slot_offset + k
                        if slot_indices[index][dim.dim] is not None:
                            slot_indices[
                                (i * (dim.extent * slot_offset) + j * slot_offset + k)
                            ][dim.dim] += (j * dim.stride)
                    elif dim.dim_type == DimType.EMPTY:
                        for key in dim_offset.keys():
                            if j:
                                slot_indices[
                                    (
                                        i * (dim.extent * slot_offset)
                                        + j * slot_offset
                                        + k
                                    )
                                ][key] = None
        if dim.dim is not None and dim.dim_type == DimType.FILL:
            dim_offset[dim.dim] *= dim.extent
        slot_offset *= dim.extent

    indices = [slot_indices]
    if layout.ct_dims:
        for dim in layout.ct_dims[::-1]:
            new_indices = []
            for i in range(dim.extent):
                ct_indices = copy(indices)
                for slot_indices in ct_indices:
                    if dim.dim is not None:
                        for j in range(n):
                            slot_indices[j][dim.dim] += i * dim.stride
                    new_indices.append(slot_indices)
            indices = new_indices

    if layout.rolls:
        dim_indices = convert_dim_to_base_indices(layout)

        for roll in layout.rolls:
            dim_order = [dim.dim for dim in layout.dims]
            # 3. apply roll to kernel indices
            counter = 0
            for ct in indices:
                for index in ct:
                    if index[0] is not None:
                        index[dim_order[roll.dim_to_roll]] = (
                            index[dim_order[roll.dim_to_roll]]
                            + dim_indices[roll.dim_to_roll_by][counter]
                        ) % layout.dims[dim_order[roll.dim_to_roll]].extent
                    counter += 1
    return indices


def convert_dim_to_base_indices(layout):
    # HACK assume that tensors are only 2 dimensions
    layout_len = len(layout)

    dim_indices = []
    slot_offset = 1
    for dim in (layout.ct_dims + layout.slot_dims)[::-1]:
        indices = [None] * layout_len
        for i in range(layout_len // (dim.extent * slot_offset)):
            for j in range(dim.extent):
                for k in range(slot_offset):
                    indices[i * (dim.extent * slot_offset) + j * slot_offset + k] = j
        slot_offset *= dim.extent
        dim_indices.insert(0, indices)
    return dim_indices


def convert_to_row(layout, values):
    ct_indices = convert_layout_to_indices(layout)
    dim_extents = {}
    for dim in layout.get_dims():
        if dim.dim is not None:
            if dim.dim not in dim_extents:
                dim_extents[dim.dim] = 1
            dim_extents[dim.dim] *= dim.extent

    row_flattened = []
    row_indices = []
    for indices in ct_indices:
        row_indices += indices

    index_map = {}
    for i, ct in enumerate(ct_indices):
        for j, index in enumerate(ct):
            index_map[tuple(index)] = i * len(ct) + j

    for ct in ct_indices:
        if len(dim_extents) == 1:
            for x in range(dim_extents[0]):
                row_flattened.append(values[index_map[(x,)]])
        elif len(dim_extents) == 2:
            for x in range(dim_extents[0]):
                for y in range(dim_extents[1]):
                    row_flattened.append(values[index_map[(x, y)]])
        else:
            raise NotImplementedError(f"len of dim extents: {len(dim_extents)}")
    return row_flattened


def layout_dim_map(layout):
    dim_map = {}
    for i, dim in enumerate(layout.dims):
        dim_map[i] = dim
    return dim_map


def ct_slot_dim_maps(layout):
    ct_dim_map = {}
    for i, dim in enumerate(layout.ct_dims):
        ct_dim_map[dim] = i
    slot_dim_map = {}
    for i, dim in enumerate(layout.slot_dims):
        slot_dim_map[i + len(ct_dim_map)] = dim
    return ct_dim_map, slot_dim_map


def base_dim_segments(dims):
    layout_len = prod([dim.extent for dim in dims])
    segments = {}
    for i in range(len(dims)):
        segment_len = int(prod([dim.extent for dim in dims[i + 1 :]]))
        extent = dims[i].extent
        count = layout_len // segment_len // extent
        segments[i] = [count, extent, segment_len]
    return segments


def get_mask_from_segment(segment):
    i_len = segment[0]
    j_len = segment[1]
    k_len = segment[2]
    indices = []
    for _ in range(i_len):
        for j in range(j_len):
            for _ in range(k_len):
                if not j:
                    indices.append(1)
                else:
                    indices.append(0)
    return indices


def get_dim_map_by_dim(dims):
    dim_map = {}
    for dim in dims:
        if dim.dim not in dim_map:
            dim_map[dim.dim] = []
        dim_map[dim.dim].append(dim)
    return dim_map


def next_non_empty_dim(dims):
    for dim in dims:
        if dim.dim is not None:
            return dim
    return None


def split_dim(dim, extent):
    assert dim.extent > extent
    rem_extent = dim.extent // extent
    split_1 = Dim(dim.dim, extent, rem_extent * dim.stride, dim.dim_type)
    split_2 = Dim(dim.dim, rem_extent, dim.stride, dim.dim_type)
    return split_1, split_2


def separate_dims(dims, n):
    total_extent = 1
    ct_dims = []
    slot_dims = []
    for dim in dims[::-1]:
        if dim.extent * total_extent <= n:
            slot_dims.insert(0, dim)
        else:
            ct_dims.insert(0, dim)
        total_extent *= dim.extent
    return ct_dims, slot_dims


def count_ops(circuit_ir):
    ops = {}
    for _, ct in circuit_ir.items():
        for term in ct.post_order():
            if term.op not in ops:
                ops[term.op] = 0
            ops[term.op] += 1
    return ops


def max_levels(circuit_ir):
    max_level = 0
    for _, ct in circuit_ir.items():
        for term in ct.post_order():
            max_level = max(max_level, term.level)
    return max_level


def round_up(n):
    return int(np.ceil(n))


def round_to_ceiling_power_of_2(n):
    if n < 1:
        raise ValueError("Input must be a positive number.")
    return 1 if n == 1 else 2 ** math.ceil(math.log2(n))


def get_slot_dims(dims, n):
    extent = 1
    ct_dims = []
    slot_dims = []
    for dim in dims[::-1]:
        extent *= dim.extent
        if extent <= n:
            slot_dims.insert(0, dim)
        else:
            ct_dims.insert(0, dim)
    return ct_dims, slot_dims


def get_sum_dim(term, kernel):
    match term.op:
        case TensorOp.MATMUL:
            if kernel.layout.term == term.cs[0]:
                return max(
                    dim.dim for dim in kernel.layout.get_dims() if dim.dim is not None
                )
            else:
                return min(
                    dim.dim for dim in kernel.layout.get_dims() if dim.dim is not None
                )
        case TensorOp.BLOCK_MATMUL:
            if kernel.layout.term == term.cs[0]:
                return 2
            else:
                return 1


def split_lists(lst, num_sublists):
    if len(lst) % num_sublists != 0:
        raise ValueError("The list length is not divisible by the number of sublists.")
    sublist_length = len(lst) // num_sublists
    return [
        lst[i * sublist_length : (i + 1) * sublist_length] for i in range(num_sublists)
    ]
