from copy import copy as copy

import numpy as np
from _pytest._py.error import R

from ir.dim import Dim, DimType
from ir.layout import Layout
from ir.roll import Roll
from util.util import prod, split_dim


def transpose(l):
    return list(map(list, zip(*l)))


def swap_rolls(layout, roll):
    # swap rolls
    new_rolls = []
    for base_roll in layout.rolls:
        if base_roll == roll:
            new_rolls.append(Roll(roll.dim_to_roll_by, roll.dim_to_roll))
        elif base_roll.dim_to_roll == roll.dim_to_roll:
            new_rolls.append(Roll(roll.dim_to_roll_by, base_roll.dim_to_roll_by))
        else:
            new_rolls.append(base_roll.copy())

    # swap dimensions
    dims = layout.dims
    new_dims = copy(dims)
    new_dims[roll.dim_to_roll] = dims[roll.dim_to_roll_by].copy()
    new_dims[roll.dim_to_roll_by] = dims[roll.dim_to_roll].copy()

    # create new layout
    eq_layout = Layout(layout.term, new_rolls, new_dims, layout.n, layout.secret)
    return eq_layout


def merge_dims(dims, in_roll):
    """Given a list of dimensions, try and merge the dimensions"""
    merged_dims = []
    if not dims:
        return merged_dims

    base_dim = copy(dims[0])
    for next_dim in dims[1:]:
        # don't merge dimensions that are used in rolls
        if base_dim in in_roll or next_dim in in_roll:
            merged_dims.append(base_dim)
            base_dim = next_dim
            continue
        if (
            base_dim.dim == next_dim.dim
            and base_dim.dim_type == next_dim.dim_type
            and base_dim.stride == next_dim.extent * next_dim.stride
        ):
            base_dim.extent *= next_dim.extent
            base_dim.stride = next_dim.stride
        elif (
            base_dim.dim is None
            and next_dim.dim is None
            and base_dim.dim_type == next_dim.dim_type
        ):
            base_dim.extent *= next_dim.extent
            base_dim.stride = 1
        else:
            merged_dims.append(base_dim)
            base_dim = next_dim
    merged_dims.append(base_dim)
    return merged_dims


def dimension_merging(layout):
    """Merge similar dimensions, filled or empty, together
    HACK: does not adjust rollutations on the layout
    QUESTION: can rolls cross the slot, ct dimensions?
    """
    merged_dims = []

    # find original rollutation indices
    roll_map = {}
    in_roll = set()
    for roll in layout.rolls:
        if roll.dim_to_roll not in roll_map:
            roll_map[roll.dim_to_roll] = []
        roll_map[roll.dim_to_roll].append(roll.dim_to_roll_by)
        in_roll.add(roll.dim_to_roll)
        in_roll.add(roll.dim_to_roll_by)

    # filter for only filled ct_dims
    # or if the dimension is in a roll
    layout_ct_dims = [
        copy(dim)
        for dim in layout.ct_dims
        if dim.dim_type == DimType.FILL or dim in in_roll
    ]

    merged_ct_dims = merge_dims(copy(layout_ct_dims), in_roll)
    merged_dims += merged_ct_dims

    # merge slot dims
    merged_slot_dims = merge_dims(
        [copy(slot_dim) for slot_dim in layout.slot_dims], in_roll
    )
    merged_dims += merged_slot_dims
    new_rolls = []
    for k, v in roll_map.items():
        for r in v:
            assert k in merged_dims
            assert r in merged_dims
            new_rolls.append(Roll(k, r))

    return Layout(layout.term, new_rolls, merged_dims, layout.n, layout.secret)


def align_dimension_extents(a_dims, b_dims):
    a_dim_len = prod([a_dim.extent for a_dim in a_dims])
    b_dim_len = prod([b_dim.extent for b_dim in b_dims])
    if a_dim_len > b_dim_len:
        b_dims = [Dim(None, a_dim_len // b_dim_len, 1, DimType.EMPTY)] + b_dims
    elif a_dim_len < b_dim_len:
        a_dims = [Dim(None, b_dim_len // a_dim_len, 1, DimType.EMPTY)] + a_dims

    a_dim_len = prod([a_dim.extent for a_dim in a_dims])
    b_dim_len = prod([b_dim.extent for b_dim in b_dims])
    assert a_dim_len == b_dim_len

    # find smallest dimension extent
    # min extent is set to 2
    min_extent = 2

    split_a = []
    for a_dim in a_dims:
        while a_dim.extent > min_extent:
            split_1, split_2 = split_dim(a_dim, min_extent)
            split_a.append(split_1)
            a_dim = split_2
        split_a.append(a_dim)

    split_b = []
    for b_dim in b_dims:
        while b_dim.extent > min_extent:
            split_1, split_2 = split_dim(b_dim, min_extent)
            split_b.append(split_1)
            b_dim = split_2
        split_b.append(b_dim)

    return split_a, split_b


def get_extent_dims(dims):
    # find smallest dimension extent
    # min extent is set to 2
    min_extent = 2

    split = []
    for dim in dims:
        while dim.extent > min_extent:
            split_1, split_2 = split_dim(dim, min_extent)
            split.append(split_1)
            dim = split_2
        split.append(dim)
    return split


def align_dimension_extents_compact(a_dims, b_dims):
    a_dim_len = prod(
        [a_dim.extent for a_dim in a_dims if a_dim.dim_type == DimType.FILL]
    )
    b_dim_len = prod(
        [b_dim.extent for b_dim in b_dims if b_dim.dim_type == DimType.FILL]
    )
    assert a_dim_len == b_dim_len

    # find smallest dimension extent
    # min extent is set to 2
    min_extent = 2

    split_a = []
    for a_dim in a_dims:
        while a_dim.extent > min_extent:
            split_1, split_2 = split_dim(a_dim, min_extent)
            split_a.append(split_1)
            a_dim = split_2
        split_a.append(a_dim)

    split_b = []
    for b_dim in b_dims:
        while b_dim.extent > min_extent:
            split_1, split_2 = split_dim(b_dim, min_extent)
            split_b.append(split_1)
            b_dim = split_2
        split_b.append(b_dim)
    return split_a, split_b


def match_dims(dims, to_match):
    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0

    dims = dims[::-1]
    to_match = to_match[::-1]

    matched_dims = []
    replicated_stride = 1
    while a_next < len(dims) and b_next < len(to_match):
        a_dim = dims[a_next]
        b_dim = to_match[b_next]
        # both emtpy dimensions, skip
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        a_extent = prod([dim.extent for dim in dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in to_match[b_start : b_next + 1]])
        if a_extent == b_extent:
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
            dim = copy(b_dim)
            dim.dim = a_dim.dim
            if a_dim.dim is None:
                dim.stride = replicated_stride
                replicated_stride *= dim.extent
            else:
                dim.stride = a_dim.extent // b_dim.extent * a_dim.stride
            dim.dim_type = a_dim.dim_type
            matched_dims.insert(0, dim)
        elif a_extent > b_extent:
            dim = copy(b_dim)
            dim.dim = a_dim.dim
            dim.stride = a_dim.stride
            dim.dim_type = a_dim.dim_type
            matched_dims.insert(0, dim)
            b_next += 1
        else:
            a_next += 1
    return matched_dims


def index_layout(layout, indexing_map, cts):
    dim_indices = get_dim_indices(layout.ct_dims)

    # group indices together
    grouped_dim_indices = {}
    for i in range(len(cts)):
        grouped_dim_indices[i] = []
    for indices in dim_indices:
        for i, index in enumerate(indices):
            grouped_dim_indices[i].append(index)

    # map values to dim_indices:
    for i, (dim, index) in enumerate(zip(layout.ct_dims, dim_indices)):
        keys_to_remove = []
        if dim in indexing_map:
            for k, v in grouped_dim_indices.items():
                if v[i] != indexing_map[dim]:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del grouped_dim_indices[k]

    # get corresponding ct
    ct_indices = sorted(list(grouped_dim_indices.keys()))
    indexed_cts = [cts[i] for i in ct_indices]

    indexed_cts_dims = copy(layout.ct_dims)
    for dim in indexing_map:
        indexed_cts_dims.remove(dim)
    return indexed_cts, indexed_cts_dims


def convert_layout_to_mask(layout):
    slot_dims = layout.slot_dims
    slot_dim_indices = get_dim_indices(slot_dims)
    mask = [1] * layout.n
    for dim, indices in zip(slot_dims, slot_dim_indices):
        if dim.dim_type != DimType.EMPTY:
            continue
        for i, index in enumerate(indices):
            if index != 0:
                mask[i] = 0
    return mask


def convert_layout_to_stride_mask(layout, h_o, w_o, stride):
    """Mask for conv2d stride > 1: zero slots where (dim1,dim2) is not a valid output position.
    Keep only slots where dim1 % stride == 0, dim2 % stride == 0,
    dim1 < h_o*stride, dim2 < w_o*stride. Layout uses input spatial size (h_i, w_i)."""
    indices_by_dim = get_dim_indices_by_dim(layout.slot_dims)
    mask = [1] * layout.n
    dim1_indices = indices_by_dim.get(1)
    dim2_indices = indices_by_dim.get(2)
    if dim1_indices is not None:
        for i, idx in enumerate(dim1_indices):
            if idx is not None and (idx % stride != 0 or idx >= h_o * stride):
                mask[i] = 0
    if dim2_indices is not None:
        for i, idx in enumerate(dim2_indices):
            if idx is not None and (idx % stride != 0 or idx >= w_o * stride):
                mask[i] = 0
    return mask


def layout_to_ct_indices(layout):
    dims = layout.ct_dims
    dim_indices = get_dim_indices(dims)

    # keep only the pertinent dims
    indices_map = {}
    for i, dim in enumerate(dims):
        if dim.dim is not None:
            dim_indices[i] = mul(dim_indices[i], dim.stride)

            if dim.dim not in indices_map:
                indices_map[dim.dim] = dim_indices[i]
            else:
                indices_map[dim.dim] = add_vec(indices_map[dim.dim], dim_indices[i])

    if len(indices_map) == 1:
        dim = [dim.dim for dim in dims if dim.dim is not None][0]
        return indices_map[dim]
    elif len(indices_map) == 2:
        keys = sorted(indices_map.keys())
        return [(a, b) for a, b in zip(indices_map[keys[0]], indices_map[keys[1]])]
    else:
        raise NotImplementedError("more than 2 dimensions")


def get_dim_map(dims):
    dim_map = {}
    for i, dim in enumerate(dims):
        dim_map[dim] = i
    return dim_map


def get_segments(dims):
    n = 1
    for dim in dims:
        n *= dim.extent

    segments = {}
    for i in range(len(dims)):
        segment_len = int(prod([dim.extent for dim in dims[i + 1 :]]))
        extent = dims[i].extent
        count = n // segment_len // extent
        segments[i] = [count, extent, segment_len]
    return segments


def get_segment(dim, dims):
    segments = get_segments(dims)
    dim_map = get_dim_map(dims)
    return segments[dim_map[dim]]


def get_cts_by_dim(layout_cts, dim):
    """Get ciphertext groups by dimension from LayoutCiphertexts.

    Args:
        layout_cts: LayoutCiphertexts object containing layout and ciphertexts
        dim: Dim object to group by

    Returns:
        List of lists of HETerm objects, grouped by the dimension
    """
    # Access cts and layout from LayoutCiphertexts object
    cts = layout_cts.cts
    ct_dims = layout_cts.layout.ct_dims
    assert dim in ct_dims
    ct_dim_map = get_dim_map(ct_dims)
    ct_dim_index = ct_dim_map[dim]

    groups = []
    dim_indices = get_dim_indices(ct_dims)
    indices = dim_indices[ct_dim_index]
    while any(i is not None for i in indices):
        group = []
        for i in range(dim.extent):
            for j in range(len(indices)):
                if indices[j] == i:
                    group.append(j)
                    indices[j] = None
                    break
        groups.append(group)

    ct_groups = []
    for group in groups:
        ct_group = []
        for g in group:
            ct_group.append(cts[g])
        ct_groups.append(ct_group)
    return ct_groups


def get_ct_idxs_by_dim(ct_dims, dim):
    assert dim in ct_dims
    ct_dim_map = get_dim_map(ct_dims)
    ct_dim_index = ct_dim_map[dim]

    groups = []
    dim_indices = get_dim_indices(ct_dims)
    indices = dim_indices[ct_dim_index]
    while any(i is not None for i in indices):
        group = []
        for i in range(dim.extent):
            for j in range(len(indices)):
                if indices[j] == i:
                    group.append(j)
                    indices[j] = None
                    break
        groups.append(group)

    ct_groups = []
    for group in groups:
        ct_group = []
        for g in group:
            ct_group.append(g)
        ct_groups.append(ct_group)
    return ct_groups


def get_dim_indices(dims):
    segments = get_segments(dims)
    all_indices = []
    for i in range(len(dims)):
        segment = segments[i]
        i_len = segment[0]
        j_len = segment[1]
        k_len = segment[2]
        indices = []
        for _ in range(i_len):
            for j in range(j_len):
                for _ in range(k_len):
                    indices.append(j)
        all_indices.append(indices)
    return all_indices


def get_dim_indices_by_dim(dims):
    dim_indices = get_dim_indices(dims)
    dim_map = {}
    for dim, dim_index in zip(dims, dim_indices):
        dim_index = mul(dim_index, dim.stride)
        if dim.dim not in dim_map:
            dim_map[dim.dim] = dim_index
        else:
            dim_map[dim.dim] = add_vec(dim_map[dim.dim], dim_index)
    return dim_map


def add_vec(a, b):
    return [x + y if x is not None and y is not None else None for x, y in zip(a, b)]


def add_vecs(a, b):
    result = []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            result.append(x + y)
        elif x is not None:
            result.append(x)
        elif y is not None:
            result.append(y)
        else:
            result.append(None)
    return result


def add_vecs_of_vecs(a, b):
    return [add_vecs(x, y) for x, y in zip(a, b)]


def mul_vec(a, b):
    return [x * y if x is not None and y is not None else None for x, y in zip(a, b)]


def add(vec, n):
    return [v + n if v is not None else None for v in vec]


def mul(vec, n):
    return [v * n if v is not None else None for v in vec]


def apply_layout(pt_tensor, layout):
    """apply a layout to a pt tensor"""
    layout_len = max(len(layout), layout.n)
    # get base_term indices
    dims = layout.get_dims()
    dim_indices = get_dim_indices(dims)

    # apply any base term rolls
    for roll in layout.rolls:
        roll_index = roll.roll_index(dims)
        dim_indices[roll_index[0]] = [
            (dim_indices[roll_index[0]][i] + dim_indices[roll_index[1]][i])
            % roll.dim_to_roll.extent
            for i in range(layout_len)
        ]

    # update dim with strides
    for i in range(len(dim_indices)):
        dim_indices[i] = mul(dim_indices[i], dims[i].stride)

    # update dims
    for i in range(len(dims)):
        print("dims:", dims[i].dim, dims[i].dim_type)
        if dims[i].dim_type == DimType.EMPTY:
            dim_indices[i] = [j if not j else None for j in dim_indices[i]]
    print("dims:", dims)
    print("len dims:", len(dims))
    print("len dim_indices:", len(dim_indices))
    if len(dims) == 3:
        print("dim_indices[0]:", dim_indices[0])
        print("dim_indices[1]:", dim_indices[1])
        print("dim_indices[2]:", dim_indices[2])

    # split indices by dimensions:
    indices_map = {}
    for i, dim in enumerate(dims):
        if dim.dim is not None:
            if dim.dim in indices_map:
                indices_map[dim.dim] = add_vec(indices_map[dim.dim], dim_indices[i])
            else:
                indices_map[dim.dim] = dim_indices[i]

    for i, dim in enumerate(dims):
        # used for adding gap slots into layout
        if dim.dim is None and dim.dim_type == DimType.EMPTY:
            for pertinent_dim in indices_map:
                indices_map[pertinent_dim] = add_vec(
                    indices_map[pertinent_dim], dim_indices[i]
                )

    # map to pertinent dimensions
    base_indices = [[0] * (max(indices_map) + 1) for _ in range(layout_len)]
    for dim, indices in indices_map.items():
        for i, index in enumerate(indices):
            base_indices[i][dim] = index

    # split by cts
    base_indices_by_cts = [
        base_indices[i * layout.n : (i + 1) * layout.n]
        for i in range((layout_len // layout.n))
    ]

    print("base_indices_by_cts:", base_indices_by_cts)

    # combine cts if ct_dim is a gap dimension
    combined_cts = copy(base_indices_by_cts)
    ct_dims = copy(layout.ct_dims)
    for ct_dim in layout.ct_dims:
        if ct_dim.dim_type == DimType.EMPTY:
            new_combined_cts = []
            ct_indices = get_ct_idxs_by_dim(ct_dims, ct_dim)
            for indices in ct_indices:
                base = base_indices_by_cts[indices[0]]
                for index in indices[1:]:
                    base = add_vecs_of_vecs(base, base_indices_by_cts[index])
                new_combined_cts.append(base)
            ct_dims.remove(ct_dim)
            combined_cts = new_combined_cts
    base_indices_by_cts = combined_cts

    # Get the actual tensor dimensionality
    pt_tensor_ndim = np.ndim(pt_tensor)
    print("pt_tensor:", pt_tensor)
    print("pt_tensor_shape:", pt_tensor.shape)
    print("layout:", layout)

    cts = []
    for ct_index in range(len(base_indices_by_cts)):
        ct_indices = base_indices_by_cts[ct_index]
        ct = []

        for index in ct_indices:
            # Check if any required index is None
            if any(index[i] is None for i in range(pt_tensor_ndim)):
                ct.append(0)
                continue

            if any(index[i] >= pt_tensor.shape[i] for i in range(pt_tensor_ndim)):
                ct.append(0)
                continue

            # Access tensor using the appropriate number of indices
            if pt_tensor_ndim == 0:
                value = pt_tensor.item()
            elif pt_tensor_ndim == 1:
                value = pt_tensor[index[0]]
            elif pt_tensor_ndim == 2:
                value = pt_tensor[index[0]][index[1]]
            elif pt_tensor_ndim == 3:
                value = pt_tensor[index[0]][index[1]][index[2]]
            elif pt_tensor_ndim == 4:
                value = pt_tensor[index[0]][index[1]][index[2]][index[3]]
            else:
                raise NotImplementedError(
                    f"tensors with {pt_tensor_ndim} dimensions are not supported"
                )

            # Convert single-element arrays to scalars
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.size == 1:
                value = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 0:
                value = value.item()
            ct.append(value)

        # this places cts in row-major order
        cts.append(ct)
    return cts


def apply_punctured_layout(pt_tensor, layout):
    """apply a layout to a pt tensor"""
    cts = apply_layout(pt_tensor, layout)

    # make the punctured matrix holey!
    masks = layout.term.cs[4]
    assert len(cts) == len(masks)

    for i in range(len(cts)):
        cts[i] = [c * m for c, m in zip(cts[i], masks[i])]
    return cts


def parse_layout(layout_str, n, secret):
    import re

    result = {"roll": None, "dims": []}

    # Extract roll numbers if present
    roll_match = re.search(r"roll\((\d+),(\d+)\)", layout_str)
    if roll_match:
        result["roll"] = {
            "to": int(roll_match.group(1)),
            "from": int(roll_match.group(2)),
        }

    # Extract all bracket terms [number:number:number] or [number:number]
    dim_matches = re.findall(r"\[\d+:\d+(?::\d+)?\]", layout_str)
    if dim_matches:
        for dim in dim_matches:
            # Remove the brackets and split by colon
            result["dims"].append(dim)

    # transform rolls
    rolls = []
    for roll in result["roll"]:
        rolls.append(Roll(roll["to"], roll["from"]))
    dims = []
    for dim in result["dims"]:
        dims.append(Dim.parse(dim))

    return Layout(None, rolls, dims, n, secret)
