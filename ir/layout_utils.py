"""
Layout utility functions for the IR module.

This module contains layout-related utility functions that are used by
the IR module to avoid circular imports with util.layout_util.
"""

from copy import copy as copy

from ir.dim import DimType
from ir.layout import Layout
from ir.roll import Roll


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
