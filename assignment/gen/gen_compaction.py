"""
Layout compaction utilities for FHE computations.

This module provides functions for compacting tensor layouts to optimize
memory usage and improve performance in FHE computations. Compaction
involves reducing the number of dimensions while maintaining the same
logical structure.

Key functions:
- compact_ct_dim: Compacts ciphertext dimensions in layouts
- find_compaction: Finds optimal compaction strategies for layouts
"""

from copy import deepcopy as copy

from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from ir.roll import Roll
from util.util import get_dim_map_by_dim, next_non_empty_dim, split_dim


def compact_ct_dim(ct_dim, slot_dims):
    """Compacts a ciphertext dimension within slot dimensions.

    This function optimizes the layout by compacting ciphertext dimensions
    with slot dimensions to reduce memory usage and improve performance.
    It applies heuristics to keep related dimensions together during
    compaction.

    Args:
        ct_dim: Ciphertext dimension to compact
        slot_dims: List of slot dimensions to compact with

    Returns:
        tuple: (new_ct_dims, new_slot_dims) representing the compacted layout
    """
    # return new_ct_dim and new_slot_dims
    new_ct_dims = []
    new_slot_dims = []

    # create slot_dim_map for HEURISTIC:
    # keep dimensions together when compacted
    slot_dim_map = {}
    for i, dim in enumerate(slot_dims):
        if dim.dim is not None:
            slot_dim_map[i] = dim

    ct_extent = ct_dim.extent
    ct_stride = ct_dim.stride
    for i, slot_dim in enumerate(slot_dims):
        if slot_dim.dim is not None or ct_extent <= 1:
            new_slot_dims.append(slot_dim)
        else:
            if slot_dim.extent > ct_extent:
                # if gap dimension is greater than the ct_extent
                # then find where to compact ct_dim
                if i - 1 in slot_dim_map:
                    new_slot_dims.append(ct_dim.copy())
                    new_slot_dims.append(Dim.parse(f"G:{slot_dim.extent // ct_extent}"))
                elif i + 1 in slot_dim_map:
                    new_slot_dims.append(Dim.parse(f"G:{slot_dim.extent // ct_extent}"))
                    new_slot_dims.append(ct_dim.copy())
            elif slot_dim.extent == ct_extent:
                new_slot_dims.append(ct_dim.copy())
            else:
                # if the gap dimension is smaller than the ct_extent,
                # then compact a little and continue
                new_slot_dim = Dim(ct_dim.dim, slot_dim.extent, ct_dim.stride)
                new_slot_dims.append(new_slot_dim)
                ct_stride *= slot_dim.extent
            ct_extent //= slot_dim.extent

    if ct_extent > 1:
        new_ct_dims.append(Dim(ct_dim.dim, ct_extent, ct_stride))
    return new_ct_dims, new_slot_dims


def naive_compaction(ct_dims, slot_dims):
    all_new_dims = []
    for i, ct_dim in enumerate(ct_dims):
        new_ct_dims, new_slot_dims = compact_ct_dim(ct_dim, slot_dims)
        new_ct_dims = ct_dims[:i] + new_ct_dims + ct_dims[i + 1 :]
        if check_empty_slot_dim(new_slot_dims):
            recursive = naive_compaction(copy(new_ct_dims), copy(new_slot_dims))
            all_new_dims += recursive
        else:
            all_new_dims.append((new_ct_dims, new_slot_dims))
    return all_new_dims


def compaction_heuristic(ct_dims, slot_dims):
    """
    At a high level, the heuristic tries to keep runs of the dimensions together
    to prevent internal fragmentation, where a dimension is cut multiple times
    """

    # track the positions of ciphertext dimensions
    remaining_ct_dims = copy(ct_dims)
    ct_dim_map = get_dim_map_by_dim(copy(ct_dims))

    # track the positions of slot dimensions
    new_slot_dims = []
    remaining_slot_dims = copy(slot_dims)

    # slot dim to adjust
    next_dim_index = None
    while remaining_slot_dims and remaining_ct_dims:
        slot_dim = remaining_slot_dims[0]

        # if the slot dimnension is EMPTY, then
        # this can be filled with a ciphertext dimension
        if slot_dim.dim_type == DimType.EMPTY:
            # find the next_dim_index
            # next_dim_index is found by finding the next non-empty slot dimension if
            # the next_dim_index is not set.
            # a ct_dim with a matching next_dim_index is then found, otherwise a random ct is
            # drawn at random
            if next_dim_index is None:
                next_dim = next_non_empty_dim(remaining_slot_dims)
                if next_dim:
                    next_dim_index = next_dim.dim

            if next_dim_index not in ct_dim_map:
                next_dim_index = next_non_empty_dim(remaining_ct_dims).dim

            # get matched ct_dim
            cts = ct_dim_map[next_dim_index]
            ct_dim = cts[0]

            # compact slot dimension with ct_dim
            if slot_dim.extent > ct_dim.extent:
                split_slot_1, split_slot_2 = split_dim(slot_dim, ct_dim.extent)
                new_ct_dim = Dim(ct_dim.dim, split_slot_1.extent, ct_dim.stride)
                remaining_slot_dims.remove(slot_dim)
                next_remaining_slot_dim = next_non_empty_dim(remaining_slot_dims[1:])
                if (
                    next_remaining_slot_dim
                    and next_remaining_slot_dim.dim == next_dim_index
                ):
                    remaining_slot_dims.insert(0, split_slot_2)
                    remaining_slot_dims.insert(0, new_ct_dim)
                else:
                    remaining_slot_dims.insert(0, new_ct_dim)
                    remaining_slot_dims.insert(0, split_slot_2)

                remaining_ct_dims.remove(ct_dim)
                cts.remove(ct_dim)
                if not cts:
                    del ct_dim_map[next_dim_index]
                else:
                    ct_dim_map[next_dim_index] = cts
            elif slot_dim.extent == ct_dim.extent:
                new_slot_dims.append(ct_dim.copy())
                remaining_slot_dims.remove(slot_dim)
                remaining_ct_dims.remove(ct_dim)
                cts.remove(ct_dim)
                if not cts:
                    del ct_dim_map[next_dim_index]
                else:
                    ct_dim_map[next_dim_index] = cts

                if remaining_slot_dims:
                    slot_dim = remaining_slot_dims[0]
            else:
                split_ct_1, split_ct_2 = split_dim(ct_dim, slot_dim.extent)
                new_slot_dims.append(
                    Dim(ct_dim.dim, split_ct_1.extent, split_ct_1.stride)
                )
                remaining_slot_dims.remove(slot_dim)

                ct_index = remaining_ct_dims.index(ct_dim)
                remaining_ct_dims.remove(ct_dim)
                remaining_ct_dims.insert(ct_index, split_ct_2)

                cts.remove(ct_dim)
                cts.insert(0, split_ct_2)
                ct_dim_map[next_dim_index] = cts
                if remaining_slot_dims:
                    slot_dim = remaining_slot_dims[0]

        else:
            next_dim_index = slot_dim.dim
            new_slot_dims.append(slot_dim)
            remaining_slot_dims.remove(slot_dim)
            if remaining_slot_dims:
                slot_dim = remaining_slot_dims[0]

    # add any remaining slot dims to new_slot_dims
    # this is because either there aren't any ct_dims left to compact
    # or remaining slot dims should already be empty
    new_slot_dims += remaining_slot_dims
    return [[remaining_ct_dims, new_slot_dims]]


def check_empty_slot_dim(slot_dims):
    return any([dim.dim_type == DimType.EMPTY for dim in slot_dims])


def find_compaction(kernel):
    """Apply compaction to a kernel, if applicable"""

    layout = kernel.layout
    # find gap dimensions in layout
    if layout.ct_dims and not layout.rolls and check_empty_slot_dim(layout.slot_dims):
        # all_new_dims = self.naive_compaction(layout.ct_dims, layout.slot_dims)
        all_new_dims = compaction_heuristic(layout.ct_dims, layout.slot_dims)
        for new_ct_dims, new_slot_dims in all_new_dims:
            compacted_layout = dimension_merging(
                Layout(
                    layout.term,
                    layout.rolls,
                    new_ct_dims + new_slot_dims,
                    layout.n,
                    layout.secret,
                )
            )
            return Kernel(KernelOp.COMPACT, [kernel], compacted_layout)
    elif layout.ct_dims and layout.rolls:
        if len(layout.rolls) > 1:
            raise NotImplementedError("what to do with multiple rolls?")
        dims = layout.get_dims().copy()
        roll = layout.rolls[0]
        if dims[roll.dim_to_roll_by].dim is None:
            raise NotImplementedError("keep dimensions!")
        else:
            if roll.roll_index(dims) == (0, 1):
                new_dims = [dims[1], dims[0]]
                new_roll = Roll(dims[1], dims[0])
            elif roll.roll_index(dims) == (0, 2):
                new_dims = [dims[0], dims[2]]
                new_roll = Roll(dims[0], dims[2])
            else:
                raise NotImplementedError("other types of rolls")

            compacted_layout = dimension_merging(
                Layout(layout.term, [new_roll], new_dims, layout.n, layout.secret)
            )
            return Kernel(KernelOp.COMPACT, [kernel], compacted_layout)
    else:
        return kernel
