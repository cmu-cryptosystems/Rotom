"""
Binary operation layout generation utilities.

This module provides functions for generating optimal layouts for binary
tensor operations (add, subtract, multiply, matrix multiplication) in FHE
computations. It handles dimension alignment, layout merging, and cost
optimization for various binary operations.

Key functions:
- check_extent_alignment: Validates dimension extent alignment
- match_kernel_dims: Matches dimensions between kernel operands
- gen_binop: Main function for generating binary operation layouts
"""

from copy import deepcopy as copy

from assignment.alignment import get_dim_alignment
from assignment.gen.gen_compaction import find_compaction
from frontends.tensor import TensorOp
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from ir.roll import Roll
from util.layout_util import (
    align_dimension_extents,
    dimension_merging,
    get_dim_map,
    match_dims,
    merge_dims,
)
from util.util import get_sum_dim, prod, split_dim


def check_extent_alignment(alignment, a_dims, b_dims):
    """Checks if dimension extents are properly aligned between operands.

    This function validates that the extents and strides of aligned dimensions
    match between two operands. This is crucial for ensuring that binary
    operations can be performed correctly in the HE domain.

    Args:
        alignment: Set of (dim_a, dim_b) tuples representing dimension alignments
        a_dims: List of Dim objects from the first operand
        b_dims: List of Dim objects from the second operand

    Returns:
        bool: True if all aligned dimensions have matching extents and strides
    """
    a_extent_map = {}
    for dim in a_dims:
        if dim.dim not in a_extent_map:
            a_extent_map[dim.dim] = []
        a_extent_map[dim.dim].append((dim.extent, dim.stride))

    b_extent_map = {}
    for dim in b_dims:
        if dim.dim not in b_extent_map:
            b_extent_map[dim.dim] = []
        b_extent_map[dim.dim].append((dim.extent, dim.stride))

    for k, v in alignment:
        if k in a_extent_map and sorted(a_extent_map[k]) != sorted(b_extent_map[v]):
            return False
    return True


def get_fill_len(dims):
    return int(prod([dim.extent for dim in dims if dim.dim_type == DimType.FILL]))


def match_kernel_dims(a_kernel, b_kernel):
    if len(a_kernel) < len(b_kernel):
        diff = len(b_kernel) // len(a_kernel)
        # find stride:
        stride = 1
        for dim in a_kernel.layout.get_dims():
            if dim.dim_type == DimType.EMPTY:
                stride *= dim.extent
        dim = Dim(None, diff, stride, DimType.EMPTY)
        new_dims = a_kernel.layout.get_dims()
        new_dims.insert(0, dim)
        new_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls,
            new_dims,
            a_kernel.layout.offset,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        new_a_kernel = copy(a_kernel)
        new_a_kernel.layout = new_layout
        a_kernel = new_a_kernel
    elif len(b_kernel) < len(a_kernel):
        diff = len(a_kernel) // len(b_kernel)

        # find stride:
        stride = 1
        for dim in b_kernel.layout.get_dims():
            if dim.dim_type == DimType.EMPTY:
                stride *= dim.extent

        dim = Dim(None, diff, stride, DimType.EMPTY)
        new_dims = b_kernel.layout.get_dims()
        new_dims.insert(0, dim)
        new_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls,
            new_dims,
            b_kernel.layout.offset,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        new_b_kernel = copy(b_kernel)
        new_b_kernel.layout = new_layout
        b_kernel = new_b_kernel
    assert len(a_kernel) == len(b_kernel)
    return a_kernel, b_kernel


def replicate_dimensions(a_kernel, b_kernel, shapes, alignment):
    # match kernel lengths
    a_kernel, b_kernel = match_kernel_dims(a_kernel, b_kernel)

    # figure out how much replication is needed
    a_fill_len = get_fill_len(a_kernel.layout.get_dims())
    b_fill_len = get_fill_len(b_kernel.layout.get_dims())
    fill_len = max(a_fill_len, b_fill_len)

    # replication should be based on the alignment of the term
    a_shape = shapes[0]
    b_shape = shapes[1]
    replicated_shape = []
    for a_dim, b_dim in alignment:
        if a_dim is not None:
            replicated_shape.append(a_shape[a_dim])
        elif b_dim is not None:
            replicated_shape.append(b_shape[b_dim])

    # figure out how much each term needs to be replicated to match alignment
    replicated_kernels = []
    for kernel in [a_kernel, b_kernel]:
        shape_len = max(fill_len, prod(replicated_shape))
        kernel_len = int(
            prod(
                [
                    dim.extent
                    for dim in kernel.layout.get_dims()
                    if dim.dim_type is DimType.FILL
                ]
            )
        )

        shape_len //= kernel_len
        if shape_len <= 1:
            # no replication needed!
            replicated_kernels.append(kernel)
            continue

        new_dims = []
        replicated = 1
        for dim in kernel.layout.get_dims()[::-1]:
            if shape_len > 1 and dim.dim_type == DimType.EMPTY:
                if dim.extent <= shape_len:
                    shape_len //= dim.extent
                    new_dim = Dim(dim.dim, dim.extent, replicated, DimType.FILL)
                    new_dims.insert(0, new_dim)
                    replicated *= dim.extent
                else:
                    split_dim_1, split_dim_2 = split_dim(dim, dim.extent // shape_len)
                    shape_len //= dim.extent
                    split_dim_1.dim_type = DimType.EMPTY
                    split_dim_2.dim_type = DimType.FILL
                    split_dim_2.stride = replicated
                    replicated *= dim.extent
                    new_dims.insert(0, split_dim_2)
                    new_dims.insert(0, split_dim_1)
            else:
                new_dim = copy(dim)
                new_dims.insert(0, new_dim)

        # replicate ct dimensions
        if shape_len > 1:
            new_dims.insert(0, Dim(None, shape_len, replicated))

        layout = dimension_merging(
            Layout(
                kernel.layout.term,
                kernel.layout.rolls,
                new_dims,
                kernel.layout.offset,
                kernel.layout.n,
                kernel.layout.secret,
            )
        )
        replicated_kernels.append(Kernel(KernelOp.REPLICATE, [kernel], layout))

    a_fill_len = get_fill_len(replicated_kernels[0].layout.get_dims())
    b_fill_len = get_fill_len(replicated_kernels[1].layout.get_dims())
    assert a_fill_len == b_fill_len
    return replicated_kernels


def check_dim_len_eq(a_dims, b_dims):
    a_dim_len = int(
        prod([dim.extent for dim in a_dims if dim.dim_type is DimType.FILL])
    )
    b_dim_len = int(
        prod([dim.extent for dim in b_dims if dim.dim_type is DimType.FILL])
    )
    return a_dim_len == b_dim_len


def check_ct_dim_alignment(alignment, a_kernel, b_kernel):
    a_ct_dims = a_kernel.layout.ct_dims
    b_ct_dims = b_kernel.layout.ct_dims

    a_ct_dims_map = {}
    for ct_dim in a_ct_dims:
        if ct_dim.dim not in a_ct_dims_map:
            a_ct_dims_map[ct_dim.dim] = 1
        a_ct_dims_map[ct_dim.dim] *= ct_dim.extent
    b_ct_dims_map = {}
    for ct_dim in b_ct_dims:
        if ct_dim.dim not in b_ct_dims_map:
            b_ct_dims_map[ct_dim.dim] = 1
        b_ct_dims_map[ct_dim.dim] *= ct_dim.extent

    for a, b in alignment:
        if a is None or b is None:
            continue
        if a in a_ct_dims_map and b not in b_ct_dims_map:
            return False
        elif a not in a_ct_dims_map and b in b_ct_dims_map:
            return False
        elif (
            a in a_ct_dims_map
            and b in b_ct_dims_map
            and a_ct_dims_map[a] != b_ct_dims_map[b]
        ):
            return False

    a_stride = 1
    for ct_dim in a_ct_dims:
        if ct_dim.dim is not None:
            a_stride *= ct_dim.stride
    b_stride = 1
    for ct_dim in b_ct_dims:
        if ct_dim.dim is not None:
            b_stride *= ct_dim.stride
    return True


def check_slot_dim_alignment(alignment, a_kernel, b_kernel):
    a_dims = a_kernel.layout.slot_dims
    b_dims = b_kernel.layout.slot_dims
    # assert check_dim_len_eq(a_dims, b_dims)
    # print(a_kernel)
    # print(b_kernel)
    # print(a_dims)
    # print(b_dims)

    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0
    while a_next < len(a_dims) and b_next < len(b_dims):
        a_dim = a_dims[a_next]
        b_dim = b_dims[b_next]
        # both emtpy dimensions, skip
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        # dimension type mismatch
        if a_dim.dim_type != b_dim.dim_type:
            return False

        # alignment mismatch
        if (a_dim.dim, b_dim.dim) not in alignment:
            return False

        a_extent = prod([dim.extent for dim in a_dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in b_dims[b_start : b_next + 1]])
        if a_extent == b_extent:
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
        elif a_extent > b_extent:
            b_next += 1
        else:
            a_next += 1
    return True


def check_ct_roll_alignment(alignment, a_kernel, b_kernel):
    alignment_map = {}
    rev_alignment_map = {}
    for k, v in alignment:
        alignment_map[k] = v
        rev_alignment_map[v] = k

    a_rolls = a_kernel.layout.rolls
    b_rolls = b_kernel.layout.rolls

    # map rolls to their dimension and extent
    a_roll_set = set()
    for a_roll in a_rolls:
        a_roll_set.add((a_roll.dim_to_roll.dim, a_roll.dim_to_roll.extent))
    b_roll_set = set()
    for b_roll in b_rolls:
        b_roll_set.add((b_roll.dim_to_roll.dim, b_roll.dim_to_roll.extent))

    for a in a_roll_set:
        if a[0] not in alignment_map:
            return False
        if (alignment_map[a[0]], a[1]) not in b_roll_set:
            return False
    for b in b_roll_set:
        if b[0] not in rev_alignment_map:
            return False
        if (rev_alignment_map[b[0]], b[1]) not in a_roll_set:
            return False
    return True


def check_dim_alignment(alignment, a_kernel, b_kernel):
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    assert check_dim_len_eq(a_dims, b_dims)

    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0
    while a_next < len(a_dims) and b_next < len(b_dims):
        a_dim = a_dims[a_next]
        b_dim = b_dims[b_next]
        # both emtpy dimensions, skip
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        # dimension type mismatch
        if a_dim.dim_type != b_dim.dim_type:
            return False

        # alignment mismatch
        if (a_dim.dim, b_dim.dim) not in alignment:
            return False

        a_extent = prod([dim.extent for dim in a_dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in b_dims[b_start : b_next + 1]])
        if a_extent == b_extent:
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
        elif a_extent > b_extent:
            b_next += 1
        else:
            a_next += 1
    return True


def check_roll_alignment(term, alignment, a_kernel, b_kernel):
    # check rolls based on term.op
    # check to see that the summing dimensions have the same rolls
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    a_sum_dim = get_sum_dim(term, a_kernel)
    b_sum_dim = get_sum_dim(term, b_kernel)

    alignment_map = {}
    alignment_map_rev = {}
    for k, v in alignment:
        alignment_map[k] = v
        alignment_map_rev[v] = k

    a_roll_idx_set = set()
    b_roll_idx_set = set()
    for a_roll in a_kernel.layout.rolls:
        a_roll_idx_set.add(a_roll.roll_index(a_dims))

    for b_roll in b_kernel.layout.rolls:
        b_roll_idx_set.add(b_roll.roll_index(b_dims))

    # check that aligned dimensions have aligned rolls
    # this only applies to dimensions that will be summed together
    for a_roll in a_kernel.layout.rolls:
        if (
            a_roll.dim_to_roll.dim == a_sum_dim
            and a_roll.roll_index(a_dims) not in b_roll_idx_set
        ):
            return False
    for b_roll in b_kernel.layout.rolls:
        if (
            b_roll.dim_to_roll.dim == b_sum_dim
            and b_roll.roll_index(b_dims) not in a_roll_idx_set
        ):
            return False

    if term.op == TensorOp.MATMUL:
        if len(a_kernel.layout.rolls) == 1:
            dim_to_roll = a_kernel.layout.rolls[0].dim_to_roll
            dim_to_roll_by = a_kernel.layout.rolls[0].dim_to_roll_by
            if (
                dim_to_roll in a_kernel.layout.slot_dims
                and dim_to_roll_by in a_kernel.layout.slot_dims
                and dim_to_roll.dim is not None
                and dim_to_roll_by.dim is not None
                and dim_to_roll.dim == 0
            ):
                return False

    return True


def check_alignment(term, alignment, a_kernel, b_kernel):
    return check_dim_alignment(alignment, a_kernel, b_kernel) and check_roll_alignment(
        term, alignment, a_kernel, b_kernel
    )


def match_layout(term, kernels, alignment, roll_flag):
    # find dimension alignment
    if check_alignment(term, alignment, kernels[0], kernels[1]):
        # if alignment between the dimensions are correct, return kernels
        return [kernels]
    else:
        # if alignment is off, then convert kernels before returning
        # swapping dimensions can be done using either converions
        # or by applying a roll with an empty dimension
        matched_layouts = []

        if not kernels[0].layout.rolls and not kernels[1].layout.rolls:
            # align dimensions using conversions
            matched_layouts += conv_dimensions(alignment, kernels)

        # align dimensions using rolls
        if roll_flag:
            matched_layouts += roll_dimensions(term, alignment, kernels)

        # assert that dimensions are aligned
        matched = []
        for a_kernel, b_kernel in matched_layouts:

            if not check_alignment(term, alignment, a_kernel, b_kernel):
                print("failed match:")
                print(a_kernel)
                print(b_kernel)
                print()
            else:
                matched.append((a_kernel, b_kernel))
            # assert check_alignment(term, alignment, a_kernel, b_kernel)
        return matched


def get_dim_from_alignment(alignment, dim, index, remaining_dims):
    if dim.dim_type == DimType.EMPTY:
        return dim

    for aligned_dims in alignment:
        if aligned_dims[index] == dim.dim:
            # check if this dimension is in remaining dims
            for remaining_dim in remaining_dims:
                if aligned_dims[index ^ 1] == remaining_dim.dim:
                    remaining_dims.remove(remaining_dim)
                    return remaining_dim

    # dim was not in alignment
    return remaining_dims[0]


def match_public_kernel(alignment, a_kernel, b_kernel, left):
    # match a_kernel with b_kernel
    aligned_b_dims, aligned_a_dims = get_aligned_dimensions(
        alignment, a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
    )

    if left:
        roll_indices = []
        for roll in b_kernel.layout.rolls:
            roll_indices.append(roll.roll_index(b_kernel.layout.get_dims()))

        new_rolls = []
        for roll_index in roll_indices:
            new_rolls.append(
                Roll(aligned_a_dims[roll_index[0]], aligned_a_dims[roll_index[1]])
            )

        aligned_a_layout = Layout(
            a_kernel.layout.term,
            new_rolls,
            aligned_a_dims,
            a_kernel.layout.offset,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        aligned_a_kernel = Kernel(
            KernelOp.TENSOR,
            [],
            aligned_a_layout,
        )
        return (aligned_a_kernel, b_kernel)
    else:
        # match a_kernel with b_kernel
        aligned_b_dims, aligned_a_dims = get_aligned_dimensions(
            alignment, a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
        )

        roll_indices = []
        for roll in a_kernel.layout.rolls:
            roll_indices.append(roll.roll_index(a_kernel.layout.get_dims()))

        new_rolls = []
        for roll_index in roll_indices:
            new_rolls.append(
                Roll(aligned_b_dims[roll_index[0]], aligned_b_dims[roll_index[1]])
            )
        aligned_b_layout = Layout(
            b_kernel.layout.term,
            new_rolls,
            aligned_b_dims,
            b_kernel.layout.offset,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        aligned_b_kernel = Kernel(
            KernelOp.TENSOR,
            [],
            aligned_b_layout,
        )
        return (a_kernel, aligned_b_kernel)


def find_misaligned_dims(alignment, a_kernel, b_kernel):
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    assert check_dim_len_eq(a_dims, b_dims)

    misaligned_a_dims = []
    misaligned_b_dims = []

    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0
    while a_next < len(a_dims) and b_next < len(b_dims):
        a_dim = a_dims[a_next]
        b_dim = b_dims[b_next]
        # both emtpy dimensions, skip
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        if a_dim.dim is not None and (a_dim.dim, b_dim.dim) not in alignment:
            misaligned_a_dims.append(a_dim)
        if b_dim.dim is not None and (a_dim.dim, b_dim.dim) not in alignment:
            misaligned_b_dims.append(b_dim)

        a_extent = prod([dim.extent for dim in a_dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in b_dims[b_start : b_next + 1]])
        if a_extent == b_extent:
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
        elif a_extent > b_extent:
            b_next += 1
        else:
            a_next += 1
    return misaligned_a_dims, misaligned_b_dims


def roll_dim_alignment(term, alignment, a_kernel, b_kernel):
    # check to see if sum dimensions are aligned, otherwise prune
    aligned_kernels = []

    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    extent_a_dims, extent_b_dims = align_dimension_extents(
        a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
    )

    a_sum_dim = get_sum_dim(term, a_kernel)
    b_sum_dim = get_sum_dim(term, b_kernel)

    # initial checks to see if summation dimensions are aligned
    for a_dim, b_dim in zip(extent_a_dims, extent_b_dims):
        if a_dim.dim == a_sum_dim and b_dim.dim != b_sum_dim:
            return []
        if b_dim.dim == b_sum_dim and a_dim.dim != a_sum_dim:
            return []
        if (
            a_dim.dim == a_sum_dim
            and b_dim.dim == b_sum_dim
            and a_dim.stride != b_dim.stride
        ):
            return []
        if (
            b_dim.dim == b_sum_dim
            and a_dim.dim == a_sum_dim
            and a_dim.stride != b_dim.stride
        ):
            return []

    # check to see if sum dimensions have the same rollutations
    a_sum_rolls = set()
    for roll in a_kernel.layout.rolls:
        if roll.dim_to_roll.dim == a_sum_dim:
            a_sum_rolls.add(roll.roll_index(a_dims))
    b_sum_rolls = set()
    for roll in b_kernel.layout.rolls:
        if roll.dim_to_roll.dim == b_sum_dim:
            b_sum_rolls.add(roll.roll_index(b_dims))

    if a_sum_rolls != b_sum_rolls:
        return []

    # check if there are enough an empty dimensions to swap to in aligned dimensions
    aligned_b_dims, aligned_a_dims = get_aligned_dimensions(alignment, a_dims, b_dims)
    for a_dim in a_dims:
        if a_dim not in aligned_a_dims:
            return []
    for b_dim in b_dims:
        if b_dim not in aligned_b_dims:
            return []

    # find which dimensions swapped with b
    b_dim_map = get_dim_map(b_dims)
    aligned_b_dim_map = get_dim_map(aligned_b_dims)

    # find rolls
    b_rolls = []
    for k, v in aligned_b_dim_map.items():
        if k.dim is not None and v != b_dim_map[k]:
            if k.extent != b_dims[v].extent:
                return []
            b_rolls.append(Roll(k, b_dims[v]))

    # apply found rolls
    b_dims = copy(b_dims)
    rolled_b_kernel = b_kernel
    for roll in b_rolls:
        if roll in b_kernel.layout.rolls:
            continue  # roll already exists

        # get dimensions to swap
        roll_idx = roll.roll_index(b_dims)

        # create new rolls on
        b_dims[roll_idx[0]], b_dims[roll_idx[1]] = (
            b_dims[roll_idx[1]],
            b_dims[roll_idx[0]],
        )

        # remap original rolls
        b_rolls = []
        for existing_roll in b_kernel.layout.rolls:
            update = existing_roll.roll_update(b_kernel.layout.get_dims())
            dim_to_roll = update[0]
            dim_to_roll_by = b_dims[update[1]]
            og_dim_to_roll_by = existing_roll.dim_to_roll_by
            if dim_to_roll != dim_to_roll_by:
                b_rolls.append(Roll(dim_to_roll, dim_to_roll_by))
            else:
                b_rolls.append(Roll(dim_to_roll, og_dim_to_roll_by))

        # append new roll
        b_rolls.append(roll)
        rolled_b_layout = Layout(
            b_kernel.layout.term,
            b_rolls,
            b_dims,
            a_kernel.layout.offset,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        rolled_b_kernel = Kernel(
            KernelOp.ROLL,
            [roll, rolled_b_kernel],
            rolled_b_layout,
        )
    if check_alignment(term, alignment, a_kernel, rolled_b_kernel):
        aligned_kernels.append((a_kernel, rolled_b_kernel))

    # find which dimensions swapped with a
    a_dim_map = get_dim_map(a_dims)
    aligned_a_dim_map = get_dim_map(aligned_a_dims)

    a_rolls = []
    for k, v in aligned_a_dim_map.items():
        if k.dim is not None and v != a_dim_map[k]:
            a_rolls.append(Roll(k, a_dims[v]))

    a_dims = copy(a_dims)
    rolled_a_kernel = a_kernel
    for roll in a_rolls:
        # get dimensions to swap
        roll_idx = roll.roll_index(a_dims)

        # create new rolls on
        a_dims[roll_idx[0]], a_dims[roll_idx[1]] = (
            a_dims[roll_idx[1]],
            a_dims[roll_idx[0]],
        )

        # remap original rolls
        a_rolls = []
        for existing_roll in a_kernel.layout.rolls:
            update = existing_roll.roll_update(a_kernel.layout.get_dims())
            dim_to_roll = update[0]
            dim_to_roll_by = a_dims[update[1]]
            og_dim_to_roll_by = existing_roll.dim_to_roll_by
            if dim_to_roll != dim_to_roll_by:
                a_rolls.append(Roll(dim_to_roll, dim_to_roll_by))
            else:
                a_rolls.append(Roll(dim_to_roll, og_dim_to_roll_by))

        # roll already exists!
        if roll in a_rolls:
            updated_a_layout = Layout(
                a_kernel.layout.term,
                a_rolls,
                a_dims,
                a_kernel.layout.offset,
                a_kernel.layout.n,
                a_kernel.layout.secret,
            )
            updated_a_kernel = copy(rolled_a_kernel)
            updated_a_kernel.layout = updated_a_layout
            rolled_a_kernel = updated_a_kernel
        else:
            # append new roll
            a_rolls.append(roll)
            rolled_a_layout = Layout(
                a_kernel.layout.term,
                a_rolls,
                a_dims,
                a_kernel.layout.offset,
                a_kernel.layout.n,
                a_kernel.layout.secret,
            )
            rolled_a_kernel = Kernel(
                KernelOp.ROLL,
                [roll, rolled_a_kernel],
                rolled_a_layout,
            )
    if check_alignment(term, alignment, rolled_a_kernel, b_kernel):
        aligned_kernels.append((rolled_a_kernel, b_kernel))
    return aligned_kernels


def roll_roll_alignment(term, a_kernel, b_kernel):
    # rollutations are not aligned
    # need to match rollutations
    # transform sum rolls into indices
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    a_sum_dim = get_sum_dim(term, a_kernel)
    b_sum_dim = get_sum_dim(term, b_kernel)

    if len(a_dims) > len(b_dims):
        b_dims = match_dims(b_dims, a_dims)
    elif len(b_dims) > len(a_dims):
        a_dims = match_dims(a_dims, b_dims)

    a_sum_rolls = []
    b_sum_rolls = []
    for roll in a_kernel.layout.rolls:
        if roll.dim_to_roll.dim == a_sum_dim:
            dim_to_roll_idx = a_dims.index(roll.dim_to_roll)
            dim_to_roll_by_idx = a_dims.index(roll.dim_to_roll_by)
            a_sum_rolls.append((dim_to_roll_idx, dim_to_roll_by_idx))
    for roll in b_kernel.layout.rolls:
        if roll.dim_to_roll.dim == b_sum_dim:
            dim_to_roll_idx = b_dims.index(roll.dim_to_roll)
            dim_to_roll_by_idx = b_dims.index(roll.dim_to_roll_by)
            b_sum_rolls.append((dim_to_roll_idx, dim_to_roll_by_idx))

    # rolls to add
    a_roll_indices = []
    b_roll_indices = []
    for a_roll in a_sum_rolls:
        if a_roll not in b_sum_rolls:
            b_roll_indices.append(a_roll)
    for b_roll in b_sum_rolls:
        if b_roll not in a_sum_rolls:
            a_roll_indices.append(b_roll)

    # reconstruct rolls
    a_rolls = []
    for roll in a_roll_indices:
        a_rolls.append(Roll(a_dims[roll[0]], a_dims[roll[1]]))
    b_rolls = []
    for roll in b_roll_indices:
        b_rolls.append(Roll(b_dims[roll[0]], b_dims[roll[1]]))

    if b_rolls:
        rolled_b_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls + b_rolls,
            b_dims,
            b_kernel.layout.offset,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        rolled_b_kernel = Kernel(
            KernelOp.ROLL,
            [b_rolls[0], b_kernel],
            rolled_b_layout,
        )
    else:
        rolled_b_kernel = b_kernel

    if a_rolls:
        rolled_a_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls + a_rolls,
            a_dims,
            a_kernel.layout.offset,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        rolled_a_kernel = Kernel(
            KernelOp.ROLL,
            [a_rolls[0], a_kernel],
            rolled_a_layout,
        )

    else:
        rolled_a_kernel = a_kernel
    return (rolled_a_kernel, rolled_b_kernel)


def apply_roll_conversion(term, alignment, kernels):
    aligned_kernels = []
    dim_alignment = check_dim_alignment(alignment, kernels[0], kernels[1])
    roll_alignment = check_roll_alignment(term, alignment, kernels[0], kernels[1])
    if dim_alignment and roll_alignment:
        aligned_kernels.append(kernels)
    elif dim_alignment and not roll_alignment:
        aligned_kernels.append(roll_roll_alignment(term, kernels[0], kernels[1]))
    elif (
        roll_alignment
        and not dim_alignment
        and not kernels[0].layout.rolls
        and not kernels[1].layout.rolls
    ):
        aligned_kernels += roll_dim_alignment(term, alignment, kernels[0], kernels[1])
    elif roll_alignment and not dim_alignment:
        aligned_kernels += roll_dim_alignment(term, alignment, kernels[0], kernels[1])
    return aligned_kernels


def roll_dimensions(term, alignment, kernels):
    a_kernel = kernels[0]
    b_kernel = kernels[1]

    aligned_kernels = []
    if not a_kernel.layout.secret:
        aligned_kernels.append(match_public_kernel(alignment, a_kernel, b_kernel, True))
    elif not b_kernel.layout.secret:
        aligned_kernels.append(
            match_public_kernel(alignment, a_kernel, b_kernel, False)
        )
    else:
        # check if ct dimensions are matched
        ct_dim_alignment = check_ct_dim_alignment(alignment, a_kernel, b_kernel)
        slot_dim_alignment = check_slot_dim_alignment(alignment, a_kernel, b_kernel)
        ct_roll_alignment = check_ct_roll_alignment(alignment, a_kernel, b_kernel)

        if ct_dim_alignment and slot_dim_alignment and ct_roll_alignment:
            a_rolls = {}
            for roll in kernels[0].layout.rolls:
                roll_index = roll.roll_index(kernels[0].layout.get_dims())
                if roll_index[0] not in a_rolls:
                    a_rolls[roll_index[0]] = []
                a_rolls[roll_index[0]].append(roll_index[1])

            b_rolls = {}
            for roll in kernels[1].layout.rolls:
                roll_index = roll.roll_index(kernels[1].layout.get_dims())
                if roll_index[0] not in b_rolls:
                    b_rolls[roll_index[0]] = []
                b_rolls[roll_index[0]].append(roll_index[1])

            # mismatched rolls
            for k, v in a_rolls.items():
                if k in b_rolls and v != b_rolls[k]:
                    return []
            for k, v in b_rolls.items():
                if k in a_rolls and v != a_rolls[k]:
                    return []

            try:
                ct_kernels = conv_ct_dimensions(alignment, kernels)
                for kernels in ct_kernels:
                    skip = False
                    for roll in kernels[0].layout.rolls:
                        if (
                            roll.dim_to_roll not in kernels[0].layout.get_dims()
                            or roll.dim_to_roll_by not in kernels[0].layout.get_dims()
                        ):
                            skip = True
                    for roll in kernels[1].layout.rolls:
                        if (
                            roll.dim_to_roll not in kernels[1].layout.get_dims()
                            or roll.dim_to_roll_by not in kernels[1].layout.get_dims()
                        ):
                            skip = True
                    if skip:
                        continue
                    aligned_kernels += apply_roll_conversion(term, alignment, kernels)
            except:
                aligned_kernels += apply_roll_conversion(term, alignment, kernels)
        else:
            try:
                aligned_kernels += apply_roll_conversion(term, alignment, kernels)
            except:
                aligned_kernels += []
    return aligned_kernels


def get_aligned_dimensions(alignment, a_dims, b_dims):
    # map alignment
    a_alignment = {}
    b_alignment = {}
    for a, b in alignment:
        a_alignment[a] = b
        b_alignment[b] = a

    # create aligned a_dims:
    aligned_b_dims = []
    for a_dim in a_dims:
        dim = copy(a_dim)
        if dim.dim in a_alignment:
            dim.dim = a_alignment[dim.dim]
        elif dim.dim is not None:
            raise NotImplementedError
        aligned_b_dims.append(dim)

    # create aligned b_dims:
    aligned_a_dims = []
    for b_dim in b_dims:
        dim = copy(b_dim)
        if dim.dim in b_alignment:
            dim.dim = b_alignment[dim.dim]
        elif dim.dim is not None:
            raise NotImplementedError
        aligned_a_dims.append(dim)
    return aligned_b_dims, aligned_a_dims


def conv_dimensions(alignment, kernels):
    a_kernel = kernels[0]
    b_kernel = kernels[1]

    aligned_kernels = []
    if not a_kernel.layout.secret:
        aligned_kernels.append(match_public_kernel(alignment, a_kernel, b_kernel, True))
    elif not b_kernel.layout.secret:
        aligned_kernels.append(
            match_public_kernel(alignment, a_kernel, b_kernel, False)
        )
    else:
        # align dimension extents
        extent_a_dims, extent_b_dims = align_dimension_extents(
            a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
        )
        aligned_b_dims, aligned_a_dims = get_aligned_dimensions(
            alignment, extent_a_dims, extent_b_dims
        )

        a_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls,
            extent_a_dims,
            a_kernel.layout.offset,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        a_extent_kernel = Kernel(
            a_kernel.op,
            a_kernel.cs,
            a_layout,
        )

        conv_b_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls,
            aligned_b_dims,
            b_kernel.layout.offset,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )

        conv_b_kernel = Kernel(
            KernelOp.CONVERSION,
            [tuple(extent_b_dims), tuple(aligned_b_dims), b_kernel],
            conv_b_layout,
        )
        aligned_kernels.append((a_extent_kernel, conv_b_kernel))

        b_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls,
            extent_b_dims,
            b_kernel.layout.offset,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        b_extent_kernel = Kernel(
            b_kernel.op,
            b_kernel.cs,
            b_layout,
        )
        conv_a_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls,
            aligned_a_dims,
            a_kernel.layout.offset,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        conv_a_kernel = Kernel(
            KernelOp.CONVERSION,
            [tuple(extent_a_dims), tuple(aligned_a_dims), a_kernel],
            conv_a_layout,
        )
        aligned_kernels.append((conv_a_kernel, b_extent_kernel))

    # return aligned kernels
    return aligned_kernels


def conv_ct_dimensions(alignment, kernels):
    aligned_kernels = []
    a_kernel = kernels[0]
    b_kernel = kernels[1]

    aligned_b_dims, aligned_a_dims = get_aligned_dimensions(
        alignment, a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
    )
    conv_b_layout = Layout(
        b_kernel.layout.term,
        b_kernel.layout.rolls,
        aligned_b_dims,
        b_kernel.layout.offset,
        b_kernel.layout.n,
        b_kernel.layout.secret,
    )
    conv_b_kernel = Kernel(
        KernelOp.CONVERSION,
        [tuple(b_kernel.layout.get_dims()), tuple(aligned_b_dims), b_kernel],
        conv_b_layout,
    )

    aligned_kernels.append((a_kernel, conv_b_kernel))

    conv_a_layout = Layout(
        a_kernel.layout.term,
        a_kernel.layout.rolls,
        aligned_a_dims,
        a_kernel.layout.offset,
        a_kernel.layout.n,
        a_kernel.layout.secret,
    )
    conv_a_kernel = Kernel(
        KernelOp.CONVERSION,
        [tuple(a_kernel.layout.get_dims()), tuple(aligned_a_dims), a_kernel],
        conv_a_layout,
    )

    aligned_kernels.append((conv_a_kernel, b_kernel))

    # return aligned kernels
    return aligned_kernels


def apply_sum_roll(term, kernel):
    """apply roll onto a kernel to move summing dimension to ct dimension
    - Doesn't work if you have to split a dimension that already rolled
    """

    # 1. find summing dimension
    sum_dim = get_sum_dim(term, kernel)

    layout = kernel.layout
    # 2. a roll can be applied if there exists an empty ct_dimension that
    # we can move a sum dimension to.

    if any(dim.dim is None for dim in layout.ct_dims) and any(
        dim.dim == sum_dim for dim in layout.slot_dims
    ):
        for roll in layout.rolls:
            if roll.dim_to_roll.dim == sum_dim or roll.dim_to_roll_by.dim == sum_dim:
                return

        # 1. find empty ct extent
        empty_ct_dim_index = 0
        count = 0
        for i, ct_dim in enumerate(merge_dims(layout.ct_dims, set())):
            if ct_dim.dim is None:
                empty_ct_dim_index = i
                count += 1

        # Assume that there is only a single empty ct dimension. Otherwise, the
        # dimensions can be swapped such that there exists only one.
        assert count == 1

        # 2. find largest slot dimension (that are split and compatible for rolls)
        # the slot dimension should be a sum dimension.
        slot_dim_index = 0
        slot_dim_len = 0
        for i, slot_dim in enumerate(layout.slot_dims):
            if slot_dim.dim == sum_dim and slot_dim.extent > slot_dim_len:
                slot_dim_index = i
                slot_dim_len = slot_dim.extent

        # 3. check if we need to split slot_dim or ct_dim
        # additionally, update alignment
        ct_dim_to_roll = layout.ct_dims[empty_ct_dim_index]
        slot_dim_to_roll = layout.slot_dims[slot_dim_index]
        if slot_dim_to_roll.extent == ct_dim_to_roll.extent:
            new_ct_dims = layout.ct_dims
            new_slot_dims = layout.slot_dims
        elif slot_dim_to_roll.extent <= ct_dim_to_roll.extent:
            split_extent_1 = slot_dim_to_roll.extent
            split_extent_2 = ct_dim_to_roll.extent // slot_dim_to_roll.extent
            split_dim = Dim(ct_dim_to_roll.dim, split_extent_1, ct_dim_to_roll.stride)
            split_dim_rep = Dim(
                ct_dim_to_roll.dim,
                split_extent_2,
                split_extent_1 * ct_dim_to_roll.stride,
            )

            # create new dimensions
            new_ct_dims = (
                layout.ct_dims[:empty_ct_dim_index]
                + [split_dim, split_dim_rep]
                + layout.ct_dims[empty_ct_dim_index + 1 :]
            )
            new_slot_dims = layout.slot_dims
        else:
            # RESTRICT: figure out order of split_dim to prevent internal fragmentation
            ct_extent = ct_dim_to_roll.extent
            if i + 1 < len(layout.slot_dims) and layout.slot_dims[i + 1].dim is None:
                split_extent_1 = slot_dim_to_roll.extent // ct_extent
                split_extent_2 = ct_extent
            else:
                split_extent_1 = ct_extent
                split_extent_2 = slot_dim_to_roll.extent // ct_extent

            split_dim = Dim(
                slot_dim_to_roll.dim,
                split_extent_1,
                split_extent_2 * slot_dim_to_roll.stride,
            )
            split_dim_rep = Dim(
                slot_dim_to_roll.dim, split_extent_2, slot_dim_to_roll.stride
            )

            new_ct_dims = layout.ct_dims
            new_slot_dims = (
                layout.slot_dims[:slot_dim_index]
                + [split_dim, split_dim_rep]
                + layout.slot_dims[slot_dim_index + 1 :]
            )

        new_dims = new_ct_dims + new_slot_dims
        (
            new_dims[empty_ct_dim_index],
            new_dims[len(new_ct_dims) + slot_dim_index],
        ) = (
            new_slot_dims[slot_dim_index],
            new_ct_dims[empty_ct_dim_index],
        )

        new_roll = Roll(
            new_dims[empty_ct_dim_index],
            new_dims[len(new_ct_dims) + slot_dim_index],
        )

        # update existing rolls
        roll_indices = [roll.roll_index(layout.get_dims()) for roll in layout.rolls]
        updated_rolls = []
        for roll, index in zip(layout.rolls, roll_indices):
            updated_rolls.append(Roll(roll.dim_to_roll, new_dims[index[1]]))

        # roll already exists!
        if new_roll in updated_rolls:
            updated_layout = Layout(
                kernel.layout.term,
                updated_rolls,
                new_dims,
                kernel.layout.offset,
                kernel.layout.n,
                kernel.layout.secret,
            )
            updated_kernel = copy(kernel)
            updated_kernel.layout = updated_layout
            return updated_kernel
        else:
            new_layout = Layout(
                kernel.layout.term,
                updated_rolls + [new_roll],
                new_dims,
                kernel.layout.offset,
                kernel.layout.n,
                kernel.layout.secret,
            )
            return Kernel(KernelOp.ROLL, [new_roll, kernel], new_layout)


def output_layout(term, alignment, a_kernel, b_kernel):
    """Determine the output layout from aligned input kernels"""
    match term.op:
        case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
            output_dims = []
            for a_dim in a_kernel.layout.dims:
                b_dim_index = get_dim_from_alignment(
                    alignment, a_dim, 0, copy(b_kernel.layout.get_dims())
                ).dim
                if a_dim.dim is not None and b_dim_index is None:
                    output_dims.append(a_dim)
                elif a_dim.dim is None and b_dim_index is not None:
                    output_dims.append(
                        Dim(b_dim_index, a_dim.extent, a_dim.stride, a_dim.dim_type)
                    )
                elif a_dim.dim is None and b_dim_index is None:
                    output_dims.append(a_dim)
                else:
                    output_dims.append(
                        Dim(
                            max(a_dim.dim, b_dim_index),
                            a_dim.extent,
                            a_dim.stride,
                            a_dim.dim_type,
                        )
                    )

            update_offset = copy(a_kernel.layout.offset)
            for k, v in b_kernel.layout.offset.items():
                if not isinstance(v, list):
                    if isinstance(update_offset[k], list):
                        update_offset[k].append(v)
                    else:
                        update_offset[k] = [update_offset[k], v]
                else:
                    if k in update_offset:
                        update_offset[k] += v
                    else:
                        update_offset[k] = [v]

            output_layout = dimension_merging(
                Layout(
                    term,
                    a_kernel.layout.rolls,
                    output_dims,
                    update_offset,
                    a_kernel.layout.n,
                    a_kernel.layout.secret or b_kernel.layout.secret,
                )
            )
            match term.op:
                case TensorOp.ADD:
                    layout_op = KernelOp.ADD
                case TensorOp.SUB:
                    layout_op = KernelOp.SUB
                case TensorOp.MUL:
                    layout_op = KernelOp.MUL
            output_kernel = Kernel(layout_op, [a_kernel, b_kernel], output_layout)
            return output_kernel
        case TensorOp.MATMUL:
            a_layout = a_kernel.layout
            b_layout = b_kernel.layout
            output_dims = []

            if len(a_layout.get_dims()) != len(b_layout.get_dims()):
                for a_dim in a_layout.get_dims():
                    if a_dim.dim is None:
                        output_dims.append(
                            Dim(1, a_dim.extent, a_dim.stride, DimType.FILL)
                        )
                    elif a_dim.dim == 1:
                        output_dims.append(
                            Dim(None, a_dim.extent, a_dim.stride, DimType.EMPTY)
                        )
                    elif a_dim.dim == 0:
                        output_dims.append(a_dim)
                    else:
                        raise NotImplementedError
            else:
                for a_dim, b_dim in zip(a_layout.get_dims(), b_layout.get_dims()):
                    if a_dim.dim is not None and b_dim.dim is None:
                        output_dims.append(a_dim)
                    elif a_dim.dim is None and b_dim.dim is not None:
                        # HACK:
                        if len(alignment) == 4:
                            b_dim = copy(b_dim)
                            b_dim.dim = 2
                        output_dims.append(b_dim)

                    else:
                        # set summation dimension to None
                        output_dims.append(
                            Dim(None, a_dim.extent, a_dim.stride, DimType.EMPTY)
                        )

            # remove any rolls on the summation dimension
            a_rolls = [roll for roll in a_layout.rolls if roll.dim_to_roll.dim != 1]
            b_rolls = [
                roll for roll in b_kernel.layout.rolls if roll.dim_to_roll.dim != 0
            ]

            # update rolls with new output dimensions
            roll_indices = []
            for roll in a_rolls:
                roll_indices.append(roll.roll_index(a_kernel.layout.get_dims()))
            for roll in b_rolls:
                roll_indices.append(roll.roll_index(b_kernel.layout.get_dims()))

            new_rolls = []
            for roll_index in roll_indices:
                new_rolls.append(
                    Roll(output_dims[roll_index[0]], output_dims[roll_index[1]])
                )

            output_layout = dimension_merging(
                Layout(
                    term,
                    new_rolls,
                    output_dims,
                    a_kernel.layout.offset,
                    a_kernel.layout.n,
                    a_kernel.layout.secret or b_kernel.layout.secret,
                )
            )
            output_kernel = Kernel(KernelOp.MATMUL, [a_kernel, b_kernel], output_layout)
            return output_kernel
        case _:
            raise NotImplementedError(term.op)


def apply_sum_rolls(term, replicated_kernels):
    # if the roll_flag is set, apply rolls to move summation dimensions the vector dimensions
    rolled = []
    for kernels in replicated_kernels:
        # apply rolls to move summation to ciphertext dimensions
        rolled_a_kernel = apply_sum_roll(term, copy(kernels[0]))
        rolled_b_kernel = apply_sum_roll(term, copy(kernels[1]))

        # add kernel pairs with successful rolls
        if rolled_a_kernel and rolled_b_kernel:
            rolled.append(tuple([rolled_a_kernel, rolled_b_kernel]))
        if rolled_b_kernel:
            rolled.append(tuple([kernels[0], rolled_b_kernel]))
        if rolled_a_kernel:
            rolled.append(tuple([rolled_a_kernel, kernels[1]]))
    return rolled


def gen_binop(term, cs_kernels, shapes, roll_flag):
    # get alignment
    alignment = get_dim_alignment(term, shapes)

    # TODO: remove this
    # # fast path:
    # if term.op == TensorOp.ADD:
    #     replicated_kernels = []
    #     for a in cs_kernels[0][:5]:
    #         for b in cs_kernels[1][:5]:
    #             a_cs_placeholder = Kernel(KernelOp.CS, [0], cs_kernels[0][0].layout)
    #             b_cs_placeholder = Kernel(KernelOp.CS, [1], cs_kernels[1][0].layout)
    #             replicated_kernels.append(tuple(
    #                         replicate_dimensions(
    #                             a_cs_placeholder, b_cs_placeholder, shapes, alignment
    #                         )
    #                     )
    #                 )
    #     output_kernels = set()
    #     for kernels in replicated_kernels:
    #         if check_alignment(term, alignment, kernels[0], kernels[1]):
    #             output_kernel = output_layout(
    #                 term, alignment, kernels[0], kernels[1]
    #             )
    #             # if not output_kernel.layout.rolls:
    #                 # compacted_kernel = find_compaction(output_kernel)
    #                 # output_kernels.add(compacted_kernel)
    #             output_kernels.add(output_kernel)

    #         if output_kernels:
    #             return output_kernels

    # replicate dimensions such that tensor dimensions match in extent
    replicated_kernels = []
    for a in cs_kernels[0]:
        for b in cs_kernels[1]:
            # create placeholders
            a_cs_placeholder = Kernel(KernelOp.CS, [0], a.layout)
            b_cs_placeholder = Kernel(KernelOp.CS, [1], b.layout)
            replicated_kernels.append(
                tuple(
                    replicate_dimensions(
                        a_cs_placeholder, b_cs_placeholder, shapes, alignment
                    )
                )
            )

    # if the roll_flag is set, apply rolls to move summation dimensions the vector dimensions
    if roll_flag and term.op == TensorOp.MATMUL:
        replicated_kernels += apply_sum_rolls(term, replicated_kernels)

    output_kernels = set()
    for kernels in replicated_kernels:
        # add conversions or rolls to align layouts
        matched_layouts = match_layout(term, kernels, alignment, roll_flag)

        for matched_a_kernel, matched_b_kernel in matched_layouts:
            # find output layout
            output_kernel = output_layout(
                term, alignment, matched_a_kernel, matched_b_kernel
            )
            # try compacting if no rolls are applied:
            if not output_kernel.layout.rolls:
                compacted_kernel = find_compaction(output_kernel)
                output_kernels.add(compacted_kernel)
            else:
                output_kernels.add(output_kernel)
    # print("len output:", len(output_kernels))
    return output_kernels
