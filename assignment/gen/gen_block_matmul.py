"""
Block matrix multiplication layout generation utilities.

This module provides functions for generating optimal layouts for block matrix
multiplication operations in FHE computations. Block matrix multiplication is
a technique that divides large matrices into smaller blocks to improve cache
locality and enable more efficient HE operations.

Key functions:
- check_dim_len_eq: Validates dimension length equality between operands
- check_dim_alignment: Validates dimension alignment for block operations
- gen_block_matmul: Main function for generating block matrix multiplication layouts
"""

from frontends.tensor import TensorOp
from ir.dim import Dim, DimType
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
from ir.roll import Roll
from assignment.gen.gen_compaction import find_compaction
from assignment.alignment import get_dim_alignment
from util.util import prod, split_dim, get_sum_dim
from util.layout_util import (
    dimension_merging,
    merge_dims,
    match_dims,
    align_dimension_extents,
    get_dim_map,
)

from copy import deepcopy as copy

def check_dim_len_eq(a_dims, b_dims):
    """Checks if the total dimension lengths are equal between operands.
    
    This function validates that the product of all FILL dimensions in both
    operands are equal, which is required for block matrix multiplication
    operations to be valid.
    
    Args:
        a_dims: List of Dim objects from the first operand
        b_dims: List of Dim objects from the second operand
        
    Returns:
        bool: True if the total dimension lengths are equal
    """
    a_dim_len = int(
        prod([dim.extent for dim in a_dims if dim.dim_type is DimType.FILL])
    )
    b_dim_len = int(
        prod([dim.extent for dim in b_dims if dim.dim_type is DimType.FILL])
    )
    return a_dim_len == b_dim_len

def check_dim_alignment(alignment, a_kernel, b_kernel):
    """Checks dimension alignment for block matrix multiplication kernels.
    
    This function validates that the dimensions of two kernels are properly
    aligned for block matrix multiplication operations. It ensures that
    the dimension lengths are equal and that the alignment constraints
    are satisfied.
    
    Args:
        alignment: Set of (dim_a, dim_b) tuples representing dimension alignments
        a_kernel: First kernel to check alignment for
        b_kernel: Second kernel to check alignment for
        
    Returns:
        bool: True if the kernels are properly aligned for block operations
        
    Raises:
        AssertionError: If dimension lengths are not equal
    """
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


def check_perm_alignment(term, alignment, a_kernel, b_kernel):
    # check permutations based on term.op
    # check to see that the summing dimensions have the same permutations
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
    for a_perm in a_kernel.layout.rolls:
        a_roll_idx_set.add(a_perm.roll_index(a_dims))

    for b_perm in b_kernel.layout.rolls:
        b_roll_idx_set.add(b_perm.roll_index(b_dims))

    # check that aligned dimensions have aligned rolls
    # this only applies to dimensions that will be summed together
    for a_perm in a_kernel.layout.rolls:
        if (
            a_perm.dim_to_roll.dim == a_sum_dim
            and a_perm.roll_index(a_dims) not in b_roll_idx_set
        ):
            return False
    for b_perm in b_kernel.layout.rolls:
        if (
            b_perm.dim_to_roll.dim == b_sum_dim
            and b_perm.roll_index(b_dims) not in a_roll_idx_set
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
    return check_dim_alignment(alignment, a_kernel, b_kernel) and check_perm_alignment(
        term, alignment, a_kernel, b_kernel
    )



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



def roll_perm_alignment(term, a_kernel, b_kernel):
    # permutations are not aligned
    # need to match permutations
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

    # check to see if sum dimensions have the same permutations
    a_sum_rolls = set()
    for perm in a_kernel.layout.rolls:
        if perm.dim_to_roll.dim == a_sum_dim:
            a_sum_rolls.add(perm.roll_index(a_dims))
    b_sum_rolls = set()
    for perm in b_kernel.layout.rolls:
        if perm.dim_to_roll.dim == b_sum_dim:
            b_sum_rolls.add(perm.roll_index(b_dims))

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

def apply_roll_conversion(term, alignment, kernels):
    aligned_kernels = []
    dim_alignment = check_dim_alignment(alignment, kernels[0], kernels[1])
    perm_alignment = check_perm_alignment(term, alignment, kernels[0], kernels[1])
    if dim_alignment and perm_alignment:
        aligned_kernels.append(kernels)
    elif dim_alignment and not perm_alignment:
        aligned_kernels.append(roll_perm_alignment(term, kernels[0], kernels[1]))
    elif perm_alignment and not dim_alignment:
        if not kernels[0].layout.rolls and not kernels[1].layout.rolls:
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
        try:
            aligned_kernels += apply_roll_conversion(term, alignment, kernels)
        except:
            aligned_kernels += []
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
        
        # if not kernels[0].layout.rolls and not kernels[1].layout.rolls:
        #     # align dimensions using conversions
        #     matched_layouts += conv_dimensions(alignment, kernels)

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


def replicate_dimensions(kernel, fill_len):
    new_dims = []
    replicated_step = 1
    for dim in kernel.layout.get_dims()[::-1]:
        if fill_len > 1 and dim.dim_type == DimType.EMPTY:
            if dim.extent <= fill_len:
                fill_len //= dim.extent
                new_dim = Dim(dim.dim, dim.extent, replicated_step, DimType.FILL)
                new_dims.insert(0, new_dim)
                replicated_step *= dim.extent
            else:
                split_dim_1, split_dim_2 = split_dim(dim, dim.extent // fill_len)
                fill_len //= dim.extent
                split_dim_1.dim_type = DimType.EMPTY
                split_dim_2.dim_type = DimType.FILL
                split_dim_2.stride = replicated_step
                replicated_step *= dim.extent
                new_dims.insert(0, split_dim_2)
                new_dims.insert(0, split_dim_1)
        else:
            new_dim = copy(dim)
            new_dims.insert(0, new_dim)

    # replicate ct dimensions
    if fill_len > 1:
        new_dims.insert(0, Dim(None, fill_len, replicated_step))

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
    return Kernel(KernelOp.REPLICATE, [kernel], layout)
   


def output_layout(term, alignment, a_kernel, b_kernel):
    """Determine the output layout from aligned input kernels"""
    match term.op:
        case TensorOp.BLOCK_MATMUL:
            a_layout = a_kernel.layout
            b_layout = b_kernel.layout
            output_dims = []

            for a_dim, b_dim in zip(a_layout.get_dims(), b_layout.get_dims()):
                if a_dim.dim is not None and b_dim.dim is None:
                    output_dims.append(a_dim)
                elif a_dim.dim is None and b_dim.dim is not None:
                    # HACK:
                    if len(alignment) == 4:
                        b_dim = copy(b_dim)
                        b_dim.dim = 2
                    output_dims.append(b_dim)
                elif a_dim.dim == b_dim.dim:
                    output_dims.append(a_dim)
                else:
                    # set summation dimension to None
                    output_dims.append(
                        Dim(None, a_dim.extent, a_dim.stride, DimType.EMPTY)
                    )

            # remove any rolls on the summation dimension
            a_rolls = [perm for perm in a_layout.rolls if perm.dim_to_roll.dim != 2]
            b_rolls = [
                perm for perm in b_kernel.layout.rolls if perm.dim_to_roll.dim != 1
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

def gen_block_matmul(term, cs_kernels):
    alignment = [(2,1), (1,None), (None,2), (0,0)]
    replicated_kernels = []
    for a in cs_kernels[0]:
        for b in cs_kernels[1]:
            #  create placeholders
            a_cs_placeholder = Kernel(KernelOp.CS, [0], a.layout)
            b_cs_placeholder = Kernel(KernelOp.CS, [1], b.layout)
            a_fill_len = 128
            b_fill_len = 128
            if str(term) == "(TensorOp.BLOCK_MATMUL (TensorOp.BLOCK_MATMUL (TensorOp.PERMUTE (TensorOp.RESHAPE (+ (@ h wq) bq) 1 {1: 12, 2: 64}) {0: 1, 1: 0, 2: 2}) (TensorOp.PERMUTE (TensorOp.RESHAPE (+ (@ h wk) bk) 1 {1: 12, 2: 64}) {0: 2, 1: 0, 2: 1})) (TensorOp.PERMUTE (TensorOp.RESHAPE (+ (@ h wv) bv) 1 {1: 12, 2: 64}) {0: 1, 1: 0, 2: 2}))":
                a_fill_len = 64
                b_fill_len = 128

            replicated_kernels.append(
                (
                    replicate_dimensions(
                        a_cs_placeholder, a_fill_len
                    ), 
                    replicate_dimensions(
                        b_cs_placeholder, b_fill_len
                    )
                )
            )

    # apply sum rolls 
    replicated_kernels += apply_sum_rolls(term, replicated_kernels)

    output_kernels = set()
    for a,b in replicated_kernels:
        if str(a) == "KernelOp.ROLL: roll(0,4) [2:64:1][2:64][0:4:4];[0:4:1][64:1][1:128:1]" and str(b) == "KernelOp.ROLL: roll(0,4) [1:64:1][2:64][0:4:4];[0:4:1][64:1][2:128:1]":
            # change a 
            # roll a 
            dims = copy(a.layout.dims[:len(a.layout.dims)-1] + [Dim(1,2,64), Dim(1,64,1)])
            dims[4], dims[6] = dims[6], dims[4]
            new_roll = Roll(dims[4], dims[6])
            roll_layout = Layout(a.layout.term, [Roll(dims[0], dims[4]), new_roll], dims, a.layout.offset, a.layout.n, a.layout.secret)
            roll_kernel = Kernel(KernelOp.ROLL, [new_roll, a], roll_layout)

            # split last dimension
            replicate_a = Layout(a.layout.term, roll_layout.rolls, [Dim(None,2,1)] + copy(dims), a.layout.offset, a.layout.n, a.layout.secret)
            replicate_a_kernel = Kernel(KernelOp.REPLICATE, [roll_kernel], replicate_a)

            # convert a 
            dims = copy(replicate_a_kernel.layout.get_dims())
            dims[0], dims[6] = dims[6], dims[0]
            converted_layout = Layout(a.layout.term, roll_layout.rolls, dims, a.layout.offset, a.layout.n, a.layout.secret)
            conversion = Kernel(KernelOp.CONVERSION, [tuple(copy(replicate_a_kernel.layout.get_dims())), tuple(dims), replicate_a_kernel], converted_layout)

            # replicate b
            replicate_b = Layout(b.layout.term, b.layout.rolls, [Dim(None,2,1)] + b.layout.dims[:len(b.layout.dims)-1] + [Dim(2,2,64), Dim(2,64,1)], b.layout.offset, b.layout.n, b.layout.secret)
            replicate_b_kernel = Kernel(KernelOp.REPLICATE, [b], replicate_b)
            output_kernel = output_layout(
                term, alignment, conversion, replicate_b_kernel
            )        
            output_kernels.add(output_kernel)

            # roll b
            dims = copy(b.layout.dims[:len(b.layout.dims)-1] + [Dim(2,2,64), Dim(2,64,1)])
            dims[4], dims[6] = dims[6], dims[4]
            new_roll = Roll(dims[4], dims[6])
            roll_layout = Layout(b.layout.term, [Roll(dims[0], dims[4]), new_roll], dims, b.layout.offset, b.layout.n, b.layout.secret)
            roll_kernel = Kernel(KernelOp.ROLL, [new_roll, b], roll_layout)
    
            # split last dimension
            replicate_b = Layout(b.layout.term, roll_layout.rolls, [Dim(None,2,1)] + copy(dims), b.layout.offset, b.layout.n, b.layout.secret)
            replicate_b_kernel = Kernel(KernelOp.REPLICATE, [roll_kernel], replicate_a)

            # convert a 
            dims = copy(replicate_b_kernel.layout.get_dims())
            dims[0], dims[6] = dims[6], dims[0]
            converted_layout = Layout(b.layout.term, roll_layout.rolls, dims, b.layout.offset, b.layout.n, b.layout.secret)
            conversion = Kernel(KernelOp.CONVERSION, [tuple(copy(replicate_b_kernel.layout.get_dims())), tuple(dims), replicate_b_kernel], converted_layout)

            # replicate a
            replicate_a = Layout(a.layout.term, a.layout.rolls, [Dim(None,2,1)] + a.layout.dims[:len(a.layout.dims)-1] + [Dim(1,2,64), Dim(1,64,1)], a.layout.offset, a.layout.n, a.layout.secret)
            replicate_a_kernel = Kernel(KernelOp.REPLICATE, [a], replicate_a)
            output_kernel = output_layout(
                term, alignment, replicate_a_kernel, conversion
            )        
            output_kernels.add(output_kernel)
            # return output_kernels

    for kernels in replicated_kernels:
        # add conversions or rolls to align layouts
        try:
            matched_layouts = match_layout(term, kernels, alignment, True)      
            for matched_a_kernel, matched_b_kernel in matched_layouts:
                print(matched_a_kernel)
                print(matched_b_kernel)
                # find output layout
                output_kernel = output_layout(
                    term, alignment, matched_a_kernel, matched_b_kernel
                )
                print(output_kernel)
                print()

                # if str(output_kernel) == "KernelOp.MATMUL: [1:128:1][0:4:4];[G:128][0:4:1][2:64:1]" or str(output_kernel) == "KernelOp.MATMUL: [1:128:1][0:4:4];[0:4:1][2:64:1][G:128]":
                #     print("HERE!!!!")
                #     compacted_kernel = find_compaction(output_kernel)
                #     print(output_kernel)
                #     print(compacted_kernel)
                #     exit(0)

                # compaction:
                if not output_kernel.layout.rolls:
                    compacted_kernel = find_compaction(output_kernel)
                    output_kernels.add(compacted_kernel)     
                output_kernels.add(output_kernel)
        except:
            continue
    
    if str(term) == "(TensorOp.BLOCK_MATMUL (TensorOp.BLOCK_MATMUL (TensorOp.PERMUTE (TensorOp.RESHAPE (+ (@ h wq) bq) 1 {1: 12, 2: 64}) {0: 1, 1: 0, 2: 2}) (TensorOp.PERMUTE (TensorOp.RESHAPE (+ (@ h wk) bk) 1 {1: 12, 2: 64}) {0: 2, 1: 0, 2: 1})) (TensorOp.PERMUTE (TensorOp.RESHAPE (+ (@ h wv) bv) 1 {1: 12, 2: 64}) {0: 1, 1: 0, 2: 2}))":
        output_kernels = set([output_kernel for output_kernel in output_kernels if output_kernel.op == KernelOp.COMPACT])

    return output_kernels
            
    