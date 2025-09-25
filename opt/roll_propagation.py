"""
Roll propagation optimization for FHE kernels.

This module implements roll propagation optimization that moves roll operations
to optimal positions in the computation graph. The optimization focuses on
moving cheaper roll operations above more expensive replication operations
to reduce overall computation cost.

Key Concepts:
- Roll Propagation: Moving roll operations to optimal positions
- Replication Optimization: Avoiding expensive replication after rolls
- Cost Reduction: Minimizing the cost of data movement operations
- Path Analysis: Analyzing computation paths for optimization opportunities
"""

from ir.roll import Roll
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
from ir.dim import DimType

from copy import deepcopy as copy
from util.layout_util import match_dims


def check_replication_on_child_path(kernel):
    """
    Check if there is a replication operation on the child path.
    
    This function recursively traverses the kernel tree to determine
    if there is a replication operation along the path, which is
    important for roll propagation optimization.
    
    Args:
        kernel: Kernel to check for replication operations
        
    Returns:
        Boolean indicating if replication is found on the path
    """
    if kernel.op not in [
        KernelOp.ROLL,
        KernelOp.ROT_ROLL,
        KernelOp.BSGS_ROLL,
        KernelOp.REPLICATE,
    ]:
        return False
    elif kernel.op == KernelOp.REPLICATE:
        return True
    else:
        return check_replication_on_child_path(kernel.cs[1])


def check_no_shared_rolls(kernel):
    """
    Check that no dimensions are rolled by multiple other dimensions.
    
    This function ensures that each dimension is only rolled by one other
    dimension, which is a requirement for roll propagation optimization.
    
    Args:
        kernel: Kernel to check for shared rolls
        
    Returns:
        Boolean indicating if no shared rolls are found
    """
    seen = set()
    for roll in kernel.layout.rolls:
        if roll.dim_to_roll in seen:
            return False
        seen.add(roll.dim_to_roll_by)
    return True


def roll_propogation(kernel):
    """
    Apply roll propagation optimization to move cheaper rolls above replication.
    
    This function implements the core roll propagation optimization that
    moves roll operations to optimal positions in the computation graph.
    The goal is to move cheaper roll operations above more expensive
    replication operations to reduce overall cost.
    
    Args:
        kernel: Kernel to apply roll propagation to
        
    Returns:
        Optimized kernel with rolls moved to optimal positions
    """

    # get all cs kernels (that are rolls) and the replicate kernel
    cs_kernels = []
    cs_kernel = kernel.cs[1]
    while cs_kernel.op != KernelOp.REPLICATE:
        cs_kernels.insert(0, cs_kernel)
        cs_kernel = cs_kernel.cs[1]
    matched_replicate_kernel_dims = match_dims(
        cs_kernel.layout.get_dims(), kernel.layout.get_dims()
    )

    replicate_kernel = Kernel(
        cs_kernel.op,
        cs_kernel.cs,
        Layout(
            cs_kernel.layout.term,
            cs_kernel.layout.rolls,
            matched_replicate_kernel_dims,
            cs_kernel.layout.offset,
            cs_kernel.layout.n,
            cs_kernel.layout.secret,
        ),
    )

    # get base kernel
    matched_base_kernel_dims = match_dims(
        replicate_kernel.cs[0].layout.get_dims(), kernel.layout.get_dims()
    )

    base_kernel = Kernel(
        replicate_kernel.cs[0].op,
        replicate_kernel.cs[0].cs,
        Layout(
            replicate_kernel.cs[0].layout.term,
            replicate_kernel.cs[0].layout.rolls,
            matched_base_kernel_dims,
            replicate_kernel.cs[0].layout.offset,
            replicate_kernel.cs[0].layout.n,
            replicate_kernel.cs[0].layout.secret,
        ),
    )

    # apply propogated roll to base kernel
    roll = kernel.cs[0]
    roll_dims = [roll.dim_to_roll, roll.dim_to_roll_by]

    # assert that the roll is a slot only roll
    slot_indices = []
    for roll_dim in roll_dims:
        assert roll_dim in kernel.layout.slot_dims
        slot_indices.append(kernel.layout.slot_dims.index(roll_dim))

    # create new roll
    base_dims = base_kernel.layout.slot_dims
    dim_to_roll_index = base_dims.index(roll_dims[0])
    slot_indices.remove(dim_to_roll_index)
    dim_to_roll_by_index = slot_indices[0]
    ct_dim_len = len(base_kernel.layout.ct_dims)

    dims = base_kernel.layout.get_dims()

    dim_to_roll_by = dims[dim_to_roll_by_index + ct_dim_len]
    if dim_to_roll_by.dim_type == DimType.EMPTY:
        # requires replicate first, so just exit
        return kernel

    new_roll = Roll(
        dims[dim_to_roll_index + ct_dim_len],
        dims[dim_to_roll_by_index + ct_dim_len],
    )

    # create new rolled layouts
    new_roll_layout = Layout(
        base_kernel.layout.term,
        base_kernel.layout.rolls + [new_roll],
        dims,
        base_kernel.layout.offset,
        base_kernel.layout.n,
        base_kernel.layout.secret,
    )
    new_roll_kernel = Kernel(
        KernelOp.BSGS_ROLL, [new_roll, base_kernel], new_roll_layout
    )

    # replicate kernel
    replicate_kernel_dims = replicate_kernel.layout.get_dims()

    # update roll!
    replicated_rolls = []
    for roll in new_roll_layout.rolls:
        # filled dim_to_roll_by
        filled_dim_to_roll_by = copy(roll.dim_to_roll_by)
        filled_dim_to_roll_by.dim_type = DimType.FILL
        if (
            roll.dim_to_roll_by not in replicate_kernel_dims
            and filled_dim_to_roll_by in replicate_kernel_dims
        ):
            replicated_rolls.append(Roll(copy(roll.dim_to_roll), filled_dim_to_roll_by))
        else:
            replicated_rolls.append(roll)

    replicate_layout = Layout(
        new_roll_kernel.layout.term,
        replicated_rolls,
        replicate_kernel_dims,
        new_roll_kernel.layout.offset,
        new_roll_kernel.layout.n,
        new_roll_kernel.layout.secret,
    )
    replicate_kernel = Kernel(KernelOp.REPLICATE, [new_roll_kernel], replicate_layout)

    # assume remaining rolls can be directly applied to the replicated kernel
    tmp_kernel = replicate_kernel
    for cs_kernel in cs_kernels:
        diff = 0
        if len(cs_kernel.layout.get_dims()) != len(tmp_kernel.layout.get_dims()):
            diff = len(cs_kernel.layout.get_dims()) - len(tmp_kernel.layout.get_dims())

        # get original rolls
        tmp_indices = [
            list(roll.roll_index(tmp_kernel.layout.get_dims()))
            for roll in tmp_kernel.layout.rolls
        ]
        if diff:
            assert diff > 0
            for index in tmp_indices:
                index[0] += diff
                index[1] += diff

        # update roll
        cs_dims = cs_kernel.layout.get_dims()
        new_rolls = []
        for index in tmp_indices:
            new_rolls.append(Roll(cs_dims[index[0]], cs_dims[index[1]]))

        cs_roll = cs_kernel.cs[0]
        new_layout = Layout(
            cs_kernel.layout.term,
            new_rolls + [cs_roll],
            cs_dims,
            cs_kernel.layout.offset,
            cs_kernel.layout.n,
            cs_kernel.layout.secret,
        )
        new_cs_kernel = Kernel(cs_kernel.op, [cs_roll, tmp_kernel], new_layout)
        tmp_kernel = new_cs_kernel
    return tmp_kernel


def run_roll_propogation(candidate):
    update_map = {}
    for kernel in candidate.post_order():
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                kernel.cs[i] = update_map[cs]
        match kernel.op:
            case KernelOp.ROLL:
                roll = kernel.cs[0]
                slot_dims = kernel.layout.slot_dims
                if (
                    roll.dim_to_roll in slot_dims
                    and roll.dim_to_roll_by in slot_dims
                    and check_replication_on_child_path(kernel)
                ):
                    try:
                        update_map[kernel] = roll_propogation(kernel)
                    except:
                        update_map[kernel] = kernel
            case _:
                update_map[kernel] = kernel
    return candidate
