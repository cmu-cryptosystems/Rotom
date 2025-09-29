"""
Rotation roll optimization for FHE kernels.

This module implements rotation roll optimization that converts certain
roll operations into more efficient rotation operations. The optimization
identifies roll operations that can be represented as simple rotations
to reduce computation cost.

Key Concepts:
- Rotation Conversion: Converting rolls to rotations where possible
- Cost Reduction: Reducing the cost of data movement operations
- Operation Simplification: Simplifying complex roll operations
- Efficiency Improvement: Improving overall operation efficiency
"""

from ir.kernel import KernelOp


def run_rot_roll(candidate):
    """
    Apply rotation roll optimization to convert rolls to rotations.

    This optimization identifies roll operations that can be represented
    as simple rotations, reducing computation cost and improving efficiency.

    Args:
        candidate: Kernel to apply rotation roll optimization to

    Returns:
        Optimized kernel with rolls converted to rotations where applicable
    """
    update_map = {}
    for kernel in candidate.post_order():
        # update kernel cs
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                kernel.cs[i] = update_map[cs]

        # update kernel with rewrites
        match kernel.op:
            case KernelOp.ROLL | KernelOp.SPLIT_ROLL:
                ct_dims = kernel.layout.ct_dims
                dims = kernel.layout.get_dims()
                roll = kernel.cs[0]
                if (
                    len(ct_dims)
                    and dims.index(roll.dim_to_roll_by) == len(ct_dims)
                    and len(kernel.layout.rolls) == 1
                ):
                    kernel.op = KernelOp.ROT_ROLL
                update_map[kernel] = kernel
            case _:
                update_map[kernel] = kernel
    return candidate
