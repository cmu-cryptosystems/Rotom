"""
BSGS (Baby-Step Giant-Step) matrix multiplication optimization.

This module implements BSGS optimization for matrix multiplication operations
in homomorphic encryption. BSGS reduces the number of rotations required
for matrix multiplication by reorganizing the computation pattern.

Key Concepts:
- BSGS Algorithm: Reduces rotation complexity from O(n) to O(√n)
- Roll Matching: Requires both operands to have matching roll patterns
- Plaintext Optimization: Works best when one operand is plaintext
- Rotation Reduction: Significantly reduces expensive rotation operations
"""

from ir.kernel import KernelOp


def run_bsgs_matmul(candidate):
    """
    Apply BSGS optimization to matrix multiplication kernels.

    BSGS (Baby-Step Giant-Step) can be applied to matrix multiplication
    when both layouts have the same roll pattern and one operand is
    plaintext. This optimization reduces rotation costs significantly.

    Args:
        candidate: Kernel to apply BSGS optimization to

    Returns:
        Updated kernel with BSGS optimization applied where applicable
    """
    update_map = {}
    for kernel in candidate.post_order():
        # update kernel cs
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                kernel.cs[i] = update_map[cs]

        # update kernel with rewrites
        match kernel.op:
            case KernelOp.MATMUL:
                kernel_0_roll_index = None
                kernel_1_roll_index = None
                if kernel.cs[0].layout.rolls:
                    kernel_0_roll_index = (
                        kernel.cs[0]
                        .layout.rolls[-1]
                        .roll_index(kernel.cs[0].layout.get_dims())
                    )
                if kernel.cs[1].layout.rolls:
                    kernel_1_roll_index = (
                        kernel.cs[1]
                        .layout.rolls[-1]
                        .roll_index(kernel.cs[1].layout.get_dims())
                    )

                if (
                    kernel.cs[0].op == KernelOp.ROT_ROLL
                    and kernel.cs[1].op == KernelOp.TENSOR
                    and kernel.cs[1].layout.rolls
                    and not kernel.cs[1].layout.secret
                    and kernel_0_roll_index == kernel_1_roll_index
                ):
                    # # remove the rot_roll and change the operation to BSGS_MATMUL
                    # kernel.cs[0] = kernel.cs[0].cs[1]
                    kernel.cs[0].op = KernelOp.BSGS_ROT_ROLL
                    kernel.cs.insert(0, kernel_0_roll_index)
                    kernel.op = KernelOp.BSGS_MATMUL

                elif (
                    kernel.cs[1].op == KernelOp.ROT_ROLL
                    and kernel.cs[0].op == KernelOp.TENSOR
                    and kernel.cs[0].layout.rolls
                    and not kernel.cs[0].layout.secret
                    and kernel_0_roll_index == kernel_1_roll_index
                ):
                    # # remove the rot_roll and change the operation to BSGS_MATMUL
                    # kernel.cs[1] = kernel.cs[1].cs[1]
                    kernel.cs[1].op = KernelOp.BSGS_ROT_ROLL
                    kernel.cs.insert(0, kernel_0_roll_index)
                    kernel.op = KernelOp.BSGS_MATMUL

                update_map[kernel] = kernel
            case _:
                update_map[kernel] = kernel
    return candidate
