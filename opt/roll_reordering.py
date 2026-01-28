"""
Roll reordering optimization for FHE kernels.

This module implements roll reordering optimization that reorders
consecutive roll operations to improve efficiency. The optimization
focuses on reordering roll operations to minimize the total cost
of data movement and rotation operations.

Key Concepts:
- Roll Reordering: Reordering consecutive roll operations
- Cost Optimization: Minimizing the cost of roll sequences
- Operation Efficiency: Improving the efficiency of data movement
"""

from copy import deepcopy as copy

from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def roll_reordering(kernel):
    """
    Apply roll reordering optimization to consecutive roll operations.

    This function reorders consecutive roll operations to improve
    efficiency by minimizing the total cost of data movement.

    Args:
        kernel: Kernel to apply roll reordering to

    Returns:
        Optimized kernel with reordered roll operations
    """
    roll_kernels = []
    for term in kernel.post_order():
        if term.op != KernelOp.ROLL:
            roll_kernels = []
        else:
            roll_kernels.append(term)

    if len(roll_kernels) == 2:
        first_roll = roll_kernels[0].cs[0]
        second_roll = roll_kernels[1].cs[0]
        first_kernel_layout = Layout(
            kernel.layout.term,
            [second_roll],
            kernel.layout.get_dims(),
            kernel.layout.n,
            kernel.layout.secret,
        )
        first_kernel = Kernel(
            KernelOp.ROLL, [second_roll, kernel.cs[1].cs[1]], first_kernel_layout
        )

        second_kernel_layout = Layout(
            kernel.layout.term,
            [second_roll, first_roll],
            kernel.layout.get_dims(),
            kernel.layout.n,
            kernel.layout.secret,
        )
        second_kernel = Kernel(
            KernelOp.ROLL, [first_roll, first_kernel], second_kernel_layout
        )
        return [kernel, second_kernel]
    else:
        return [kernel]


def run_roll_reordering(candidate):
    """
    Some rolls can be reordered with no adverse affects
    """
    update_map = {}
    for kernel in candidate.post_order():
        # update kernel cs
        kernels = set([kernel])
        for i, cs in enumerate(kernel.cs):
            if cs in update_map:
                for k in update_map[cs]:
                    new_kernel = copy(kernel)
                    new_kernel.cs[i] = k
                    kernels.add(new_kernel)

        # update kernel with rewrites
        new_kernels = set()
        for k in kernels:
            match k.op:
                case KernelOp.ROLL:
                    for reordered in roll_reordering(k):
                        new_kernels.add(reordered)
                case _:
                    new_kernels.add(k)
        update_map[kernel] = new_kernels

    return update_map[candidate]
