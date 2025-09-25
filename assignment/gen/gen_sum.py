"""
Sum operation layout generation utilities.

This module provides functions for generating optimal layouts for sum
operations in FHE computations. Sum operations reduce tensors along
specified dimensions and require special handling to maintain correct
computation semantics in the homomorphic encryption domain.

Key functions:
- gen_sum: Main function for generating sum operation layouts
"""

from ir.dim import DimType
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
from ir.layout_utils import dimension_merging

from copy import deepcopy as copy


def gen_sum(term, kernels):
    """Generates layouts for sum operations along a specified dimension.
    
    This function creates kernel layouts for sum operations that reduce
    tensors along a specified dimension. The sum dimension is converted
    to an empty dimension type while preserving other dimensions.
    
    Args:
        term: TensorTerm representing the sum operation
        kernels: List of input kernels to generate sum layouts for
        
    Returns:
        Set of Kernel objects representing sum operation layouts
    """
    sum_dim_idx = term.cs[1]
    output_kernels = set()
    for kernel in kernels:
        kernel = copy(kernel)
        new_dims = []
        sum_dims = []
        for dim in copy(kernel.layout.get_dims()):
            if dim.dim == sum_dim_idx:
                sum_dims.append(copy(dim))
                new_dim = dim
                new_dim.dim_type = DimType.EMPTY
                new_dims.append(new_dim)
            else:
                new_dims.append(dim)

        new_perms = []
        for roll in kernel.layout.rolls:
            if roll.dim_to_roll.dim != sum_dim_idx:
                new_perms.append(roll)

        sum_layout = dimension_merging(
            Layout(
                term,
                new_perms,
                new_dims,
                {}, 
                kernel.layout.n,
                kernel.layout.secret,
            )
        )

        # create placeholder for transpose kernel
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
        sum_kernel = Kernel(
            KernelOp.SUM,
            [cs_placeholder, tuple(sum_dims)],
            sum_layout,
        )
        output_kernels.add(sum_kernel)
    return output_kernels
