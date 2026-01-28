"""
Permute operation layout generation utilities.

This module provides functions for generating optimal layouts for permute
operations in FHE computations. Permute operations reorder dimensions of
tensors according to a specified permutation and require careful handling
to maintain correct computation semantics in the homomorphic encryption domain.

Key functions:
- gen_permute: Main function for generating permute operation layouts
"""

from copy import deepcopy as copy

from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_permute(term, kernels):
    """Generates layouts for permute operations.

    This function creates kernel layouts for permute operations that reorder
    dimensions of tensors according to a specified permutation mapping.
    The permutation is applied to all non-None dimensions in the layout.

    Args:
        term: TensorTerm representing the permute operation
        kernels: List of input kernels to generate permute layouts for

    Returns:
        Set of Kernel objects representing permute operation layouts
    """
    output_kernels = set()
    for kernel in kernels:
        new_dims = []
        for dim in kernel.layout.get_dims():
            if dim.dim is not None:
                new_dims.append(
                    Dim(
                        term.cs[1][dim.dim],
                        dim.extent,
                        dim.stride,
                        dim.dim_type,
                    )
                )
            else:
                new_dims.append(copy(dim))

        permuted_layout = Layout(
            term,
            copy(kernel.layout.rolls),
            new_dims,
            kernel.layout.n,
            kernel.layout.secret,
        )

        # create placeholder for permuted kernel
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
        permuted_kernel = Kernel(KernelOp.PERMUTE, [cs_placeholder], permuted_layout)
        output_kernels.add(permuted_kernel)
    return output_kernels
