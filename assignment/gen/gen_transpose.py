"""
Transpose operation layout generation utilities.

This module provides functions for generating optimal layouts for transpose
operations in FHE computations. Transpose operations swap dimensions of
tensors and require careful handling to maintain correct computation
semantics in the homomorphic encryption domain.

Key functions:
- gen_transpose: Main function for generating transpose operation layouts
"""

from copy import deepcopy as copy

from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_transpose(term, kernels):
    """Generates layouts for transpose operations.

    This function creates kernel layouts for transpose operations that swap
    the dimensions of tensors. The transpose operation swaps dimension indices
    using XOR with 1 (0 becomes 1, 1 becomes 0).

    Args:
        term: TensorTerm representing the transpose operation
        kernels: List of input kernels to generate transpose layouts for

    Returns:
        Set of Kernel objects representing transpose operation layouts

    Raises:
        NotImplementedError: If kernels have roll operations (not yet supported)
    """
    output_kernels = set()
    for kernel in kernels:
        if kernel.layout.rolls:
            raise NotImplementedError("still need to explore transpose with rolls")

        new_dims = []
        for dim in kernel.layout.get_dims():
            if dim.dim is not None:
                new_dims.append(
                    Dim(
                        dim.dim ^ 1,
                        dim.extent,
                        dim.stride,
                        dim.dim_type,
                    )
                )
            else:
                new_dims.append(copy(dim))

        transposed_layout = Layout(
            term,
            copy(kernel.layout.rolls),
            new_dims,
            {},
            kernel.layout.n,
            kernel.layout.secret,
        )

        # create placeholder for transpose kernel
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)

        transposed_kernel = Kernel(
            KernelOp.TRANSPOSE, [cs_placeholder], transposed_layout
        )
        output_kernels.add(transposed_kernel)
    return output_kernels
