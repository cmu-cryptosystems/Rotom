"""
Rescale operation layout generation utilities.

This module provides functions for generating optimal layouts for rescale
operations in FHE computations. Rescale operations divide tensors by a power of 2,
which is commonly used in homomorphic encryption to manage scale factors.

Key functions:
- gen_rescale: Main function for generating rescale operation layouts
"""

from ir.kernel import Kernel, KernelOp
from ir.layout import Layout


def gen_rescale(term, kernels):
    """Generates layouts for rescale operations.

    This function creates kernel layouts for rescale operations that divide
    tensors by 2^scale_exp. The rescale operation maintains the same layout
    as the input tensor.

    Args:
        term: TensorTerm representing the rescale operation
        kernels: List of input kernels to generate rescale layouts for

    Returns:
        Set of Kernel objects representing rescale operation layouts
    """
    output_kernels = set()
    for kernel in kernels:
        # Create placeholder for rescale kernel
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)

        # Create rescale kernel with a new layout that references the rescale term
        # The scale exponent is stored in cs[1] of the term
        scale_exp = term.cs[1]

        # Create a new layout with the rescale term as the layout term
        new_layout = Layout(
            term,  # Use the rescale term
            kernel.layout.rolls,
            kernel.layout.get_dims(),
            kernel.layout.offset,
            kernel.layout.n,
            kernel.layout.secret,
        )

        rescaled_kernel = Kernel(
            KernelOp.RESCALE, [cs_placeholder, scale_exp], new_layout
        )
        output_kernels.add(rescaled_kernel)
    return output_kernels
