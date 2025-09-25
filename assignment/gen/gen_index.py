"""
Index operation layout generation utilities.

This module provides functions for generating optimal layouts for index
operations in FHE computations. Index operations extract specific elements
or slices from tensors and require careful handling to maintain correct
computation semantics in the homomorphic encryption domain.

Key functions:
- gen_index: Main function for generating index operation layouts
"""

from assignment.gen.gen_compaction import find_compaction
from ir.dim import DimType
from ir.layout import Layout
from ir.kernel import Kernel, KernelOp
from copy import deepcopy as copy
from ir.layout_utils import dimension_merging


def gen_index(term, kernels):
    """Generates layouts for index operations.
    
    This function creates kernel layouts for index operations that extract
    specific elements or slices from tensors. The index operation removes
    the first dimension and adjusts the remaining dimensions accordingly.
    
    Args:
        term: TensorTerm representing the index operation
        kernels: List of input kernels to generate index layouts for
        
    Returns:
        Set of Kernel objects representing index operation layouts
    """
    output_kernels = set()
    for kernel in kernels:
        # no rolls
        if not kernel.layout.rolls:
            # create next offset
            next_dim = len(kernel.layout.offset)
            next_offset = copy(kernel.layout.offset)
            next_offset[next_dim] = term.cs[1]

            # adjust dimensions
            new_dims = []
            for dim in kernel.layout.get_dims():
                dim = copy(dim)
                if dim.dim is None:
                    new_dims.append(dim)
                elif dim.dim == 0:
                    dim.dim = None
                    dim.dim_type = DimType.EMPTY
                    new_dims.append(dim)
                else:
                    dim.dim -= 1
                    new_dims.append(dim)

            indexed_layout = dimension_merging(
                Layout(
                    term,
                    copy(kernel.layout.rolls),
                    new_dims,
                    next_offset,
                    kernel.layout.n,
                    kernel.layout.secret,
                )
            )

            # create placeholder for indexed kernel
            cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
            indexed_kernel = Kernel(KernelOp.INDEX, [cs_placeholder], indexed_layout)

            # compact index kernel
            compacted_kernel = find_compaction(indexed_kernel)
            output_kernels.add(compacted_kernel)
    return output_kernels
