"""
Index operation layout generation utilities.

This module provides functions for generating optimal layouts for index
operations in FHE computations. Index operations extract specific elements
or slices from tensors and require careful handling to maintain correct
computation semantics in the homomorphic encryption domain.

Key functions:
- gen_index: Main function for generating index operation layouts
"""

from copy import deepcopy as copy
import math

from assignment.gen.gen_compaction import find_compaction
from ir.dim import DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from util.shape_util import get_term_shape


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

    def _normalize_index_spec(index_spec, rank: int):
        if isinstance(index_spec, tuple):
            items = list(index_spec)
        else:
            items = [index_spec]
        if len(items) < rank:
            items += [slice(None, None, None)] * (rank - len(items))
        return items[:rank]

    def _slice_new_extent(extent: int, slc: slice) -> int:
        start = 0 if slc.start is None else int(slc.start)
        stop = extent if slc.stop is None else int(slc.stop)
        step = 1 if slc.step is None else int(slc.step)
        if step <= 0:
            raise NotImplementedError("INDEX slice step must be positive")
        if stop <= start:
            return 0
        return int(math.ceil((stop - start) / step))

    output_kernels = set()
    for kernel in kernels:
        # no rolls
        if not kernel.layout.rolls:
            input_shape = get_term_shape(term.cs[0])
            rank = len(input_shape)
            idx_items = _normalize_index_spec(term.cs[1], rank)
            # Map old logical dim -> new logical dim after integer-index removals.
            old_to_new = {}
            next_dim = 0
            for d in range(rank):
                if isinstance(idx_items[d], int):
                    old_to_new[d] = None
                else:
                    old_to_new[d] = next_dim
                    next_dim += 1

            # adjust dimensions
            new_dims = []
            for dim in kernel.layout.get_dims():
                dim = copy(dim)
                if dim.dim is None:
                    new_dims.append(dim)
                else:
                    original_dim = dim.dim
                    spec = idx_items[original_dim]
                    if isinstance(spec, int):
                        dim.dim = None
                        dim.dim_type = DimType.EMPTY
                    else:
                        dim.dim = old_to_new[original_dim]
                        # Scale layout stride when slicing with step > 1.
                        step = 1 if spec.step is None else int(spec.step)
                        dim.stride *= step
                        # Conservative: only rewrite extent when this logical dim appears once.
                        num_fragments = sum(
                            1
                            for d in kernel.layout.get_dims()
                            if d.dim is not None and d.dim == original_dim
                        )
                        if num_fragments == 1:
                            dim.extent = _slice_new_extent(dim.extent, spec)
                    new_dims.append(dim)

            indexed_layout = dimension_merging(
                Layout(
                    term,
                    copy(kernel.layout.rolls),
                    new_dims,
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
