"""
Reshape operation layout generation utilities.

This module provides functions for generating optimal layouts for reshape
operations in FHE computations. Reshape operations change the shape of
tensors while preserving the total number of elements and require careful
handling to maintain correct computation semantics in the homomorphic encryption domain.

Key functions:
- gen_reshape: Main function for generating reshape operation layouts
"""

from copy import deepcopy as copy

from ir.dim import DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from util.layout_util import get_extent_dims
from util.util import prod, round_to_ceiling_power_of_2


def gen_reshape(term, cs_kernels):
    """Generates layouts for reshape operations.

    This function creates kernel layouts for reshape operations that change
    the shape of tensors while preserving the total number of elements.
    The reshape operation applies a new dimension mapping based on the
    specified shape changes.

    Args:
        term: TensorTerm representing the reshape operation
        cs_kernels: List of input kernels to generate reshape layouts for

    Returns:
        Set of Kernel objects representing reshape operation layouts
    """
    rounded_shapes_map = {}
    for k, v in term.cs[2].items():
        rounded_shapes_map[k] = round_to_ceiling_power_of_2(v)
    strides = []
    for k, v in rounded_shapes_map.items():
        strides.append((k, v))

    # create map where set dimensions based on stride
    stride_map = []
    stride = 1
    for s in strides[::-1]:
        stride_map.insert(0, (s[0], stride))
        stride *= s[1]

    kernels = []
    for cs_kernel in cs_kernels:
        new_dims = []
        extent_dims = get_extent_dims(cs_kernel.layout.get_dims())
        for dim in extent_dims:
            dim = copy(dim)
            if dim.dim == term.cs[1]:
                for s in stride_map:
                    if dim.stride >= s[1]:
                        dim.dim = s[0]
                        break
                new_dims.append(dim)
            else:
                new_dims.append(dim)

        for k, v in rounded_shapes_map.items():
            if k != term.cs[1]:
                reshape_amt = v
        for new_dim in new_dims:
            if new_dim.dim == term.cs[1]:
                new_dim.stride //= reshape_amt

        cs_placeholder = Kernel(KernelOp.CS, [0], cs_kernel.layout)

        new_layout = dimension_merging(
            Layout(
                term,
                cs_kernel.layout.rolls,
                new_dims,
                cs_kernel.layout.n,
                cs_kernel.layout.secret,
            )
        )
        new_kernel = Kernel(KernelOp.RESHAPE, [cs_placeholder], new_layout)
        kernels.append(new_kernel)
    return kernels
