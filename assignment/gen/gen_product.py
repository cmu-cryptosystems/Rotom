"""Product (reduce-mul) layout generation — mirrors ``gen_sum`` with ``KernelOp.PRODUCT``."""

from copy import deepcopy as copy

from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging


def gen_product(term, kernels):
    """Layouts for product-reduce along an axis (same layout transform as sum-reduce)."""
    dim_idx = term.cs[1]
    output_kernels = set()
    for kernel in kernels:
        kernel = copy(kernel)
        new_dims = []
        prod_dims = []

        active_dim_indices = sorted(
            {
                d.dim
                for d in copy(kernel.layout.get_dims())
                if d.dim is not None and d.dim_type != DimType.EMPTY
            }
        )
        if not (0 <= int(dim_idx) < len(active_dim_indices)):
            continue
        dim_value = active_dim_indices[int(dim_idx)]

        for dim in copy(kernel.layout.get_dims()):
            if dim.dim == dim_value and dim.dim_type != DimType.EMPTY:
                prod_dims.append(copy(dim))
                new_dims.append(Dim(None, dim.extent, dim.stride, DimType.EMPTY))
            else:
                new_dims.append(dim)

        new_rolls = []
        for roll in kernel.layout.rolls:
            if (
                roll.dim_to_roll.dim == dim_value
                or roll.dim_to_roll_by.dim == dim_value
            ):
                continue
            new_rolls.append(roll)

        out_layout = dimension_merging(
            Layout(
                term,
                new_rolls,
                new_dims,
                kernel.layout.n,
                kernel.layout.secret,
            )
        )
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
        prod_kernel = Kernel(
            KernelOp.PRODUCT,
            [cs_placeholder, tuple(prod_dims)],
            out_layout,
        )
        output_kernels.add(prod_kernel)
    return output_kernels
