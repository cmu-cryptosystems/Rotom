"""Layout generation for MEAN (reduce-mean over one or more axes, keepdims=True)."""

from copy import deepcopy as copy
from math import prod

from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging


def gen_mean(term, kernels):
    """Like ``gen_sum`` but reduces all listed axes and records ``1/count`` for lowering."""
    axes = term.cs[1]
    if isinstance(axes, int):
        axes = (axes,)
    axes = tuple(int(a) for a in axes)
    output_kernels = set()
    for kernel in kernels:
        kernel = copy(kernel)
        new_dims = []
        mean_dims = []

        active_dim_indices = sorted(
            {
                d.dim
                for d in copy(kernel.layout.get_dims())
                if d.dim is not None and d.dim_type != DimType.EMPTY
            }
        )
        sum_dim_values = []
        for dim_idx in axes:
            if not (0 <= int(dim_idx) < len(active_dim_indices)):
                sum_dim_values = None
                break
            sum_dim_values.append(active_dim_indices[int(dim_idx)])
        if sum_dim_values is None:
            continue
        sum_dim_set = frozenset(sum_dim_values)

        for dim in copy(kernel.layout.get_dims()):
            if dim.dim in sum_dim_set and dim.dim_type != DimType.EMPTY:
                mean_dims.append(copy(dim))
                new_dims.append(Dim(None, dim.extent, dim.stride, DimType.EMPTY))
            else:
                new_dims.append(dim)

        if not mean_dims:
            continue

        new_rolls = []
        for roll in kernel.layout.rolls:
            if (
                roll.dim_to_roll.dim in sum_dim_set
                or roll.dim_to_roll_by.dim in sum_dim_set
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
        count = prod(float(d.extent) for d in mean_dims)
        inv_scale = 1.0 / count if count > 0 else 0.0
        cs_placeholder = Kernel(KernelOp.CS, [0], kernel.layout)
        mean_kernel = Kernel(
            KernelOp.MEAN,
            [cs_placeholder, tuple(mean_dims), inv_scale],
            out_layout,
        )
        output_kernels.add(mean_kernel)
    return output_kernels
