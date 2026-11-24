"""Layout generation for MEAN (reduce-mean over one or more axes, keepdims=True)."""

from copy import deepcopy as copy
from math import prod

from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging


def _layout_dims_of_axes(axes, shape, active_dim_indices):
    """The layout dims that hold the tensor axes `axes`, or None.

    A layout has no piece for an axis of extent 1, so its dims are the
    tensor's other axes in order: for a `[C, 1, H, W]` tensor laid out on dims
    0, 2, 3, axis 2 is the second of them, and axis 1 is none. An axis of
    extent 1 contributes nothing: the mean over it is the identity.
    """
    held = list(range(len(active_dim_indices)))
    if shape is not None and len(active_dim_indices) < len(shape):
        held = [axis for axis, extent in enumerate(shape) if extent != 1]
        if len(held) != len(active_dim_indices):
            return None
    dims = []
    for axis in axes:
        if shape is not None and 0 <= axis < len(shape) and shape[axis] == 1:
            continue
        if axis not in held:
            return None
        dims.append(active_dim_indices[held.index(axis)])
    return dims


def gen_mean(term, kernels, shape=None):
    """Like ``gen_sum`` but reduces all listed axes and records ``1/count`` for lowering.

    Args:
        term: the MEAN term
        kernels: the operand's kernels
        shape: the operand's shape, which says which axes have extent 1
    """
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
        sum_dim_values = _layout_dims_of_axes(axes, shape, active_dim_indices)
        if sum_dim_values is None:
            continue
        if not sum_dim_values:
            # Every axis has extent 1: the mean is the tensor itself.
            output_kernels.add(
                Kernel(
                    KernelOp.CAST,
                    [Kernel(KernelOp.CS, [0], kernel.layout)],
                    Layout(
                        term,
                        kernel.layout.rolls,
                        kernel.layout.get_dims(),
                        kernel.layout.n,
                        kernel.layout.secret,
                    ),
                )
            )
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
