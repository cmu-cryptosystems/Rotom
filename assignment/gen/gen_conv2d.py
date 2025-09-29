"""
2D Convolution layout generation utilities.

This module provides functions for generating optimal layouts for 2D convolution
operations in FHE computations. Convolution operations require special handling
for padding, stride, and spatial dimensions to ensure correct computation
in the homomorphic encryption domain.

Key functions:
- apply_replication: Applies replication to convolution kernels
- apply_roll: Applies roll operations for convolution alignment
- calculate_padding: Calculates padding requirements for convolution
- gen_conv2d: Main function for generating convolution layouts
"""

from copy import deepcopy as copy

from ir.analysis.shape import Shape
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.roll import Roll


def apply_replication(term, kernel, dim):
    """Applies replication to a convolution kernel.

    This function creates a replicated version of a kernel by adding
    a replication dimension. Replication is used in convolution operations
    to handle spatial dimensions and padding requirements.

    Args:
        term: TensorTerm representing the convolution operation
        kernel: Kernel to apply replication to
        dim: Dimension to replicate along

    Returns:
        Kernel: New kernel with replication applied
    """
    dims = copy(kernel.layout.get_dims())
    replicated_dims = [dim] + dims
    replicated_layout = Layout(
        term,
        kernel.layout.rolls,
        replicated_dims,
        kernel.layout.offset,
        kernel.layout.n,
        kernel.layout.secret,
    )
    return Kernel(KernelOp.REPLICATE, [kernel], replicated_layout)


def apply_roll(term, kernel, roll):
    """Applies a roll operation to a convolution kernel.

    This function creates a rolled version of a kernel by adding a roll
    operation. For convolution operations, split rolls are used because
    convolutions only require parts of the rolled values, with other
    values being masked out.

    Args:
        term: TensorTerm representing the convolution operation
        kernel: Kernel to apply roll to
        roll: Roll operation to apply

    Returns:
        Kernel: New kernel with roll operation applied
    """
    rolled_dims = kernel.layout.get_dims()
    rolled_rolls = copy(kernel.layout.rolls)
    rolled_rolls.append(roll)

    # rolled layout
    layout = Layout(
        term,
        rolled_rolls,
        rolled_dims,
        kernel.layout.offset,
        kernel.layout.n,
        kernel.layout.secret,
    )
    # NOTE: conv2d only requires split_rolls for alignment. This is because
    # convolutions require only parts of the left and right rolls.
    # Other values (that wrap around the dimension when rolled) are masked out.

    # NOTE: this roll will be optimized later in circuit optimizations
    return Kernel(KernelOp.SPLIT_ROLL, [roll, kernel], layout)


def calculate_padding(input_shape, filter_shape, stride, padding):
    H_in, W_in = input_shape[1], input_shape[2]
    K_h, K_w = filter_shape[1], filter_shape[2]
    S_h, S_w = stride, stride

    if padding == "valid":
        return [0, 0, 0, 0]

    elif padding == "same":
        # output dimensions
        O_h = (H_in + S_h - 1) // S_h
        O_w = (W_in + S_w - 1) // S_w

        # total padding
        total_padding_h = max(0, (O_h - 1) * S_h + K_h - H_in)
        total_padding_w = max(0, (O_w - 1) * S_w + K_w - W_in)

        # padding on each side
        pad_top = total_padding_h // 2
        pad_bottom = total_padding_h - pad_top
        pad_left = total_padding_w // 2
        pad_right = total_padding_w - pad_left
        return [pad_top, pad_bottom, pad_left, pad_right]
    else:
        raise NotImplementedError("unknown padding: " + padding)


def add_replicated_dimensions(a_shape, b_shape):
    # add replicated dimensions to the a_kernel
    # this is done to allow for the a_kernel to be replicated
    # and rolled to align with the b_kernel
    replicated_dims = {}
    if b_shape[0] > 1:
        replicated_dims[0] = Dim(None, b_shape[0], b_shape[1] * a_shape[1] * a_shape[2])
    if b_shape[1] > 1:
        replicated_dims[1] = Dim(None, b_shape[1], a_shape[1] * a_shape[2])
    if a_shape[1] > 1:
        replicated_dims[2] = Dim(None, a_shape[1], a_shape[2])
    if a_shape[2] > 1:
        replicated_dims[3] = Dim(None, a_shape[2], 1)
    return replicated_dims


def gen_conv2d(term, cs_kernels, shapes):
    # assumption is that a_kernel (input) is secret and b_kernel (weights) is public
    # the goal is to use rolls to align the a_kernel for summation
    # the b layout can then follow the same alignment as the a_kernel
    a_shape = shapes[0]
    b_shape = shapes[1]

    # find padding
    padding = calculate_padding(a_shape, b_shape, term.cs[2], term.cs[3])
    term.cs.append(padding)

    output_kernels = set()
    for a_kernel in cs_kernels:
        # assumes that a layout does not have any rolls applied to it
        if a_kernel.layout.rolls:
            continue

        # assumes input is row-major layout (though this can be relaxed later)
        dims_ = [dim.dim for dim in a_kernel.layout.get_dims() if dim.dim is not None]
        if sorted(dims_) != dims_:
            continue

        # add replicated dimensions to the a_kernel
        replicated_dims = add_replicated_dimensions(a_shape, b_shape)

        # map a_dims
        a_dim_map = {}
        for dim in a_kernel.layout.get_dims():
            # assumes that dimensions are not tiled
            assert dim.dim not in a_dim_map
            a_dim_map[dim.dim] = dim

        # map b_dims
        b_dims = []
        b_rolls = []
        b_dim_stride = 1
        for dim in a_kernel.layout.get_dims()[::-1]:
            b_dims.insert(0, Dim(None, dim.extent, b_dim_stride))
            b_dim_stride *= dim.extent

        # apply replication and roll for width
        if 3 in replicated_dims:
            a_kernel = apply_replication(term.cs[0], a_kernel, replicated_dims[3])
            a_roll = Roll(a_dim_map[2], replicated_dims[3])
            a_kernel = apply_roll(term.cs[0], a_kernel, a_roll)

        # apply replication and roll for height
        if 2 in replicated_dims:
            a_kernel = apply_replication(term.cs[0], a_kernel, replicated_dims[2])
            a_roll = Roll(a_dim_map[1], replicated_dims[2])
            a_kernel = apply_roll(term.cs[0], a_kernel, a_roll)

        # apply final replication
        if 1 in replicated_dims:
            a_kernel = apply_replication(term.cs[0], a_kernel, replicated_dims[1])

        # apply final replication
        if 0 in replicated_dims:
            a_kernel = apply_replication(term.cs[0], a_kernel, replicated_dims[0])

        # match b_dims to a_dims
        # ideally this should match based on the length of the dimension traversal
        b_dims = []
        alignment = {
            None: [2, 3],
            0: [None],
            1: [None],
            2: [None],
        }
        for a_dim in a_kernel.layout.get_dims():
            if a_dim.dim is None and a_dim.dim_type == DimType.EMPTY:
                b_dim = copy(a_dim)
                b_dims.append(b_dim)
            elif a_dim.dim is None:
                b_dim = Dim(alignment[a_dim.dim][0], a_dim.extent, 1)
                b_dims.append(b_dim)
                alignment[a_dim.dim] = alignment[a_dim.dim][1:]
            else:
                b_dim = Dim(alignment[a_dim.dim][0], a_dim.extent, 1)
                b_dims.append(b_dim)

        # back-fill, fix b_dim replication stride
        none_stride = 1
        for b_dim in b_dims[::-1]:
            if b_dim.dim is None and b_dim.dim_type == DimType.FILL:
                b_dim.stride = none_stride
                none_stride *= b_dim.extent

        b_rolls = []
        for a_roll in a_kernel.layout.rolls:
            a_roll_idx = a_roll.roll_index(a_kernel.layout.get_dims())
            b_rolls.append(Roll(b_dims[a_roll_idx[0]], b_dims[a_roll_idx[1]]))

        # create layout and kernel
        b_layout = Layout(term.cs[1], b_rolls, b_dims, {}, a_kernel.layout.n, False)
        b_kernel = Kernel(KernelOp.TENSOR, [], b_layout)

        # find output layout after convolution
        output_dims = []
        for dim in a_kernel.layout.slot_dims:
            if dim.dim:
                output_dims.append(copy(dim))
        output_layout = Layout(
            term,
            [],
            output_dims,
            a_kernel.layout.offset,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )

        kernel = Kernel(KernelOp.CONV2D, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)
    return output_kernels
