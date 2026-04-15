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
- gen_conv2d_roll: Main function for generating roll-based convolution layouts

"""

from copy import deepcopy as copy

from assignment.gen.gen_compaction import find_compaction
from frontends.tensor_args import Conv2dArgs
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.roll import Roll
from util.util import round_to_ceiling_power_of_2


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
        kernel.layout.n,
        kernel.layout.secret,
    )
    # NOTE: conv2d only requires split_rolls for alignment. This is because
    # convolutions require only parts of the left and right rolls.
    # Other values (that wrap around the dimension when rolled) are masked out.

    # NOTE: this roll will be optimized later in circuit optimizations
    return Kernel(KernelOp.SPLIT_ROLL, [roll, kernel], layout)


def calculate_padding(input_shape, filter_shape, stride, padding):
    # Input shape: [C_in, H_in, W_in]
    H_in, W_in = input_shape[1], input_shape[2]
    # Filter shape: [C_out, C_in, H_f, W_f]
    K_h, K_w = filter_shape[2], filter_shape[3]
    S_h, S_w = stride, stride

    if padding == "valid":
        return [0, 0, 0, 0]

    elif padding == "same":
        if S_h == 1:
            # Stride 1: preserve spatial size (align with tensor_evaluator._eval_conv2d).
            O_h = (H_in + S_h - 1) // S_h
            O_w = (W_in + S_w - 1) // S_w
            total_padding_h = max(0, (O_h - 1) * S_h + K_h - H_in)
            total_padding_w = max(0, (O_w - 1) * S_w + K_w - W_in)
            pad_top = total_padding_h // 2
            pad_bottom = total_padding_h - pad_top
            pad_left = total_padding_w // 2
            pad_right = total_padding_w - pad_left
            return [pad_top, pad_bottom, pad_left, pad_right]
        # Stride > 1: symmetric k//2 per side (PyTorch nn.Conv2d), matching tensor_evaluator.
        p_h = K_h // 2
        p_w = K_w // 2
        return [p_h, p_h, p_w, p_w]
    else:
        raise NotImplementedError("unknown padding: " + padding)


def add_replicated_dimensions_roll(a_shape, b_shape):
    # add replicated dimensions to the a_kernel
    # this is done to allow for the a_kernel to be replicated
    # and rolled to align with the b_kernel
    #
    # b_shape is always [C_out, C_in, H_f, W_f] in 4D form
    # We need to replicate for C_in (input channels) and spatial dims (H_f, W_f)
    # C_out (output channels) is handled separately in the output layout

    replicated_dims = {}
    c_in = b_shape[1]  # Number of input channels
    _h_f = b_shape[2]  # Filter height (unused; for documentation)
    _w_f = b_shape[3]  # Filter width (unused; for documentation)

    # Replicate for input channels (but only if we need different filters per channel)
    if c_in > 1:
        replicated_dims[1] = Dim(None, c_in, a_shape[1] * a_shape[2])

    # Replicate for spatial dimensions based on INPUT shape
    if a_shape[1] > 1:
        replicated_dims[2] = Dim(None, a_shape[1], a_shape[2])

    if a_shape[2] > 1:
        replicated_dims[3] = Dim(None, a_shape[2], 1)

    return replicated_dims


def gen_conv2d_roll(term, cs_kernels, shapes):
    # assumption is that a_kernel (input) is secret and b_kernel (weights) is public
    # the goal is to use rolls to align the a_kernel for summation
    # the b layout can then follow the same alignment as the a_kernel
    a_shape = shapes[0]
    b_shape = shapes[1]

    args = Conv2dArgs.from_term(term)
    if int(args.groups) != 1:
        raise NotImplementedError(
            "Conv2d with groups != 1 is supported for tensor_ir.eval / shape analysis only; "
            "layout generation (gen_conv2d_roll) currently requires groups=1."
        )
    # find padding
    padding = calculate_padding(a_shape, b_shape, args.stride, args.padding)
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
        replicated_dims = add_replicated_dimensions_roll(a_shape, b_shape)

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
            a_kernel = apply_replication(args.input, a_kernel, replicated_dims[3])
            a_roll = Roll(a_dim_map[2], replicated_dims[3])
            a_kernel = apply_roll(args.input, a_kernel, a_roll)

        # apply replication and roll for height
        if 2 in replicated_dims:
            a_kernel = apply_replication(args.input, a_kernel, replicated_dims[2])
            a_roll = Roll(a_dim_map[1], replicated_dims[2])
            a_kernel = apply_roll(args.input, a_kernel, a_roll)

        # apply final replication
        if 1 in replicated_dims:
            a_kernel = apply_replication(args.input, a_kernel, replicated_dims[1])

        # apply final replication
        if 0 in replicated_dims:
            a_kernel = apply_replication(args.input, a_kernel, replicated_dims[0])

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
                # Replicated dimension - consume from alignment list
                if alignment[a_dim.dim]:
                    b_dim = Dim(alignment[a_dim.dim][0], a_dim.extent, 1)
                    b_dims.append(b_dim)
                    alignment[a_dim.dim] = alignment[a_dim.dim][1:]
                else:
                    # No more alignment entries - skip this dimension
                    b_dim = Dim(None, a_dim.extent, 1)
                    b_dims.append(b_dim)
            else:
                # Data dimension - always maps to None (don't consume alignment)
                b_dim = Dim(None, a_dim.extent, 1)
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
        b_layout = Layout(args.filter, b_rolls, b_dims, a_kernel.layout.n, False)
        b_kernel = Kernel(KernelOp.TENSOR, [], b_layout)

        # find output layout after convolution
        output_dims = []

        # Add output channel dimension if C_out > 1
        c_out = b_shape[0]
        if c_out > 1:
            # Calculate stride for channel dimension (product of spatial dimensions)
            spatial_extent = 1
            for dim in a_kernel.layout.slot_dims:
                if dim.dim and dim.dim > 0:  # Spatial dimensions (1, 2)
                    spatial_extent *= dim.extent
            output_dims.append(Dim(0, c_out, spatial_extent))

        # Add remaining dimensions (spatial and any others)
        # For single output (C_out=1), just copy all dims from input
        # For multi-output (C_out>1), skip dimension 0 if it exists (input channels)
        for dim in a_kernel.layout.slot_dims:
            if dim.dim:
                # Skip input channel dimension (0) if we already added output channel dimension
                if c_out > 1 and dim.dim == 0:
                    continue
                output_dims.append(copy(dim))

        output_layout = Layout(
            term,
            [],
            output_dims,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )

        kernel = Kernel(KernelOp.CONV2D_ROLL, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)
        if not kernel.layout.rolls:
            output_kernels.add(find_compaction(kernel))
    return output_kernels


def add_replicated_dimensions(a_shape, b_shape):
    # add replicated dimensions to the a_kernel
    # this is done to allow for the a_kernel to be replicated
    # and rolled to align with the b_kernel
    #
    # b_shape is always [C_out, C_in, H_f, W_f] in 4D form
    # We need to match the dimensions for C_in (input channels),
    # then replicate for spatial dimensions (H_f, W_f), and C_out (output channels)

    replicated_dims = []
    b_dims = []
    c_out = b_shape[0]  # Number of output channels
    c_in = b_shape[1]  # Number of input channels
    h_f = b_shape[2]  # Filter height
    w_f = b_shape[3]  # Filter width

    # Replicate output channels
    if c_out > 1:
        replicated_dims.append(Dim(None, c_out, c_in * a_shape[1] * a_shape[2]))
        b_dims.append(Dim(0, c_out, 1))

    # Replicate spatial dimensions
    if h_f > 1:
        replicated_dims.append(Dim(None, h_f, a_shape[2]))
        b_dims.append(Dim(2, h_f, 1))

    if w_f > 1:
        replicated_dims.append(Dim(None, w_f, 1))
        b_dims.append(Dim(3, w_f, 1))

    # Return the replicated dimensions and the b_dims
    return replicated_dims, b_dims


def _conv2d_output_dims_zip(a_dims, b_dims, h_o_p2, w_o_p2, stride):
    """Build CONV2D output layout dims from replicated input and public filter layouts."""
    output_dims = []
    spatial_out_done = {1: False, 2: False}

    for a_dim, b_dim in zip(copy(a_dims), copy(b_dims)):
        if a_dim.dim_type == DimType.EMPTY:
            output_dims.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim in [0] or b_dim.dim in [1, 2, 3]:
            output_dims.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim is None and b_dim.dim == 0:
            output_dims.append(b_dim)
        elif (
            a_dim.dim in (1, 2) and a_dim.dim_type == DimType.FILL and b_dim.dim is None
        ):
            d = a_dim.dim
            if spatial_out_done[d]:
                return None
            spatial_out_done[d] = True
            output_dims.append(Dim(d, h_o_p2 if d == 1 else w_o_p2, 1))
            if stride == 2:
                output_dims.append(Dim(None, stride, 1, DimType.EMPTY))
            elif stride != 1:
                raise NotImplementedError("stride not supported: " + str(stride))
        else:
            output_dims.append(Dim(a_dim.dim, a_dim.extent, 1, DimType.FILL))

    return output_dims


def _materialized_tensor_shape(output_layout):
    """Product of extents per logical dim index (0,1,2), ignoring EMPTY / None dim."""
    shape_map = {}
    for d in output_layout.get_dims():
        if d.dim_type == DimType.EMPTY or d.dim is None:
            continue
        if d.dim in shape_map:
            shape_map[d.dim] *= d.extent
        else:
            shape_map[d.dim] = d.extent
    if not shape_map:
        return []
    return [shape_map.get(i, 1) for i in range(max(shape_map.keys()) + 1)]


def gen_conv2d(term, cs_kernels, shapes):
    # assumption is that a_kernel (input) is secret and b_kernel (weights) is public
    #
    # Implementation goal:
    # - gen_conv2d should analyze the input layout and determine the output layout
    # - lower_conv2d should convert the convolution to a matrix-vector multiplication
    # - lower_conv2d should determine the packing for the convolution, since the weights are public
    #
    # Input layout:
    # - Input channel, dimension 1 (height), dimension 2 (width)
    # Filter layout:
    # - Output channel, kernel dimension 1, kernel dimension 2
    # Stride:
    # - Stride is either 1 or 2
    # Padding:
    # - Padding is either valid or same
    # Output layout:
    # - Output channel, dimension 0 (height), dimension 1 (width) (depending on stride and padding)

    # The question is given any generalized input layout, how to figure out the summation dimensions and output layout?
    # Step 1: Replicate the input layout to find dimension alignment
    # Step 2: Identify the summation dimensions:
    # - This should be: the input channel, kernel dimension 0, kernel dimension 1
    # Step 3: Identify the output dimensions:
    # - This should be: the output channel, dimension 0 (height), dimension 1 (width) (depending on stride and padding)
    # Step 4: Create the layout and kernel

    a_shape = shapes[0]
    b_shape = shapes[1]

    args = Conv2dArgs.from_term(term)
    if int(args.groups) != 1:
        raise NotImplementedError(
            "Conv2d with groups != 1 is supported for tensor_ir.eval / shape analysis only; "
            "layout generation (gen_conv2d) currently requires groups=1."
        )
    # find padding
    padding = calculate_padding(a_shape, b_shape, args.stride, args.padding)
    term.cs.append(padding)

    b_term = args.filter
    b_term.cs.append(padding)

    output_kernels = set()
    for a_kernel in cs_kernels:
        # assumes that a layout does not have any rolls applied to it
        if a_kernel.layout.rolls:
            continue

        # Add replication dimensions to the a_kernel
        replicated_dims, b_dims = add_replicated_dimensions(a_shape, b_shape)
        for dim in replicated_dims[::-1]:
            a_kernel = apply_replication(args.input, a_kernel, dim)

        # Since b is public, we can create a cs_kernel for b
        # and add metada information to help with packing the weights
        for dim in a_kernel.layout.get_dims():
            if dim.dim == 0:
                b_dims.append(Dim(1, dim.extent, dim.stride))
            elif dim.dim is not None:
                b_dims.append(Dim(None, dim.extent, 1))
            elif dim.dim_type == DimType.EMPTY:
                b_dims.append(Dim(None, dim.extent, 1, DimType.EMPTY))

        assert len(b_dims) == len(a_kernel.layout.get_dims())

        b_layout = Layout(b_term, [], b_dims, a_kernel.layout.n, False)
        b_kernel = Kernel(KernelOp.PUNCTURED_TENSOR, [], b_layout)

        # Output spatial size (depends on stride and padding)
        stride = args.stride
        padding = args.padding
        h_i, w_i = a_shape[1], a_shape[2]
        f_h, f_w = b_shape[2], b_shape[3]
        if padding == "valid":
            h_o = (h_i - f_h) // stride + 1
            w_o = (w_i - f_w) // stride + 1
        else:
            if stride == 1:
                h_o, w_o = h_i, w_i
            else:
                p_h, p_w = f_h // 2, f_w // 2
                h_o = (h_i + 2 * p_h - f_h) // stride + 1
                w_o = (w_i + 2 * p_w - f_w) // stride + 1
        h_o_p2 = round_to_ceiling_power_of_2(h_o)
        w_o_p2 = round_to_ceiling_power_of_2(w_o)

        # Output layout:
        output_dims = _conv2d_output_dims_zip(
            a_kernel.layout.get_dims(), b_dims, h_o_p2, w_o_p2, stride
        )
        if output_dims is None:
            continue
        output_layout = Layout(
            term, [], output_dims, a_kernel.layout.n, a_kernel.layout.secret
        )

        expected = [b_shape[0], h_o_p2, w_o_p2]
        kernel_shape = _materialized_tensor_shape(output_layout)
        if kernel_shape != expected:
            continue

        kernel = Kernel(KernelOp.CONV2D, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)
        if not kernel.layout.rolls:
            output_kernels.add(find_compaction(kernel))
    return output_kernels
