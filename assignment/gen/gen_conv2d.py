"""
2D Convolution layout generation utilities.

This module provides functions for generating optimal layouts for 2D convolution
operations in FHE computations. Convolution operations require special handling
for padding, stride, and spatial dimensions to ensure correct computation
in the homomorphic encryption domain.

Key functions:
- apply_replication: Applies replication to convolution kernels
- calculate_padding: Calculates padding requirements for convolution
- gen_conv2d: Main function for generating convolution layouts

"""

from copy import deepcopy as copy

from assignment.alignment import conv_spec
from assignment.gen.gen_compaction import find_compaction
from assignment.gen.gen_window import (
    filter_dims,
    materialized_shape,
    output_dims,
    replication_dims,
)
from frontends.tensor_args import Conv2dArgs, axis_strides, window_axis
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
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


def calculate_padding(input_shape, filter_shape, stride, padding):
    """``[pad_top, pad_bottom, pad_left, pad_right]`` for input ``[C_in, H, W]``
    and filter ``[C_out, C_in, K_h, K_w]``."""
    pads = []
    for axis, s in enumerate(axis_strides(stride, 2)):
        _, before, after = window_axis(
            input_shape[axis + 1], filter_shape[axis + 2], s, padding
        )
        pads += [before, after]
    return pads


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
    spec = conv_spec(args.strides)
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

        # Replicate the input over the filter labels it does not hold
        replicated_dims, b_dims = replication_dims(spec, a_shape, b_shape)
        for dim in replicated_dims[::-1]:
            a_kernel = apply_replication(args.input, a_kernel, dim)

        # Since b is public, its layout is the one that faces the input's
        b_dims = filter_dims(spec, a_kernel.layout.get_dims(), b_dims)
        if b_dims is None:
            continue

        b_layout = Layout(b_term, [], b_dims, a_kernel.layout.n, False)
        b_kernel = Kernel(KernelOp.PUNCTURED_TENSOR, [], b_layout)

        # Output spatial size (depends on stride and padding)
        h_o, w_o = (
            window_axis(a_shape[axis + 1], b_shape[axis + 2], s, args.padding)[0]
            for axis, s in enumerate(args.strides)
        )
        h_o_p2 = round_to_ceiling_power_of_2(h_o)
        w_o_p2 = round_to_ceiling_power_of_2(w_o)

        # Output layout:
        dims = output_dims(
            spec,
            a_kernel.layout.get_dims(),
            b_dims,
            {"h_out": h_o_p2, "w_out": w_o_p2},
        )
        if dims is None:
            continue
        output_layout = Layout(
            term, [], dims, a_kernel.layout.n, a_kernel.layout.secret
        )

        expected = [b_shape[0], h_o_p2, w_o_p2]
        if materialized_shape(output_layout) != expected:
            continue

        kernel = Kernel(KernelOp.CONV2D, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)
        if not kernel.layout.rolls:
            output_kernels.add(find_compaction(kernel))
    return output_kernels
