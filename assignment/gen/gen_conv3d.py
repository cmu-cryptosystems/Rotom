"""
3D Convolution layout generation utilities (layout assignment only).

This mirrors `gen_conv2d` but for 3D convolutions with shapes:
- input:  [C_in, D_in, H_in, W_in]   (secret)
- filter: [C_out, C_in, Kd, Kh, Kw] (public)
- output: [C_out, D_out, H_out, W_out]
"""

from copy import deepcopy as copy

from assignment.alignment import conv_spec
from assignment.gen.gen_window import (
    filter_dims,
    materialized_shape,
    output_dims,
    replication_dims,
)
from frontends.tensor_args import Conv3dArgs, axis_strides, window_axis
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from util.util import round_to_ceiling_power_of_2


def apply_replication(term, kernel, dim):
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


def calculate_padding_3d(input_shape, filter_shape, stride, padding):
    """``[pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]`` for
    input ``[C_in, D, H, W]`` and filter ``[C_out, C_in, K_d, K_h, K_w]``."""
    pads = []
    for axis, s in enumerate(axis_strides(stride, 3)):
        _, before, after = window_axis(
            input_shape[axis + 1], filter_shape[axis + 2], s, padding
        )
        pads += [before, after]
    return pads


def gen_conv3d(term, cs_kernels, shapes):
    a_shape = shapes[0]
    b_shape = shapes[1]
    args = Conv3dArgs.from_term(term)
    spec = conv_spec(args.strides)

    padding = calculate_padding_3d(a_shape, b_shape, args.stride, args.padding)
    term.cs.append(padding)
    b_term = args.filter
    b_term.cs.append(padding)

    output_kernels = set()
    for a_kernel in cs_kernels:
        if a_kernel.layout.rolls:
            continue

        replicated_dims, b_dims = replication_dims(spec, a_shape, b_shape)
        for dim in replicated_dims[::-1]:
            a_kernel = apply_replication(args.input, a_kernel, dim)

        # Build b layout to align with a layout geometry.
        b_dims = filter_dims(spec, a_kernel.layout.get_dims(), b_dims)
        if b_dims is None:
            continue

        b_layout = Layout(b_term, [], b_dims, a_kernel.layout.n, False)
        b_kernel = Kernel(KernelOp.PUNCTURED_TENSOR, [], b_layout)

        if args.padding == "valid":
            # Shape analysis still uses dense [D_out,H_out,W_out] (p2);
            # see Conv3D(valid) in ``assignment.shape_check``.
            d_o_layout, h_o_layout, w_o_layout = a_shape[1:4]
        else:
            d_o_layout, h_o_layout, w_o_layout = (
                window_axis(a_shape[axis + 1], b_shape[axis + 2], s, args.padding)[0]
                for axis, s in enumerate(args.strides)
            )

        d_o_p2 = round_to_ceiling_power_of_2(d_o_layout)
        h_o_p2 = round_to_ceiling_power_of_2(h_o_layout)
        w_o_p2 = round_to_ceiling_power_of_2(w_o_layout)

        dims = output_dims(
            spec,
            a_kernel.layout.get_dims(),
            b_dims,
            {"d_out": d_o_p2, "h_out": h_o_p2, "w_out": w_o_p2},
        )
        if dims is None:
            continue

        output_layout = Layout(
            term, [], dims, a_kernel.layout.n, a_kernel.layout.secret
        )
        expected = [b_shape[0], d_o_p2, h_o_p2, w_o_p2]
        if materialized_shape(output_layout) != expected:
            continue

        kernel = Kernel(KernelOp.CONV3D, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)

    return output_kernels
