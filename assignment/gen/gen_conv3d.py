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
from frontends.tensor_args import Conv3dArgs
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
    # input_shape: [C_in, D, H, W], filter_shape: [C_out, C_in, Kd, Kh, Kw]
    d_in, h_in, w_in = input_shape[1], input_shape[2], input_shape[3]
    k_d, k_h, k_w = filter_shape[2], filter_shape[3], filter_shape[4]
    s = stride
    if padding == "valid":
        return [0, 0, 0, 0, 0, 0]
    if padding != "same":
        raise NotImplementedError("unknown padding: " + padding)

    if s == 1:
        o_d = (d_in + s - 1) // s
        o_h = (h_in + s - 1) // s
        o_w = (w_in + s - 1) // s
        total_pd = max(0, (o_d - 1) * s + k_d - d_in)
        total_ph = max(0, (o_h - 1) * s + k_h - h_in)
        total_pw = max(0, (o_w - 1) * s + k_w - w_in)
        pf = total_pd // 2
        pb = total_pd - pf
        pt = total_ph // 2
        pbot = total_ph - pt
        pl = total_pw // 2
        pr = total_pw - pl
        return [pf, pb, pt, pbot, pl, pr]

    # stride > 1: symmetric k//2 per side (match evaluator convention)
    p_d = k_d // 2
    p_h = k_h // 2
    p_w = k_w // 2
    return [p_d, p_d, p_h, p_h, p_w, p_w]


def gen_conv3d(term, cs_kernels, shapes):
    a_shape = shapes[0]
    b_shape = shapes[1]
    args = Conv3dArgs.from_term(term)
    spec = conv_spec([args.stride] * 3)

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

        stride = args.stride
        padding_mode = args.padding
        d_i, h_i, w_i = a_shape[1], a_shape[2], a_shape[3]
        k_d, k_h, k_w = b_shape[2], b_shape[3], b_shape[4]
        if padding_mode == "valid":
            d_o = (d_i - k_d) // stride + 1
            h_o = (h_i - k_h) // stride + 1
            w_o = (w_i - k_w) // stride + 1
            # Shape analysis still uses dense [D_out,H_out,W_out] (p2);
            # see Conv3D(valid) in ``assignment.shape_check``.
            d_o_layout, h_o_layout, w_o_layout = d_i, h_i, w_i
        else:
            if stride == 1:
                d_o, h_o, w_o = d_i, h_i, w_i
                d_o_layout, h_o_layout, w_o_layout = d_o, h_o, w_o
            else:
                p_d, p_h, p_w = k_d // 2, k_h // 2, k_w // 2
                d_o = (d_i + 2 * p_d - k_d) // stride + 1
                h_o = (h_i + 2 * p_h - k_h) // stride + 1
                w_o = (w_i + 2 * p_w - k_w) // stride + 1
                d_o_layout, h_o_layout, w_o_layout = d_o, h_o, w_o

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
