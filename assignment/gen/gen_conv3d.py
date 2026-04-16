"""
3D Convolution layout generation utilities (layout assignment only).

This mirrors `gen_conv2d` but for 3D convolutions with shapes:
- input:  [C_in, D_in, H_in, W_in]   (secret)
- filter: [C_out, C_in, Kd, Kh, Kw] (public)
- output: [C_out, D_out, H_out, W_out]
"""

from copy import deepcopy as copy

from frontends.tensor_args import Conv3dArgs
from ir.dim import Dim, DimType
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


def add_replicated_dimensions_3d(a_shape, b_shape):
    # Replicate output channels and filter spatial dims onto the secret input layout.
    replicated_dims = []
    b_dims = []
    c_out = b_shape[0]
    c_in = b_shape[1]
    k_d, k_h, k_w = b_shape[2], b_shape[3], b_shape[4]

    # Replicate output channels
    if c_out > 1:
        replicated_dims.append(
            Dim(None, c_out, c_in * a_shape[1] * a_shape[2] * a_shape[3])
        )
        b_dims.append(Dim(0, c_out, 1))

    # Replicate spatial kernel dims (based on INPUT spatial strides)
    if k_d > 1:
        replicated_dims.append(Dim(None, k_d, a_shape[2] * a_shape[3]))
        b_dims.append(Dim(2, k_d, 1))
    if k_h > 1:
        replicated_dims.append(Dim(None, k_h, a_shape[3]))
        b_dims.append(Dim(3, k_h, 1))
    if k_w > 1:
        replicated_dims.append(Dim(None, k_w, 1))
        b_dims.append(Dim(4, k_w, 1))

    return replicated_dims, b_dims


def _materialized_tensor_shape(output_layout):
    shape_map = {}
    for d in output_layout.get_dims():
        if d.dim_type == DimType.EMPTY or d.dim is None:
            continue
        shape_map[d.dim] = shape_map.get(d.dim, 1) * d.extent
    if not shape_map:
        return []
    return [shape_map.get(i, 1) for i in range(max(shape_map.keys()) + 1)]


def _pad_output_channel_dim0_to_p2_3d(output_dims, logical_c_out):
    """Same as conv2d: sole FILL dim 0 extent widened to ceil_p2(logical C_out)."""
    logical_c_out = int(logical_c_out)
    if logical_c_out <= 1:
        return output_dims
    padded = round_to_ceiling_power_of_2(logical_c_out)
    if padded == logical_c_out:
        return output_dims
    dim0_fills = [
        (i, d)
        for i, d in enumerate(output_dims)
        if d.dim == 0 and d.dim_type == DimType.FILL
    ]
    if len(dim0_fills) != 1:
        return output_dims
    i, d = dim0_fills[0]
    if int(d.extent) != logical_c_out:
        return output_dims
    out = list(output_dims)
    out[i] = Dim(0, padded, d.stride, d.dim_type)
    return out


def _conv3d_output_dims_zip(a_dims, b_dims, d_o_p2, h_o_p2, w_o_p2, stride):
    output_dims = []
    spatial_out_done = {1: False, 2: False, 3: False}
    for a_dim, b_dim in zip(copy(a_dims), copy(b_dims)):
        if a_dim.dim_type == DimType.EMPTY:
            output_dims.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim in [0] or b_dim.dim in [1, 2, 3, 4]:
            # Sum dims / filter dims do not appear in output.
            output_dims.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim is None and b_dim.dim == 0:
            # output channel
            output_dims.append(b_dim)
        elif (
            a_dim.dim in (1, 2, 3)
            and a_dim.dim_type == DimType.FILL
            and b_dim.dim is None
        ):
            d = a_dim.dim
            if spatial_out_done[d]:
                return None
            spatial_out_done[d] = True
            extent = d_o_p2 if d == 1 else (h_o_p2 if d == 2 else w_o_p2)
            output_dims.append(Dim(d, extent, 1))
            if stride == 2:
                output_dims.append(Dim(None, stride, 1, DimType.EMPTY))
            elif stride != 1:
                raise NotImplementedError("stride not supported: " + str(stride))
        else:
            output_dims.append(Dim(a_dim.dim, a_dim.extent, 1, DimType.FILL))
    return output_dims


def gen_conv3d(term, cs_kernels, shapes):
    a_shape = shapes[0]
    b_shape = shapes[1]
    args = Conv3dArgs.from_term(term)

    padding = calculate_padding_3d(a_shape, b_shape, args.stride, args.padding)
    term.cs.append(padding)
    b_term = args.filter
    b_term.cs.append(padding)

    output_kernels = set()
    for a_kernel in cs_kernels:
        if a_kernel.layout.rolls:
            continue

        replicated_dims, b_dims = add_replicated_dimensions_3d(a_shape, b_shape)
        for dim in replicated_dims[::-1]:
            a_kernel = apply_replication(args.input, a_kernel, dim)

        # Build b layout to align with a layout geometry.
        for dim in a_kernel.layout.get_dims():
            if dim.dim == 0:
                b_dims.append(Dim(1, dim.extent, dim.stride))
            elif dim.dim is not None:
                b_dims.append(Dim(None, dim.extent, 1))
            elif dim.dim_type == DimType.EMPTY:
                b_dims.append(Dim(None, dim.extent, 1, DimType.EMPTY))

        if len(b_dims) != len(a_kernel.layout.get_dims()):
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

        output_dims = _conv3d_output_dims_zip(
            a_kernel.layout.get_dims(), b_dims, d_o_p2, h_o_p2, w_o_p2, stride
        )
        if output_dims is None:
            continue

        output_dims = _pad_output_channel_dim0_to_p2_3d(output_dims, b_shape[0])
        output_layout = Layout(
            term, [], output_dims, a_kernel.layout.n, a_kernel.layout.secret
        )
        c_out_p2 = round_to_ceiling_power_of_2(int(b_shape[0]))
        expected = [c_out_p2, d_o_p2, h_o_p2, w_o_p2]
        if _materialized_tensor_shape(output_layout) != expected:
            continue

        kernel = Kernel(KernelOp.CONV3D, [a_kernel, b_kernel], output_layout)
        output_kernels.add(kernel)

    return output_kernels
