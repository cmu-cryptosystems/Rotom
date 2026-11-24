"""
2D Convolution layout generation utilities.

A convolution is generated as a direct kernel: rotations of the input, each
multiplied by a plaintext of filter elements (``util.window``). Its output
layout follows the input layout, with the output channel on ciphertexts or in
the positions the input leaves free.

Key functions:
- apply_replication: Applies replication to a kernel (used by pooling)
- calculate_padding: Calculates padding requirements for convolution
- direct_kernels: The direct kernel of one input layout, and its compaction
- gen_conv2d: Main function for generating convolution layouts
"""

from copy import deepcopy as copy

from assignment.alignment import conv_spec
from assignment.gen.gen_compaction import find_compaction
from assignment.gen.gen_window import (
    FILLS,
    direct_dims,
)
from frontends.tensor_args import Conv2dArgs, axis_strides, window_axis
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from util.window import Window, window_keys


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


def direct_kernels(term, window, a_kernel):
    """The direct kernel of one input layout, and its compaction.

    The output channel can go on ciphertexts, in the positions a stride frees,
    or in every gap (``FILLS``). The one kept is the cheapest to compute and
    compact, since each placement gives the operators downstream another
    layout to consider.
    """
    best = None
    a_layout = a_kernel.layout
    for fill in FILLS:
        dims = direct_dims(window, a_layout, fill)
        if dims is None:
            continue
        z_dims, out_dims = dims
        z_layout = Layout(term, [], z_dims, a_layout.n, a_layout.secret)
        output_layout = Layout(term, [], out_dims, a_layout.n, a_layout.secret)
        try:
            window_keys(window, a_layout, z_layout)
        except ValueError:
            # a layout that does not hold every element the window reads
            continue
        kernel = Kernel(
            KernelOp.CONV_DIRECT, [a_kernel, z_layout, window], output_layout
        )
        compacted = find_compaction(kernel)
        cost = KernelCost(compacted, "lan").total_cost()
        if best is None or cost < best[0]:
            best = (cost, {kernel, compacted})
    return best[1] if best else set()


def gen_conv2d(term, cs_kernels, shapes):
    """Generates the kernels of a 2D convolution, one input layout at a time.

    The input ``[C_in, H, W]`` is secret and the filter ``[C_out, C_in, K_h,
    K_w]`` is public, so the filter takes whatever packing the input and
    output layouts ask for (see ``util.window``).

    Args:
        term: the CONV2D term
        cs_kernels: the input's candidate kernels
        shapes: the unpadded shapes of the input and the filter

    Returns:
        Set of CONV_DIRECT kernels and their compactions
    """
    a_shape = shapes[0]
    b_shape = shapes[1]

    args = Conv2dArgs.from_term(term)
    spec = conv_spec(args.strides)
    # find padding
    padding = calculate_padding(a_shape, b_shape, args.stride, args.padding)

    output_kernels = set()
    window = Window(
        spec,
        tuple(a_shape),
        tuple(b_shape),
        (b_shape[0],)
        + tuple(
            window_axis(a_shape[axis + 1], b_shape[axis + 2], s, args.padding)[0]
            for axis, s in enumerate(args.strides)
        ),
        tuple(padding[::2]),
    )
    for a_kernel in cs_kernels:
        # assumes that a layout does not have any rolls applied to it
        if a_kernel.layout.rolls:
            continue
        # The input is a placeholder, so the search adds the input's own cost.
        a_kernel = Kernel(KernelOp.CS, [0], a_kernel.layout)
        output_kernels |= direct_kernels(term, window, a_kernel)
    return output_kernels
