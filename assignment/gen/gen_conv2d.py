"""
2D Convolution layout generation utilities.

A convolution is generated as a direct kernel: rotations of the input, each
multiplied by a plaintext of filter elements (``util.window``). Its output
layout follows the input layout, with the output channel on ciphertexts or in
the positions the input leaves free.

Key functions:
- apply_replication: Applies replication to a kernel (used by pooling)
- conv_window: The window of a CONV2D term
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
from frontends.tensor_args import Conv2dArgs
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from util.window import Window, window_keys, window_mismatches


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


def conv_window(args, a_shape, b_shape):
    """The window of a convolution over unpadded operand shapes.

    A depthwise term carries TFLite's ``[1, *taps, C]`` filter, which the
    window reads as the spec's ``[*taps, C]``.

    Args:
        args: the term's ``Conv2dArgs`` or ``Conv3dArgs``
    """
    taps = args.taps(b_shape)
    channels = args.out_channels(b_shape)
    if args.depthwise:
        if channels != a_shape[0]:
            raise NotImplementedError(
                f"depthwise channel multiplier: {a_shape[0]} -> {channels} channels"
            )
        filter_shape = taps + (channels,)
    else:
        filter_shape = tuple(b_shape)
    axes = args.window(a_shape, b_shape)
    return Window(
        conv_spec(args.strides, depthwise=args.depthwise),
        tuple(a_shape),
        filter_shape,
        (channels,) + tuple(out for out, _, _ in axes),
        tuple(before for _, before, _ in axes),
    )


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
        # Z has to sit on the input, so that only the taps rotate.
        if window_mismatches(window, a_layout, z_layout):
            continue
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
    output layouts ask for (see ``util.window``). A depthwise convolution is
    the same kernel under its own spec: the channel is carried, so nothing is
    summed and no output channel has to be placed.

    Args:
        term: the CONV2D term
        cs_kernels: the input's candidate kernels
        shapes: the unpadded shapes of the input and the filter

    Returns:
        Set of CONV_DIRECT kernels and their compactions
    """
    window = conv_window(Conv2dArgs.from_term(term), shapes[0], shapes[1])
    return window_kernels(term, window, cs_kernels)


def window_kernels(term, window, cs_kernels):
    """The direct kernels of a windowed term, one input layout at a time."""
    output_kernels = set()
    for a_kernel in cs_kernels:
        # assumes that a layout does not have any rolls applied to it
        if a_kernel.layout.rolls:
            continue
        # The input is a placeholder, so the search adds the input's own cost.
        a_kernel = Kernel(KernelOp.CS, [0], a_kernel.layout)
        output_kernels |= direct_kernels(term, window, a_kernel)
    return output_kernels
