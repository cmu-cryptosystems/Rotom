"""
3D convolution layout generation.

A 3D convolution is the direct kernel of ``gen_conv2d`` over three windowed
axes:
- input:  [C_in, D_in, H_in, W_in]   (secret)
- filter: [C_out, C_in, K_d, K_h, K_w] (public)
- output: [C_out, D_out, H_out, W_out]
"""

from assignment.gen.gen_conv2d import conv_window, window_kernels
from frontends.tensor_args import Conv3dArgs


def gen_conv3d(term, cs_kernels, shapes):
    """Generates the CONV_DIRECT kernels of a CONV3D term and their compactions."""
    window = conv_window(Conv3dArgs.from_term(term), shapes[0], shapes[1])
    return window_kernels(term, window, cs_kernels)
