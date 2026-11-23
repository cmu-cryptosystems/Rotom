"""
Convolution-related cost modeling helpers.
"""

import math

from util.util import prod


class ConvCostMixin:
    def conv2d_ops(self, ops):
        """Calculate convolution operations."""
        input_dims = self.kernel.cs[0].layout.get_dims()
        output_dims = self.kernel.layout.get_dims()

        input_dim_len = int(prod([input_dim.extent for input_dim in input_dims]))
        output_dim_len = int(prod([output_dim.extent for output_dim in output_dims]))

        ops["mul"] += output_dim_len
        ops["add"] += output_dim_len * int(math.log2(input_dim_len))
        ops["rot"] += int(math.sqrt(output_dim_len * int(math.log2(input_dim_len))))
        return ops
