"""
Convolution-related cost modeling helpers.
"""

import math

from ir.dim import DimType
from util.util import prod


class ConvCostMixin:
    def conv2d_roll_ops(self, ops):
        """Calculate 2D convolution operations."""
        ct_ops = []
        ct_adds = []
        for dim in self.kernel.cs[1].layout.ct_dims:
            if dim.dim == 0 or dim.dim == 1:
                ct_adds.append(dim.extent)
            ct_ops.append(dim.extent)

        num_muls = 0
        num_adds = 0
        if ct_ops:
            num_muls += prod(ct_ops)
        if ct_adds:
            num_adds += prod(ct_adds)

        num_ct_real = self.kernel.layout.num_ct()

        rot_and_sums = []
        for slot_dim in self.kernel.layout.slot_dims:
            if slot_dim.dim_type == DimType.EMPTY:
                rot_and_sums.append(slot_dim.extent)
        rot_and_sum = math.ceil(math.log2(prod(rot_and_sums)))
        num_adds += num_ct_real * rot_and_sum
        num_rots = num_ct_real * rot_and_sum

        ops["add"] += num_adds
        ops["rot"] += num_rots
        ops["mul"] += num_muls
        return ops

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
