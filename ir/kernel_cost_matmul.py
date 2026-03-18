"""
Matrix-multiplication-related cost modeling helpers.

Extracted from `kernel_cost.py` to keep the central cost model organized
by operator family.
"""

import math

from ir.dim import DimType
from util.util import prod


class MatmulCostMixin:
    def matmul_ops(self, ops, layout, sum_dim):
        """Calculate the operations to perform a matrix operation."""
        ops["mul"] += layout.num_ct()

        ct_sum_dims = []
        remaining_ct_extents = []
        for dim in layout.ct_dims:
            if dim.dim == sum_dim:
                ct_sum_dims.append(dim.extent)
            else:
                remaining_ct_extents.append(dim.extent)
        remaining_cts = int(prod(remaining_ct_extents))
        if ct_sum_dims:
            ops["add"] += int(prod(ct_sum_dims))

        slot_sum_dims = []
        for dim in layout.slot_dims:
            if dim.dim == sum_dim:
                slot_sum_dims.append(dim.extent)
        rot_sum_op_count = remaining_cts * math.ceil(
            math.log2(int(prod(slot_sum_dims)))
        )
        ops["add"] += rot_sum_op_count
        ops["rot"] += rot_sum_op_count

        if any([dim.dim_type == DimType.EMPTY for dim in self.kernel.layout.slot_dims]):
            ops["mul"] += remaining_cts
        return ops
