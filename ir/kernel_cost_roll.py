"""
Roll/shift/compaction-related cost modeling helpers.
"""

import math

from util.util import prod


class RollCostMixin:
    def roll_ops(self, ops):
        """Calculate the ops to perform a roll operation."""
        roll = self.kernel.cs[0]
        roll_extent = roll.dim_to_roll.extent

        num_ops = 2 * roll_extent

        if (
            roll.dim_to_roll in self.kernel.layout.ct_dims
            and roll.dim_to_roll_by in self.kernel.layout.ct_dims
        ):
            return ops

        ct_dims = []
        for ct_dim in self.kernel.layout.ct_dims:
            if ct_dim != roll.dim_to_roll:
                ct_dims.append(ct_dim)
        num_ct_real = prod([ct_dim.extent for ct_dim in ct_dims])

        ops["add"] += num_ops * num_ct_real
        ops["rot"] += num_ops * num_ct_real
        ops["mul"] += roll_extent * num_ct_real
        return ops

    def split_roll_ops(self, ops):
        """Calculate split roll operations."""
        roll = self.kernel.cs[0]
        roll_extent = roll.dim_to_roll.extent
        num_ops = roll_extent

        if (
            roll.dim_to_roll in self.kernel.layout.ct_dims
            and roll.dim_to_roll_by in self.kernel.layout.ct_dims
        ):
            return ops

        ops["add"] += num_ops
        ops["rot"] += num_ops
        ops["mul"] += num_ops
        return ops

    def bsgs_roll_ops(self, ops):
        """Calculate BSGS (Baby-Step Giant-Step) roll operations."""
        roll_ops = {}
        roll_ops["add"] = 0
        roll_ops["mul"] = 0
        roll_ops["rot"] = 0
        roll_ops = self.roll_ops(roll_ops)

        if roll_ops["rot"] >= 0:
            roll_ops["rot"] = 3 * int(math.ceil(math.sqrt(roll_ops["rot"])))

        ops["add"] += roll_ops["add"]
        ops["mul"] += roll_ops["mul"]
        ops["rot"] += roll_ops["rot"]

        return ops

    def rot_roll_ops(self, ops):
        """Calculate rotation roll operations."""
        roll = self.kernel.cs[0]
        ops["rot"] += roll.dim_to_roll_by.extent
        return ops

    def shift_ops(self, ops):
        """Calculate shift operations."""
        num_ops = len(self.kernel.cs[0])
        ops["rot"] += num_ops
        return ops

    def compact_ops(self, ops):
        """Calculate the ops to perform a compaction operation."""
        num_ct_real_reduced = (
            self.kernel.cs[0].layout.num_ct() - self.kernel.layout.num_ct()
        )
        ops["add"] += int(num_ct_real_reduced)
        ops["rot"] += int(num_ct_real_reduced)
        return ops
