"""
Core kernel cost model, composed from operator-family mixins.
"""

import math

from ir.kernel import KernelOp
from util.util import get_slot_dims

from .kernel_cost_conv import ConvCostMixin
from .kernel_cost_matmul import MatmulCostMixin
from .kernel_cost_roll import RollCostMixin
from .layout_utils import dimension_merging


class KernelCost(MatmulCostMixin, ConvCostMixin, RollCostMixin):
    def __init__(self, kernel, network):
        self.kernel = kernel
        self.network = network

    def cost_model(self):
        match self.network:
            case "lan":
                return {
                    "comm": 0.0096,
                    "add": 0.5,
                    "mul": 10.0,
                    "rot": 4.0,
                }
            case "wan":
                return {
                    "comm": 0.096,
                    "add": 0.5,
                    "mul": 10.0,
                    "rot": 4.0,
                }
            case _:
                raise NotImplementedError(f"network: {self.network}")

    def nops(self, ops):
        return ops

    def tensor_ops(self, ops):
        return ops

    def replicate_ops(self, ops):
        from_layout = dimension_merging(self.kernel.cs[0].layout)
        to_layout = dimension_merging(self.kernel.layout)

        if len(from_layout.slot_dims) == len(to_layout.slot_dims) and all(
            from_slot_dim == to_slot_dim
            for from_slot_dim, to_slot_dim in zip(
                from_layout.slot_dims, to_layout.slot_dims
            )
        ):
            return ops

        num_masks = 0
        from_dim_indices = set()
        for i, from_dim in enumerate(from_layout.dims[::-1]):
            from_dim_indices.add((i, from_dim))
        to_dim_indices = set()
        for i, to_dim in enumerate(to_layout.dims[::-1]):
            to_dim_indices.add((i, to_dim))
        fill_dim_diff = from_dim_indices - to_dim_indices
        for _, dim in fill_dim_diff:
            if dim.dim is not None:
                if not num_masks:
                    num_masks = dim.extent
                else:
                    num_masks *= dim.extent

        rot_sum_per_ct = 1
        from_slot_dim_indices = set()
        for from_dim in from_layout.slot_dims[::-1]:
            from_slot_dim_indices.add(from_dim)
        to_slot_dim_indices = set()
        for to_dim in to_layout.slot_dims[::-1]:
            to_slot_dim_indices.add(to_dim)
        slot_dim_diff = to_slot_dim_indices - from_slot_dim_indices
        for dim in slot_dim_diff:
            if dim.dim is None:
                rot_sum_per_ct *= dim.extent

        unique_ct = 1
        ct_dim_diff = set(to_layout.ct_dims) - set(from_layout.ct_dims)
        for dim in ct_dim_diff:
            if dim.dim is not None:
                unique_ct *= dim.extent

        ops["mul"] = num_masks
        ops["add"] = (
            unique_ct * math.ceil(math.log2(rot_sum_per_ct)) if rot_sum_per_ct else 0
        )
        ops["rot"] = (
            unique_ct * math.ceil(math.log2(rot_sum_per_ct)) if rot_sum_per_ct else 0
        )
        return ops

    def basic_arith_ops(self, ops):
        ops["add"] += self.kernel.layout.num_ct()
        return ops

    def mul_arith_ops(self, ops):
        ops["mul"] += self.kernel.layout.num_ct()
        return ops

    def sum_ops(self, ops):
        sum_dim_idx = self.kernel.layout.term.cs[1]
        for ct_dim in self.kernel.cs[0].layout.ct_dims:
            if ct_dim.dim == sum_dim_idx:
                ops["add"] += 1
        num_ct_reals = self.kernel.layout.num_ct()
        for slot_dim in self.kernel.cs[0].layout.slot_dims:
            if slot_dim.dim == sum_dim_idx:
                ops["add"] += math.ceil(math.log2(slot_dim.extent)) * num_ct_reals
                ops["rot"] += math.ceil(math.log2(slot_dim.extent)) * num_ct_reals
        return ops

    def product_ops(self, ops):
        for ct_dim in self.kernel.cs[0].layout.ct_dims:
            if ct_dim.dim == self.kernel.cs[1]:
                ops["mul"] += 1
        num_ct_reals = self.kernel.layout.num_ct()
        for slot_dim in self.kernel.cs[0].layout.slot_dims:
            if slot_dim.dim == self.kernel.cs[1]:
                ops["mul"] += math.ceil(math.log2(slot_dim.extent)) * num_ct_reals
                ops["rot"] += math.ceil(math.log2(slot_dim.extent)) * num_ct_reals
        return ops

    def bsgs_matmul_ops(self, ops):
        matmul_ops = {"add": 0, "mul": 0, "rot": 0}
        matmul_ops = self.matmul_ops(matmul_ops, self.kernel.cs[2].layout, 0)

        from .kernel_cost_base import KernelCost as _KernelCost  # type: ignore

        if self.kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL:
            matmul_ops = _KernelCost(self.kernel.cs[1], self.network).rot_roll_ops(
                matmul_ops
            )
        else:
            matmul_ops = _KernelCost(self.kernel.cs[2], self.network).rot_roll_ops(
                matmul_ops
            )
        matmul_ops["rot"] -= int(matmul_ops["rot"] - 2 * math.sqrt(matmul_ops["rot"]))

        ops["add"] += matmul_ops["add"]
        ops["mul"] += matmul_ops["mul"]
        ops["rot"] += matmul_ops["rot"]
        return ops

    def strassen_ops(self, ops):
        matmul_ops = {"add": 0, "mul": 0, "rot": 0}
        matmul_ops = self.matmul_ops(matmul_ops, self.kernel.cs[0].layout, 1)

        matmul_ops["mul"] = int(6 / 8 * matmul_ops["mul"])
        matmul_ops["add"] = int(6 / 8 * matmul_ops["add"])
        matmul_ops["add"] += 8

        ops["add"] += matmul_ops["add"]
        ops["mul"] += matmul_ops["mul"]
        ops["rot"] += matmul_ops["rot"]

        ops["add"] = -100000
        ops["mul"] = -100000
        ops["rot"] = -100000
        return ops

    def poly_ops(self, ops):
        cts = self.kernel.layout.num_ct()
        ops["add"] += cts
        ops["rot"] += cts
        ops["mul"] += cts
        return ops

    def poly_call_ops(self, ops):
        cts = self.kernel.layout.num_ct()
        ops["add"] += cts
        ops["rot"] += cts
        ops["mul"] += cts
        return ops

    def conversion_ops(self, ops):
        from_dims = self.kernel.cs[0]
        to_dims = self.kernel.cs[1]

        _, from_slot_dims = get_slot_dims(from_dims, self.kernel.layout.n)
        _, to_slot_dims = get_slot_dims(to_dims, self.kernel.layout.n)

        for from_dim, to_dim in zip(from_slot_dims, to_slot_dims):
            if from_dim != to_dim:
                if ops["mul"] == 0:
                    ops["mul"] = to_dim.extent
                    ops["rot"] = to_dim.extent
                else:
                    ops["mul"] *= to_dim.extent
                    ops["rot"] *= to_dim.extent

        ops["mul"] *= self.kernel.cs[2].layout.num_ct()
        ops["rot"] *= self.kernel.cs[2].layout.num_ct()

        ops["add"] = ops["rot"]
        ops["rot"] *= 2

        return ops

    def ops(self):
        ops = {"add": 0, "mul": 0, "rot": 0}

        if not self.kernel.layout.secret:
            return ops

        match self.kernel.op:
            case KernelOp.TENSOR | KernelOp.PUNCTURED_TENSOR | KernelOp.CONST:
                ops = self.tensor_ops(ops)
            case (
                KernelOp.CS
                | KernelOp.TRANSPOSE
                | KernelOp.RESHAPE
                | KernelOp.PERMUTE
                | KernelOp.INDEX
                | KernelOp.SELECT
                | KernelOp.REORDER
                | KernelOp.RESCALE
            ):
                ops = self.nops(ops)
            case KernelOp.REPLICATE:
                ops = self.replicate_ops(ops)
            case KernelOp.ADD | KernelOp.SUB:
                ops = self.basic_arith_ops(ops)
            case KernelOp.MUL:
                ops = self.mul_arith_ops(ops)
            case KernelOp.SUM | KernelOp.MEAN:
                ops = self.sum_ops(ops)
            case KernelOp.PRODUCT:
                ops = self.product_ops(ops)
            case KernelOp.MATMUL:
                ops = self.matmul_ops(ops, self.kernel.cs[0].layout, 1)
            case KernelOp.BLOCK_MATMUL:
                ops = self.matmul_ops(ops, self.kernel.cs[0].layout, 2)
            case KernelOp.BSGS_MATMUL:
                ops = self.bsgs_matmul_ops(ops)
            case KernelOp.STRASSEN_MATMUL:
                ops = self.strassen_ops(ops)
            case KernelOp.CONV2D_ROLL:
                ops = self.conv2d_roll_ops(ops)
            case KernelOp.CONV2D | KernelOp.CONV3D:
                # ConvCostMixin only defines conv2d_ops; it is layout-geometry based and
                # applies equally to 3D conv kernels.
                ops = self.conv2d_ops(ops)
            case KernelOp.ROLL:
                ops = self.roll_ops(ops)
            case KernelOp.SPLIT_ROLL:
                ops = self.split_roll_ops(ops)
            case KernelOp.BSGS_ROLL:
                ops = self.bsgs_roll_ops(ops)
            case KernelOp.ROT_ROLL:
                ops = self.rot_roll_ops(ops)
            case KernelOp.BSGS_ROT_ROLL:
                ops = self.nops(ops)
            case KernelOp.SHIFT:
                ops = self.shift_ops(ops)
            case KernelOp.COMPACT:
                ops = self.compact_ops(ops)
            case KernelOp.POLY_CALL:
                ops = self.poly_call_ops(ops)
            case KernelOp.CONVERSION:
                ops = self.conversion_ops(ops)
            case KernelOp.COMBINE:
                from .kernel_cost_base import KernelCost as _KernelCost  # type: ignore

                for cs_kernel in self.kernel.cs:
                    cs_ops = _KernelCost(cs_kernel, self.network).ops()
                    for k, v in cs_ops.items():
                        ops[k] += v
            case _:
                raise NotImplementedError(self.kernel.op)
        return ops

    def comm_cost(self):
        cost_model = self.cost_model()
        total_num_ct_real = 0
        for term in self.kernel.post_order():
            if (
                term.op == KernelOp.TENSOR or term.op == KernelOp.PUNCTURED_TENSOR
            ) and term.layout.secret:
                total_num_ct_real += term.layout.num_ct_unique()
        return cost_model["comm"] * total_num_ct_real

    def real_comm_cost(self):
        total_num_ct_real = 0
        for term in self.kernel.post_order():
            if (
                term.op == KernelOp.TENSOR or term.op == KernelOp.PUNCTURED_TENSOR
            ) and term.layout.secret:
                total_num_ct_real += term.layout.num_ct_unique()
        if self.network == "lan":
            return 0.0096 * total_num_ct_real
        else:
            return 0.096 * total_num_ct_real

    def op_cost(self):
        cost_model = self.cost_model()
        ops = self.ops()
        total = 0
        for k, v in ops.items():
            total += cost_model[k] * v
        return total

    def total_operations(self):
        ops = {}
        from .kernel_cost_base import KernelCost as _KernelCost  # type: ignore

        for term in self.kernel.post_order():
            kernel_cost_term = _KernelCost(term, self.network)
            op_cost = kernel_cost_term.ops()
            for k, v in op_cost.items():
                if k not in ops:
                    ops[k] = v
                else:
                    ops[k] += v
        return ops

    def total_cost(self):
        total_ops = self.total_operations()
        op_cost = 0
        for k, v in total_ops.items():
            op_cost += self.cost_model()[k] * v
        comm_cost = self.comm_cost()
        return op_cost + comm_cost

    def depth(self):
        d = 0
        for term in self.kernel.post_order():
            if term.layout.secret:
                match term.op:
                    case (
                        KernelOp.TENSOR
                        | KernelOp.COMPACT
                        | KernelOp.REPLICATE
                        | KernelOp.TRANSPOSE
                        | KernelOp.SHIFT
                        | KernelOp.ADD
                        | KernelOp.SUB
                        | KernelOp.SUM
                        | KernelOp.MEAN
                        | KernelOp.PRODUCT
                        | KernelOp.COMBINE
                    ):
                        pass
                    case KernelOp.MUL | KernelOp.ROLL | KernelOp.CONVERSION:
                        d += 1
                    case KernelOp.MATMUL | KernelOp.CONV2D | KernelOp.CONV3D:
                        d += 2
                    case _:
                        raise NotImplementedError(term.op)
        return d
