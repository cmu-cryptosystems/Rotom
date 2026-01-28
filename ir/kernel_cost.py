"""
Kernel cost modeling for HE operations.

This module provides cost modeling for kernel operations in Rotom,
estimating the computational and communication costs of different operations
based on operation types and network settings. Having a fast cost model is
crucial for layout assignment.

Each operation is deterministically lowered into a set of fixed HE operations,
based on the input and output layouts in the Kernel. This allows Rotom to
find the cost of a kernel without materializing the entire kernel.
"""

import math

from ir.dim import DimType
from ir.kernel import KernelOp
from util.util import get_slot_dims, prod

from .layout_utils import dimension_merging


class KernelCost:
    """
    Cost modeling for kernel operations in HE computations.

    This class provides cost estimation for various kernel operations
    based on operation characteristics. It helps
    the compiler select optimal layouts and operation sequences by
    estimating the computational and communication costs.

    Attributes:
        kernel: The kernel operation to model costs for
        network: Network type ("lan" or "wan") for cost modeling
    """

    def __init__(self, kernel, network):
        """Initialize kernel cost model.

        Args:
            kernel: The kernel operation to model costs for
            network: Network type ("lan" or "wan") for cost modeling
        """
        self.kernel = kernel
        self.network = network

    def cost_model(self):
        """Get the cost model for operations based on network type.

        Returns:
            dict: Dictionary mapping operation types to their costs in milliseconds
        """

        match self.network:
            case "lan":
                # 1Gbps
                # ct = 1.2mb
                # 0.0096 ms per ct.
                return {
                    "comm": 0.0096,
                    "add": 0.0001,
                    "mul": 0.006,
                    "rot": 0.006,
                }
            case "wan":
                # 100mbps
                # 0.096 ms per ct
                return {
                    "comm": 0.096,
                    "add": 0.0001,
                    "mul": 0.006,
                    "rot": 0.006,
                }
            case _:
                raise NotImplementedError(f"network: {self.network}")

    def nops(self, ops):
        """No cost operations (layout transformations).

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Unchanged operation counts (no cost for layout ops)
        """
        return ops

    def tensor_ops(self, ops):
        """No cost for tensor operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Unchanged operation counts (no cost for tensor ops)
        """
        return ops

    def replicate_ops(self, ops):
        """Calculate the operations to replicate and fill a tensor.

        The cost of replication is calculated as follows:
        1. If there is replication in the slot dimensions, find the number of masks
        needed. The number of masks is determined by the number of filled dimensions
        shifted.
        2. How many rotate-and-sum routines are needed to fill in gaps? The size of each
        rot-and-sum routine is based on how large the replicated dimension within the slot
        dimensions are. The total number of rotate-and-sum routines is based on the number
        of unique ciphertexts.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts including replication costs
        """
        from_layout = dimension_merging(self.kernel.cs[0].layout)
        to_layout = dimension_merging(self.kernel.layout)

        # if slot dimensions are the same, then replicating ct is free
        if len(from_layout.slot_dims) == len(to_layout.slot_dims) and all(
            from_slot_dim == to_slot_dim
            for from_slot_dim, to_slot_dim in zip(
                from_layout.slot_dims, to_layout.slot_dims
            )
        ):
            return ops

        # 1. check to see if filled dimension is moved
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

        # 2. find how many rot-and-sum routines are needed
        rot_sum_per_ct = 1
        from_slot_dim_indices = set()
        for i, from_dim in enumerate(from_layout.slot_dims[::-1]):
            from_slot_dim_indices.add(from_dim)
        to_slot_dim_indices = set()
        for i, to_dim in enumerate(to_layout.slot_dims[::-1]):
            to_slot_dim_indices.add(to_dim)
        slot_dim_diff = to_slot_dim_indices - from_slot_dim_indices
        for dim in slot_dim_diff:
            if dim.dim is None:
                rot_sum_per_ct *= dim.extent

        # 3. find number of unique ct
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
        """Calculate basic arithmetic operations based on number of ciphertexts.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with addition operations
        """
        ops["add"] += self.kernel.layout.num_ct()
        return ops

    def mul_arith_ops(self, ops):
        """Calculate multiplication operations based on number of ciphertexts.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with multiplication operations
        """
        ops["mul"] += self.kernel.layout.num_ct()
        return ops

    def sum_ops(self, ops):
        """Calculate sum operations along a dimension.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with sum operations
        """
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
        """Calculate product operations along a dimension.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with product operations
        """
        for ct_dim in self.kernel.cs[0].layout.ct_dims:
            if ct_dim.dim == self.kernel.cs[1]:
                ops["mul"] += 1
        num_ct_reals = self.kernel.layout.num_ct()
        for slot_dim in self.kernel.cs[0].layout.slot_dims:
            if slot_dim.dim == self.kernel.cs[1]:
                ops["mul"] += math.ceil(math.log2(slot_dim.extent)) * num_ct_reals
                ops["rot"] += math.ceil(math.log2(slot_dim.extent)) * num_ct_reals
        return ops

    def matmul_ops(self, ops, layout, sum_dim):
        """Calculate the operations to perform a matrix operation.

        The cost of matrix operations is as follows:
        1. A multiplication between data aligned vectors
        2. A summation across ct_dims, if applicable
        3. A summation across slot_dims, if applicable
        4. A masking cost to remove garbage values, if applicable

        Args:
            ops: Dictionary of operation counts
            layout: The layout for the matrix operation
            sum_dim: The dimension to sum along

        Returns:
            dict: Updated operation counts with matrix multiplication costs
        """
        # 1. calculate number of multiplications
        ops["mul"] += layout.num_ct()

        # 2. calculate sum along ct dimensions
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

        # 3. calculate sum along slot dimensions (* number of ct)
        slot_sum_dims = []
        for dim in layout.slot_dims:
            if dim.dim == sum_dim:
                slot_sum_dims.append(dim.extent)
        rot_sum_op_count = remaining_cts * math.ceil(
            math.log2(int(prod(slot_sum_dims)))
        )
        ops["add"] += rot_sum_op_count
        ops["rot"] += rot_sum_op_count

        # 4. calculate masking cost
        if any([dim.dim_type == DimType.EMPTY for dim in self.kernel.layout.slot_dims]):
            ops["mul"] += remaining_cts
        return ops

    def bsgs_matmul_ops(self, ops):
        """Calculate BSGS (Baby-Step Giant-Step) matrix multiplication operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with BSGS matrix multiplication costs
        """
        # get cost of matmul ops
        matmul_ops = {}
        matmul_ops["add"] = 0
        matmul_ops["mul"] = 0
        matmul_ops["rot"] = 0
        matmul_ops = self.matmul_ops(matmul_ops, self.kernel.cs[2].layout, 0)

        if self.kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL:
            matmul_ops = KernelCost(self.kernel.cs[1], self.network).rot_roll_ops(
                matmul_ops
            )
        else:
            matmul_ops = KernelCost(self.kernel.cs[2], self.network).rot_roll_ops(
                matmul_ops
            )
        matmul_ops["rot"] -= int(matmul_ops["rot"] - 2 * math.sqrt(matmul_ops["rot"]))

        ops["add"] += matmul_ops["add"]
        ops["mul"] += matmul_ops["mul"]
        ops["rot"] += matmul_ops["rot"]
        return ops

    def strassen_ops(self, ops):
        """Calculate Strassen matrix multiplication operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with Strassen matrix multiplication costs
        """
        # get cost of matmul ops
        matmul_ops = {}
        matmul_ops["add"] = 0
        matmul_ops["mul"] = 0
        matmul_ops["rot"] = 0
        matmul_ops = self.matmul_ops(matmul_ops, self.kernel.cs[0].layout, 1)

        # HACK: come back and fix strassen costs
        # remove 1 matmul
        # add 8 additions
        matmul_ops["mul"] = int(6 / 8 * matmul_ops["mul"])
        matmul_ops["add"] = int(6 / 8 * matmul_ops["add"])
        matmul_ops["add"] += 8

        ops["add"] += matmul_ops["add"]
        ops["mul"] += matmul_ops["mul"]
        ops["rot"] += matmul_ops["rot"]

        # forces strassens if enabled
        ops["add"] = -100000
        ops["mul"] = -100000
        ops["rot"] = -100000
        return ops

    def conv2d_roll_ops(self, ops):
        """Calculate 2D convolution operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with convolution costs
        """
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

    def roll_ops(self, ops):
        """Calculate the ops to perform a roll operation

        The cost of a roll is based on the size of the rolled
        dimension. Naively, rolls can be performed using
        O(2*|dim|) multiplications, additions, and rotations. Rotations can be
        further reduced using a baby-step giant-step optimization.
        """
        roll = self.kernel.cs[0]
        roll_extent = roll.dim_to_roll.extent

        num_ops = 2 * roll_extent

        # if the rolls are only between ct_dimensions, then this transformation is free
        if (
            roll.dim_to_roll in self.kernel.layout.ct_dims
            and roll.dim_to_roll_by in self.kernel.layout.ct_dims
        ):
            return ops

        # get unique cts
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
        """Calculate split roll operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with split roll costs
        """
        roll = self.kernel.cs[0]
        roll_extent = roll.dim_to_roll.extent
        num_ops = roll_extent

        # if the rolls are only between ct_dimensions, then this transformation is free
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
        """Calculate BSGS (Baby-Step Giant-Step) roll operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with BSGS roll costs
        """
        # get cost of roll ops
        roll_ops = {}
        roll_ops["add"] = 0
        roll_ops["mul"] = 0
        roll_ops["rot"] = 0
        roll_ops = self.roll_ops(roll_ops)

        # reduce rotations by sqrt factor
        if roll_ops["rot"] >= 0:
            roll_ops["rot"] = 3 * int(math.ceil(math.sqrt(roll_ops["rot"])))

        ops["add"] += roll_ops["add"]
        ops["mul"] += roll_ops["mul"]
        ops["rot"] += roll_ops["rot"]

        return ops

    def rot_roll_ops(self, ops):
        """Calculate rotation roll operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with rotation roll costs
        """
        roll = self.kernel.cs[0]
        ops["rot"] += roll.dim_to_roll_by.extent
        return ops

    def shift_ops(self, ops):
        """Calculate shift operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with shift costs
        """
        num_ops = len(self.kernel.cs[0])
        ops["rot"] += num_ops
        return ops

    def compact_ops(self, ops):
        """Calculate the ops to perform a compaction operation

        The cost of compaction is based on the number of ciphertexts
        that are reduced. Each reduced ciphertext corresponds to a
        rotate-and-sum.
        """
        num_ct_real_reduced = (
            self.kernel.cs[0].layout.num_ct() - self.kernel.layout.num_ct()
        )
        ops["add"] += int(num_ct_real_reduced)
        ops["rot"] += int(num_ct_real_reduced)
        return ops

    def poly_ops(self, ops):
        """Calculate polynomial evaluation operations.

        HACK: add additional costs based on the number of ciphertexts
        (pseudo bootstrap cost)

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with polynomial evaluation costs
        """
        cts = self.kernel.layout.num_ct()
        ops["add"] += cts
        ops["rot"] += cts
        ops["mul"] += cts
        return ops

    def conversion_ops(self, ops):
        """Calculate conversion operations (assume conversions are very expensive).

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with conversion costs
        """
        from_dims = self.kernel.cs[0]
        to_dims = self.kernel.cs[1]

        _, from_slot_dims = get_slot_dims(from_dims, self.kernel.layout.n)
        _, to_slot_dims = get_slot_dims(to_dims, self.kernel.layout.n)

        # calculate cost of moving slot dimensions to ct dimensions
        for from_dim, to_dim in zip(from_slot_dims, to_slot_dims):
            if from_dim != to_dim:
                if ops["mul"] == 0:
                    ops["mul"] = to_dim.extent
                    ops["rot"] = to_dim.extent
                else:
                    ops["mul"] *= to_dim.extent
                    ops["rot"] *= to_dim.extent

        # multiply per ct vector
        ops["mul"] *= self.kernel.cs[2].layout.num_ct()
        ops["rot"] *= self.kernel.cs[2].layout.num_ct()

        # calculate cost of moving ct dimensions to slot dimensions
        ops["add"] = ops["rot"]
        ops["rot"] *= 2

        return ops

    def conv2d_ops(self, ops):
        """Calculate convolution operations.

        Args:
            ops: Dictionary of operation counts

        Returns:
            dict: Updated operation counts with convoltuion costs
        """
        input_dims = self.kernel.cs[0].layout.get_dims()
        output_dims = self.kernel.layout.get_dims()

        input_dim_len = int(prod([input_dim.extent for input_dim in input_dims]))
        output_dim_len = int(prod([output_dim.extent for output_dim in output_dims]))

        ops["mul"] += output_dim_len
        ops["add"] += output_dim_len * int(math.log2(input_dim_len))
        ops["rot"] += int(math.sqrt(output_dim_len * int(math.log2(input_dim_len))))
        return ops

    def ops(self):
        """Calculate the number of operations required for a given layout operation.

        Returns:
            dict: Dictionary mapping operation types to their counts
        """
        ops = {
            "add": 0,
            "mul": 0,
            "rot": 0,
        }

        # if the layout is in plaintext, then the cost of the layout is free
        if not self.kernel.layout.secret:
            return ops

        match self.kernel.op:
            case KernelOp.TENSOR | KernelOp.PUNCTURED_TENSOR:
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
            case KernelOp.SUM:
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
            case KernelOp.CONV2D:
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
            case KernelOp.POLY:
                ops = self.poly_ops(ops)
            case KernelOp.CONVERSION:
                ops = self.conversion_ops(ops)
            case KernelOp.COMBINE:
                for cs_kernel in self.kernel.cs:
                    cs_ops = KernelCost(cs_kernel, self.network).ops()
                    for k, v in cs_ops.items():
                        ops[k] += v
            case _:
                raise NotImplementedError(self.kernel.op)
        return ops

    def comm_cost(self):
        """Calculate communication cost for the kernel.

        Returns:
            float: Communication cost in milliseconds
        """
        cost_model = self.cost_model()
        total_num_ct_real = 0
        for term in self.kernel.post_order():
            if (
                term.op == KernelOp.TENSOR or term.op == KernelOp.PUNCTURED_TENSOR
            ) and term.layout.secret:
                total_num_ct_real += term.layout.num_ct_unique()
        return cost_model["comm"] * total_num_ct_real

    def real_comm_cost(self):
        """Calculate real communication cost for the kernel.

        Returns:
            float: Real communication cost in milliseconds
        """
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
        """Return the operation cost of a kernel.

        Returns:
            float: Total operation cost in milliseconds
        """
        cost_model = self.cost_model()
        ops = self.ops()
        total = 0
        for k, v in ops.items():
            total += cost_model[k] * v
        return total

    def total_operations(self):
        """Return the number of operations for a kernel and its children.

        Returns:
            dict: Dictionary mapping operation types to their total counts
        """
        ops = {}
        for term in self.kernel.post_order():
            kernel_cost_term = KernelCost(term, self.network)
            op_cost = kernel_cost_term.ops()
            for k, v in op_cost.items():
                if k not in ops:
                    ops[k] = v
                else:
                    ops[k] += v
        return ops

    def total_cost(self):
        """Return the total cost for a kernel and its children.

        Returns:
            float: Total cost including operation and communication costs
        """
        total_ops = self.total_operations()
        op_cost = 0
        for k, v in total_ops.items():
            op_cost += self.cost_model()[k] * v
        comm_cost = self.comm_cost()
        return op_cost + comm_cost

    def depth(self):
        """Return the multiplicative depth of a kernel.

        Returns:
            int: The multiplicative depth of the kernel
        """
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
                        | KernelOp.PRODUCT
                        | KernelOp.COMBINE
                    ):
                        pass
                    case KernelOp.MUL | KernelOp.ROLL | KernelOp.CONVERSION:
                        d += 1
                    case KernelOp.MATMUL | KernelOp.CONV2D:
                        # HACK: depth actually varies if masking is required or not
                        d += 2
                    case KernelOp.POLY:
                        # TODO: pass for now, but depth should be parameterized on the
                        # mul depth of poly
                        pass
                    case _:
                        raise NotImplementedError(term.op)
        return d
