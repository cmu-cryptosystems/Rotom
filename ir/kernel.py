"""
Kernel operations and representations.

This module defines kernel operations that represent high-level tensor
operations in the intermediate representation. Kernels bridge the gap
between tensor operations and homomorphic encryption operations by
providing a structured representation of computations.

Key Components:

- KernelOp: Enumeration of supported kernel operations
- Kernel: Representation of a kernel with layout IR terms
"""

from enum import Enum

from .layout_utils import dimension_merging


class KernelOp(Enum):
    """
    Enumeration of kernel operations for tensor computations.

    Defines all the kernel operations supported by Rotom,
    including tensor operations, matrix operations, conversions, and
    indexing operations. These operations form the high-level interface
    between tensor computations and homomorphic encryption.

    Operation Categories:
        Tensor ops: TENSOR, CS, CONST, ADD, SUB, MUL, SUM, PRODUCT
        Matrix ops: MATMUL, BLOCK_MATMUL, BSGS_MATMUL, STRASSEN_MATMUL
        Convolution: CONV2D, CONV2D_ROLL
        Polynomial: POLY
        Conversions: CONVERSION
        Replications: REPLICATE
        Rotations: ROLL, ROT_ROLL, SPLIT_ROLL, BSGS_ROLL, BSGS_ROT_ROLL, SHIFT
        Shape ops: COMPACT, TRANSPOSE, RESHAPE, PERMUTE
        Indexing: INDEX, COMBINE, REORDER
    """

    # tensor ops
    TENSOR = "TENSOR"
    PUNCTURED_TENSOR = "PUNCTURED_TENSOR"
    CS = "CS"
    CONST = "CONST"
    ADD = "ADD"
    SUB = "SUB"
    MUL = "MUL"
    SUM = "SUM"
    PRODUCT = "PRODUCT"
    MATMUL = "MATMUL"
    BLOCK_MATMUL = "BLOCK_MATMUL"
    BSGS_MATMUL = "BSGS_MATMUL"
    STRASSEN_MATMUL = "STRASSENS"
    CONV2D = "CONV2D"
    CONV2D_ROLL = "CONV2D_ROLL"
    # poly
    POLY = "POLY"
    # conversions
    CONVERSION = "CONVERSION"
    REPLICATE = "REPLICATE"
    ROLL = "ROLL"
    SPLIT_ROLL = "SPLIT_ROLL"
    BSGS_ROLL = "BSGS_ROLL"
    ROT_ROLL = "ROT_ROLL"
    BSGS_ROT_ROLL = "BSGS_ROT_ROLL"
    SHIFT = "SHIFT"
    COMPACT = "COMPACT"
    TRANSPOSE = "TRANSPOSE"
    RESHAPE = "RESHAPE"
    PERMUTE = "PERMUTE"
    # index
    INDEX = "INDEX"
    SELECT = "SELECT"
    COMBINE = "COMBINE"
    REORDER = "REORDER"
    RESCALE = "RESCALE"


class Kernel:
    """
    Represents a kernel operation in the layout IR.

    A Kernel encapsulates a single tensor operation with its operands
    and associated layout information. Kernels are the primary unit
    of computation in the IR and are used to represent tensor operations
    like matrix multiplication, convolution, and element-wise operations.

    Attributes:
        op: The kernel operation type (from KernelOp enum)
        cs: List of child kernels (operands)
        layout: The layout associated with this kernel's output
    """

    def __init__(self, op, cs, layout):
        """
        Create a kernel operation.

        Args:
            op: The kernel operation type
            cs: List of child kernels (operands)
            layout: The layout for this kernel's output
        """
        self.op = op
        self.cs = cs
        self.layout = layout

    def __repr__(self):
        """String representation of the kernel.

        Returns:
            str: Human-readable representation of the kernel
        """
        layout = dimension_merging(self.layout)
        if self.op == KernelOp.TENSOR or self.op == KernelOp.PUNCTURED_TENSOR:
            return f"{self.op}: {self.layout.term.cs[0]} {layout.layout_str()}"
        else:
            return f"{self.op}: {layout.layout_str()}"

    def copy(self):
        """Create a copy of the kernel.

        Returns:
            Kernel: A new kernel with copied children and same layout
        """
        return Kernel(self.op, self.cs.copy(), self.layout)

    def __len__(self):
        """Get the length of the kernel (layout dimensions).

        Returns:
            int: Number of dimensions in the layout
        """
        return len(self.layout)

    def unique_str(self):
        """Generate a unique string representation of the kernel.

        Returns:
            str: Unique string identifier for the kernel
        """
        cs_unique_str = [
            cs.unique_str if isinstance(cs, Kernel) else str(cs) for cs in self.cs
        ]
        return f"{self.op}:{cs_unique_str}:{self.layout}"

    def __hash__(self):
        """Compute hash of the kernel.

        Returns:
            int: Hash value based on unique string representation
        """
        return hash(self.unique_str())

    def __eq__(self, other):
        """Check equality of two kernels.

        Args:
            other: Another kernel to compare with

        Returns:
            bool: True if kernels are equal, False otherwise
        """
        return hash(self) == hash(other)

    def helper_post_order(self, seen):
        """Helper routine for post-order traversal.

        Args:
            seen: Set of already visited nodes

        Returns:
            tuple: (list of nodes in post-order, updated seen set)
        """
        if self in seen:
            return [], seen
        match self.op:
            case (
                KernelOp.TENSOR
                | KernelOp.PUNCTURED_TENSOR
                | KernelOp.CONST
                | KernelOp.CS
            ):
                seen.add(self)
                return [self], seen
            case KernelOp.SUM | KernelOp.PRODUCT | KernelOp.INDEX | KernelOp.RESCALE:
                res = []
                _res, _seen = self.cs[0].helper_post_order(seen)
                res += _res
                seen |= _seen
                seen.add(self)
                res.append(self)
                return res, seen
            case (
                KernelOp.ROLL
                | KernelOp.SPLIT_ROLL
                | KernelOp.ROT_ROLL
                | KernelOp.BSGS_ROLL
                | KernelOp.BSGS_ROT_ROLL
            ):
                res = []
                _res, _seen = self.cs[1].helper_post_order(seen)
                res += _res
                seen |= _seen
                seen.add(self)
                res.append(self)
                return res, seen
            case KernelOp.CONVERSION:
                res = []
                _res, _seen = self.cs[2].helper_post_order(seen)
                res += _res
                seen |= _seen
                seen.add(self)
                res.append(self)
                return res, seen
            case KernelOp.BSGS_MATMUL:
                res = []
                for term in self.cs[1:]:
                    _res, _seen = term.helper_post_order(seen)
                    res += _res
                    seen |= _seen
                seen.add(self)
                res.append(self)
                return res, seen
            case _:
                res = []
                for term in self.cs:
                    _res, _seen = term.helper_post_order(seen)
                    res += _res
                    seen |= _seen
                seen.add(self)
                res.append(self)
                return res, seen

    def post_order(self):
        """Perform post-order traversal of the kernel computation DAG.

        Returns:
            list: List of kernels in post-order (children before parents)
        """
        seen = set()
        res, seen = self.helper_post_order(seen)
        return res


class KernelDag:
    """Represents a directed acyclic graph of kernel operations.

    A KernelDag groups multiple kernels into a single computation graph,
    allowing for complex multi-kernel operations and optimizations.

    Attributes:
        kernel: The root kernel operation
        cs: List of child kernel operations
    """

    def __init__(self, kernel, cs_kernels):
        """Initialize a kernel DAG.

        Args:
            kernel: The root kernel operation
            cs_kernels: List of child kernel operations
        """
        self.kernel = kernel
        self.cs = cs_kernels

    def __repr__(self):
        """String representation of the kernel DAG.

        Returns:
            str: Human-readable representation of the DAG
        """
        return f"KernelDag: {self.kernel}"

    def helper_post_order(self, seen):
        """Helper routine for post-order traversal.

        Args:
            seen: Set of already visited nodes

        Returns:
            tuple: (list of nodes in post-order, updated seen set)
        """
        if self in seen:
            return [], seen

        res = []
        for term in self.cs:
            _res, _seen = term.helper_post_order(seen)
            res += _res
            seen |= _seen
        seen.add(self)
        res.append(self)
        return res, seen

    def post_order(self):
        """Perform post-order traversal of the kernel DAG.

        Returns:
            list: List of kernels in post-order (children before parents)
        """
        seen = set()
        res, seen = self.helper_post_order(seen)
        return res
