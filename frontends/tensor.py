"""
Tensor Frontend Module

This module provides the frontend interface for defining tensor computations in Rotom.

Plaintext Example:
    >>> import numpy as np
    >>> from frontends.tensor import TensorTerm
    >>>
    >>> # Create tensor terms
    >>> a = TensorTerm.Tensor("a", [64, 64], True)  # 64x64 ciphertext matrix
    >>> b = TensorTerm.Tensor("b", [64], False)     # 64-element plaintext vector
    >>>
    >>> # Create computation
    >>> c = a @ b  # Matrix-vector multiplication
    >>>
    >>> # Evaluate with inputs
    >>> inputs = {"a": np.random.rand(64, 64), "b": np.random.rand(64)}
    >>> result = c.eval(inputs)

Layout Example:
    >>> # Create tensor with specific layout
    >>> a = TensorTerm.Tensor("a", [4, 4], True, layout="[0:4:1][1:4:1]")  # Row-major layout
    >>> b = TensorTerm.Tensor("b", [4, 4], True, layout="[1:4:1][0:4:1]")  # Column-major layout
    >>>
    >>> # Operations with layout specification
    >>> c = a.matmul(b, layout="[0:4:1][1:4:1]")  # Result with row-major layout
    >>> d = a.sum(0, layout="[0:1:1]")  # Sum with specific layout
"""

from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from .tensor_args import Conv2dArgs, Conv3dArgs
from .tensor_evaluator import TensorEvaluator


class TensorOp(Enum):
    """Supported tensor operations.

    This enum defines all the tensor operations supported by the frontend,
    including basic arithmetic, matrix operations, and neural network operations.

    Attributes:
        TENSOR: Input tensor placeholder
        CONST: Constant value
        ADD: Element-wise addition
        SUB: Element-wise subtraction
        MUL: Element-wise multiplication
        SUM: Sum along a dimension
        PRODUCT: Product along a dimension
        TRANSPOSE: Matrix transpose
        MATMUL: Matrix multiplication
        BLOCK_MATMUL: Block matrix multiplication
        CONV2D: 2D convolution
        CONV3D: 3D convolution
        POLY_CALL: Polynomial approximation
        RESHAPE: Tensor reshaping
        PERMUTE: Dimension permutation
        INDEX: Tensor indexing
    """

    TENSOR = "Tensor"  # input tensor
    CONST = "Const"  # const
    ADD = "Add"  # element-wise add
    SUB = "Sub"  # element-wise subtract
    MUL = "Mul"  # element-wise mul
    SUM = "Sum"  # sum along a dimension
    PRODUCT = "Product"  # product along a dimension
    TRANSPOSE = "Transpose"  # transpose
    MATMUL = "MatMul"  # matmul
    BLOCK_MATMUL = "Block_MatMul"  # block matmul
    CONV2D = "Conv"  # convolutions
    CONV3D = "Conv3D"  # 3D convolutions
    POLY_CALL = "PolyCall"  # polynomial approximation call
    RESHAPE = "Reshape"  # tensor reshape
    PERMUTE = "Permute"  # permute dims
    INDEX = "Index"
    RESCALE = "Rescale"  # scale division


class TensorTerm:
    """A term in a tensor computation.

    This class implements tensor operations and expressions, supporting basic arithmetic,
    matrix operations, and shape manipulations. TensorTerms can be composed to build
    complex programs for machine learning workloads.

    All operations return new TensorTerm instances that can be further composed.
    Optional tensor layout strings can be specified for any operation to control
    how tensor data is packed into homomorphic encryption vectors.

    Attributes:
        op (TensorOp): The operation this term represents
        cs (list): The children/arguments for this operation
        layout (str, optional): Tensor layout string for this operation

    Example:
        >>> # Create tensor terms
        >>> a = TensorTerm.Tensor("a", [64, 64], secret=True)
        >>> b = TensorTerm.Tensor("b", [64], secret=False)
        >>>
        >>> # Create operations using operators
        >>> c = a @ b  # Matrix multiplication
        >>> d = a + b  # Element-wise addition
        >>> e = a.T    # Transpose
        >>>
        >>> # Create operations using methods
        >>> f = a.sum(0)  # Sum along dimension 0
        >>> g = a.reshape(0, [32, 128])  # Reshape
        >>>
        >>> # Operations with layout specification
        >>> h = a.matmul(b, layout="[0:64:1][1:64:1]")  # With specific layout
        >>> i = a.sum(0, layout="[0:1:1]")  # Sum with layout
    """

    def __init__(self, op: TensorOp, cs: List[Any], layout: Optional[str] = None):
        """Initialize a tensor term.

        Args:
            op (TensorOp): The operation this term represents
            cs (list): List of child terms or arguments for this operation
            layout (str, optional): Tensor layout string for this operation
        """
        self.op = op
        self.cs = cs
        self.layout = layout

    def __hash__(self):
        """Compute hash of the tensor term for use in sets and dictionaries.

        Returns:
            int: Hash value based on string representation
        """
        # Cache the hash to avoid recomputing for large graphs
        if not hasattr(self, "_hash_cache"):
            self._hash_cache = hash(str(self))
        return self._hash_cache

    def __eq__(self, other):
        """Check equality of two tensor terms.

        Args:
            other: Another tensor term to compare with

        Returns:
            bool: True if terms are equal, False otherwise
        """
        return hash(self) == hash(other)

    def __repr__(self):
        """String representation of the tensor term.

        Returns:
            str: Human-readable representation of the composed tensor program
        """
        # Cache the string representation to avoid recomputing for large graphs
        if not hasattr(self, "_str_cache"):
            cs = " ".join([str(c) for c in self.cs])
            match self.op:
                case TensorOp.TENSOR:
                    self._str_cache = str(self.cs[0])
                case TensorOp.MATMUL:
                    self._str_cache = f"(@ {cs})"
                case TensorOp.ADD:
                    self._str_cache = f"(+ {cs})"
                case TensorOp.TRANSPOSE:
                    self._str_cache = f"{cs}.T"
                case TensorOp.CONST:
                    self._str_cache = str(self.cs[0])
                case TensorOp.INDEX:
                    self._str_cache = f"({self.cs[0]}[{self.cs[1]}])"
                case TensorOp.CONV2D:
                    args = Conv2dArgs.from_term(self)
                    self._str_cache = f"(conv2d {str(args.input)} {str(args.filter)})"
                case TensorOp.CONV3D:
                    args = Conv3dArgs.from_term(self)
                    self._str_cache = f"(conv3d {str(args.input)} {str(args.filter)})"
                case _:
                    self._str_cache = f"({self.op} {cs})"
        return self._str_cache

    @staticmethod
    def Tensor(
        name: str, shape: Iterable[int], secret: bool, layout: Optional[str] = None
    ) -> "TensorTerm":
        """Create a tensor placeholder term.

        Args:
            name (str): Variable name for the tensor
            shape (list): List of dimension sizes
            secret (bool): Whether the tensor is encrypted (True) or plaintext (False)
            layout (str, optional): Tensor layout string (e.g., "[0:4:1][1:4:1]")

        Returns:
            TensorTerm: A tensor placeholder term

        Example:
            >>> a = TensorTerm.Tensor("a", [64, 64], True)  # 64x64 ciphertext matrix
            >>> b = TensorTerm.Tensor("b", [64], False)     # 64-element plaintext vector
            >>> c = TensorTerm.Tensor("c", [4, 4], True, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.TENSOR, [name, shape, secret], layout)

    @staticmethod
    def const(value: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Create a public constant tensor term.

        Args:
            value: The constant value
            layout (str, optional): Tensor layout string (e.g., "[0:4:1][1:4:1]")

        Returns:
            TensorTerm: A constant tensor term

        Example:
            >>> c = TensorTerm.const(42)  # Constant value 42
            >>> d = TensorTerm.const(42, layout="[0:1:1]")  # With layout
        """
        return TensorTerm(TensorOp.CONST, [value], layout)

    def __add__(self, other: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Element-wise addition operator.

        Args:
            other (TensorTerm | scalar): The tensor or scalar to add
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the addition

        Example:
            >>> c = a + b  # Element-wise addition
            >>> d = a.__add__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        if not isinstance(other, TensorTerm):
            other = TensorTerm.const(other)
        return TensorTerm(TensorOp.ADD, [self, other], layout)

    def __radd__(self, other: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Right-hand addition to support scalar + TensorTerm."""
        if not isinstance(other, TensorTerm):
            other = TensorTerm.const(other)
        return TensorTerm(TensorOp.ADD, [other, self], layout)

    def __sub__(self, other: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Element-wise subtraction operator.

        Args:
            other (TensorTerm | scalar): The tensor or scalar to subtract
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the subtraction

        Example:
            >>> c = a - b  # Element-wise subtraction
            >>> d = a.__sub__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        if not isinstance(other, TensorTerm):
            other = TensorTerm.const(other)
        return TensorTerm(TensorOp.SUB, [self, other], layout)

    def __mul__(self, other: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Element-wise multiplication operator.

        Args:
            other (TensorTerm | scalar): The tensor or scalar to multiply
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the multiplication

        Example:
            >>> c = a * b  # Element-wise multiplication
            >>> d = a.__mul__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        if not isinstance(other, TensorTerm):
            other = TensorTerm.const(other)
        return TensorTerm(TensorOp.MUL, [self, other], layout)

    def __rmul__(self, other: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Right-hand multiplication to support scalar * TensorTerm."""
        if not isinstance(other, TensorTerm):
            other = TensorTerm.const(other)
        return TensorTerm(TensorOp.MUL, [other, self], layout)

    def sum(self, dim_idx: int, layout: Optional[str] = None) -> "TensorTerm":
        """Sum along a specific dimension.

        Args:
            dim_idx (int): The dimension index to sum along
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the sum

        Example:
            >>> c = a.sum(0)  # Sum along dimension 0
            >>> d = a.sum(0, layout="[0:1:1]")  # With layout
        """
        return TensorTerm(TensorOp.SUM, [self, dim_idx], layout)

    def product(self, dim_idx: int, layout: Optional[str] = None) -> "TensorTerm":
        """Product along a specific dimension.

        Args:
            dim_idx (int): The dimension index to compute product along
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the product

        Example:
            >>> c = a.product(0)  # Product along dimension 0
            >>> d = a.product(0, layout="[0:1:1]")  # With layout
        """
        return TensorTerm(TensorOp.PRODUCT, [self, dim_idx], layout)

    def matmul(self, other: "TensorTerm", layout: Optional[str] = None) -> "TensorTerm":
        """Matrix multiplication method.

        Args:
            other (TensorTerm): The tensor to multiply with
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the matrix multiplication

        Example:
            >>> c = a.matmul(b)  # Matrix multiplication
            >>> d = a.matmul(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.MATMUL, [self, other], layout)

    def __matmul__(
        self, other: "TensorTerm", layout: Optional[str] = None
    ) -> "TensorTerm":
        """Matrix multiplication operator (@).

        Args:
            other (TensorTerm): The tensor to multiply with
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the matrix multiplication

        Example:
            >>> c = a @ b  # Matrix multiplication using @ operator
            >>> d = a.__matmul__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.MATMUL, [self, other], layout)

    def block_matmul(
        self, other: "TensorTerm", layout: Optional[str] = None
    ) -> "TensorTerm":
        """Block matrix multiplication.

        Args:
            other (TensorTerm): The tensor to multiply with
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the block matrix multiplication

        Example:
            >>> c = a.block_matmul(b)  # Block matrix multiplication
            >>> d = a.block_matmul(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.BLOCK_MATMUL, [self, other], layout)

    def transpose(self, layout: Optional[str] = None) -> "TensorTerm":
        """Transpose the tensor.

        Args:
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the transpose

        Example:
            >>> b = a.transpose()  # Transpose of a
            >>> c = a.transpose(layout="[1:4:1][0:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.TRANSPOSE, [self], layout)

    def poly_call(
        self,
        name: str,
        lower_bound: float,
        upper_bound: float,
        layout: Optional[str] = None,
    ) -> "TensorTerm":
        """Call a named polynomial approximation on a bounded domain.

        Tensor layout:
            cs[0] = input term
            cs[1] = name (e.g., "relu", "silu")
            cs[2] = lower_bound
            cs[3] = upper_bound
        """
        return TensorTerm(
            TensorOp.POLY_CALL,
            [self, name, float(lower_bound), float(upper_bound)],
            layout,
        )

    def silu_poly(
        self,
        lower_bound: float = -8.0,
        upper_bound: float = 8.0,
        layout: Optional[str] = None,
    ) -> "TensorTerm":
        """Shorthand for poly_call("silu", lower_bound, upper_bound)."""
        return self.poly_call("silu", lower_bound, upper_bound, layout)

    def reshape(
        self, dim: int, shape: Dict[int, int], layout: Optional[str] = None
    ) -> "TensorTerm":
        """Reshape the tensor.

        Args:
            dim (int): The dimension to reshape
            shape (dict): Dictionary mapping dimension indices to new sizes
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the reshape

        Example:
            >>> b = a.reshape(0, {0: 32, 1: 128})  # Reshape dimension 0
            >>> c = a.reshape(0, {0: 32, 1: 128}, layout="[0:32:1][1:128:1]")  # With layout
        """
        return TensorTerm(TensorOp.RESHAPE, [self, dim, shape], layout)

    def permute(self, dim_map, layout=None):
        """Permute tensor dimensions.

        Args:
            dim_map (dict): Dictionary mapping old dimension indices to new ones
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the permutation

        Example:
            >>> b = a.permute({0: 1, 1: 0})  # Swap dimensions 0 and 1
            >>> c = a.permute({0: 1, 1: 0}, layout="[1:4:1][0:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.PERMUTE, [self, dim_map], layout)

    def __getitem__(self, item: Any, layout: Optional[str] = None) -> "TensorTerm":
        """Index the tensor.

        Args:
            item: The index or slice to apply
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the indexing

        Example:
            >>> b = a[0]  # Index first element
            >>> c = a.__getitem__(0, layout="[0:1:1]")  # With layout
        """
        return TensorTerm(TensorOp.INDEX, [self, item], layout)

    @property
    def T(self) -> "TensorTerm":
        """Transpose property.

        Returns:
            TensorTerm: A new tensor term representing the transpose

        Example:
            >>> b = a.T  # Transpose of a
        """
        return TensorTerm(TensorOp.TRANSPOSE, [self])

    def rescale(self, scale_exp: int, layout: Optional[str] = None) -> "TensorTerm":
        """Rescale the tensor by dividing by 2^scale_exp.

        Args:
            scale_exp (int): The exponent for the scale (e.g., 14 for 2^14)
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the rescaled tensor

        Example:
            >>> b = a.rescale(14)  # Divide by 2^14
            >>> c = a.rescale(14, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.RESCALE, [self, scale_exp], layout)

    @staticmethod
    def conv2d(
        a: "TensorTerm",
        b: "TensorTerm",
        stride: int,
        padding: str,
        layout: Optional[str] = None,
    ) -> "TensorTerm":
        """Create a 2D convolution operation.

        Args:
            a (TensorTerm): Input tensor
            b (TensorTerm): Filter tensor
            stride (int): Stride of the convolution
            padding (str): Padding mode ("valid" or "same")
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the convolution

        Example:
            >>> c = TensorTerm.conv2d(input, filter, 1, "same")
            >>> d = TensorTerm.conv2d(input, filter, 1, "same", layout="[0:32:1][1:32:1][2:64:1]")
        """
        return TensorTerm(TensorOp.CONV2D, [a, b, stride, padding], layout)

    @staticmethod
    def conv3d(
        a: "TensorTerm",
        b: "TensorTerm",
        stride: int,
        padding: str,
        layout: Optional[str] = None,
    ) -> "TensorTerm":
        """Create a 3D convolution operation.

        Conventions (TensorEvaluator + Shape analysis):
        - Input tensor shape: [C_in, D_in, H_in, W_in]
        - Filter tensor shape: [C_out, C_in, K_d, K_h, K_w]
        - Output tensor shape: [C_out, D_out, H_out, W_out]
        - `stride` applies to all spatial dims (D/H/W).
        - `padding` is "valid" or "same".
        """
        return TensorTerm(TensorOp.CONV3D, [a, b, stride, padding], layout)

    def helper_post_order(self, seen):
        """Helper routine for post-order traversal.

        Args:
            seen (set): Set of already visited nodes

        Returns:
            tuple: (list of nodes in post-order, updated seen set)
        """
        if self in seen:
            return [], seen
        match self.op:
            case TensorOp.TENSOR:
                seen.add(self)
                return [self], seen
            case _:
                res = []
                for term in self.cs:
                    if isinstance(term, TensorTerm):
                        _res, _seen = term.helper_post_order(seen)
                        res += _res
                        seen |= _seen
                seen.add(self)
                res.append(self)
                return res, seen

    def post_order(self):
        """Perform post-order traversal of the tensor computation DAG.

        Returns:
            list: List of tensor terms in post-order (children before parents)

        Example:
            >>> # For computation: c = a @ b
            >>> traversal = c.post_order()  # Returns [a, b, c]
        """
        # Cache the result to avoid recomputing for large graphs
        if not hasattr(self, "_post_order_cache"):
            seen = set()
            res, seen = self.helper_post_order(seen)
            self._post_order_cache = res
        return self._post_order_cache

    def root(self):
        """Return the root node of the tensor computation DAG.

        Returns:
            TensorTerm: The root node (final result) of the computation

        Example:
            >>> # For computation: c = a @ b
            >>> root = c.root()  # Returns c
        """
        return self.post_order()[-1]

    def eval(self, inputs):
        """Evaluate this tensor computation using the default TensorEvaluator."""
        evaluator = TensorEvaluator()
        return evaluator.eval(self, inputs)
