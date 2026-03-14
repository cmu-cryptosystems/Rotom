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

import math
from enum import Enum

import numpy as np

from .poly import APPROX_RELU_CHEBYSHEV_COEFFS


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
        POLY: Polynomial approximation
        POLY_CALL: Polynomial approximation call
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
    POLY = "Poly"  # polynomial approximation
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

    def __init__(self, op, cs, layout=None):
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
                    self._str_cache = f"(conv2d {str(self.cs[0])} {str(self.cs[1])})"
                case TensorOp.POLY:
                    func = self.cs[1] if len(self.cs) > 1 else "identity"
                    self._str_cache = f"(poly {cs} {func!r})"
                case _:
                    self._str_cache = f"({self.op} {cs})"
        return self._str_cache

    @staticmethod
    def Tensor(name, shape, secret, layout=None):
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
    def const(value, layout=None):
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

    def __add__(self, other, layout=None):
        """Element-wise addition operator.

        Args:
            other (TensorTerm): The tensor to add
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the addition

        Example:
            >>> c = a + b  # Element-wise addition
            >>> d = a.__add__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.ADD, [self, other], layout)

    def __sub__(self, other, layout=None):
        """Element-wise subtraction operator.

        Args:
            other (TensorTerm): The tensor to subtract
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the subtraction

        Example:
            >>> c = a - b  # Element-wise subtraction
            >>> d = a.__sub__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.SUB, [self, other], layout)

    def __mul__(self, other, layout=None):
        """Element-wise multiplication operator.

        Args:
            other (TensorTerm): The tensor to multiply
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the multiplication

        Example:
            >>> c = a * b  # Element-wise multiplication
            >>> d = a.__mul__(b, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(TensorOp.MUL, [self, other], layout)

    def sum(self, dim_idx, layout=None):
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

    def product(self, dim_idx, layout=None):
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

    def matmul(self, other, layout=None):
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

    def __matmul__(self, other, layout=None):
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

    def block_matmul(self, other, layout=None):
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

    def transpose(self, layout=None):
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

    def poly_call(self, name, upper_bound, lower_bound, layout=None):
        """Call a polynomial approximation.

        Args:
            name (str): The name of the polynomial approximation or named function
            upper_bound (float): The upper bound of the polynomial approximation
            lower_bound (float): The lower bound of the polynomial approximation
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the Poly call operation

        Example:
            >>> b = a.poly_call("relu", 20, -20)  # Call ReLU with upper bound 20 and lower bound -20
            >>> c = a.poly_call("relu", 20, -20, layout="[0:4:1][1:4:1]")  # With layout
        """
        return TensorTerm(
            TensorOp.POLY_CALL, [self, name, upper_bound, lower_bound], layout
        )

    def poly(self, func=None, layout=None):
        """Apply a polynomial approximation or named function element-wise.

        Args:
            func: Optional. One of:
                - None or "identity": output = input
                - callable: any Python function (x) -> array, same shape. Used for
                  eval only; not serializable.
                - "silu": SiLU (Swish) x * sigmoid(x), or a polynomial approximation
                - "relu": ReLU(x) ≈ p(x) where p is a fixed-degree Chebyshev
                  approximation on [-20, 20], encoded as monomial coefficients
                - ("batchnorm", mean_key, var_key, gamma_key, beta_key): BatchNorm
                  with parameter names for inputs dict; eps=1e-5 used internally
                - list or tuple of floats [c0, c1, c2, ...]: polynomial
                  c0 + c1*x + c2*x^2 + ... applied element-wise
            layout (str, optional): Tensor layout string for the result

        Returns:
            TensorTerm: A new tensor term representing the Poly operation

        Example:
            >>> b = a.poly()  # identity
            >>> c = a.poly("silu")
            >>> d = a.poly("relu")  # approximate ReLU via Chebyshev polynomial
            >>> d = a.poly([0, 1, 0.1])  # x + 0.1*x^2
            >>> e = x.poly(("batchnorm", "mean", "var", "gamma", "beta"))
            >>> f = x.poly(lambda t: gamma * (t - mean) / np.sqrt(var + 1e-5) + beta)
        """
        if func is None:
            func = "identity"
        # Map named activations to concrete polynomial coefficient lists so that
        # downstream passes only see a generic Poly with explicit coefficients.
        # Special strings (e.g. "relu_exact") are left as-is so backends and
        # eval() can apply a non-polynomial function when appropriate.
        if isinstance(func, str):
            if func == "relu":
                func = APPROX_RELU_CHEBYSHEV_COEFFS
        return TensorTerm(TensorOp.POLY, [self, func], layout)

    def reshape(self, dim, shape, layout=None):
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

    def __getitem__(self, item, layout=None):
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
    def T(self):
        """Transpose property.

        Returns:
            TensorTerm: A new tensor term representing the transpose

        Example:
            >>> b = a.T  # Transpose of a
        """
        return TensorTerm(TensorOp.TRANSPOSE, [self])

    def rescale(self, scale_exp, layout=None):
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
    def conv2d(a, b, stride, padding, layout=None):
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
    def batchnorm(x, mean_key, var_key, gamma_key, beta_key, eps=1e-5):
        """BatchNorm via Poly: gamma * (x - mean) / sqrt(var + eps) + beta.

        Parameters are looked up from the inputs dict by key. Use poly() with
        func=("batchnorm", mean_key, var_key, gamma_key, beta_key[, eps]) for eval.

        Args:
            x (TensorTerm): Input tensor
            mean_key (str): Key for mean in inputs
            var_key (str): Key for variance in inputs
            gamma_key (str): Key for gamma (scale) in inputs
            beta_key (str): Key for beta (shift) in inputs
            eps (float): Small constant for numerical stability (default 1e-5)

        Returns:
            TensorTerm: Poly term that evaluates to BatchNorm(x) when eval(inputs) is called
        """
        func = ("batchnorm", mean_key, var_key, gamma_key, beta_key, eps)
        return TensorTerm(TensorOp.POLY, [x, func])

    def round_to_ceiling_power_of_2(self, n):
        """Round a number up to the next power of 2.

        Args:
            n (int): The number to round up

        Returns:
            int: The smallest power of 2 greater than or equal to n

        Raises:
            ValueError: If n is not positive

        Example:
            >>> round_to_ceiling_power_of_2(5)  # Returns 8
            >>> round_to_ceiling_power_of_2(8)  # Returns 8
        """
        if n <= 0:
            raise ValueError("Input must be a positive number.")
        return 1 if n == 1 else 2 ** math.ceil(math.log2(n))

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

    def eval_conv2d(self, input_tensor, filter_tensor, stride, padding):
        """Evaluate a 2D convolution operation.

        Args:
            input_tensor (numpy.ndarray): Input tensor with shape [C, H, W]
            filter_tensor (numpy.ndarray): Filter tensor with shape [C_out, C_in, H_f, W_f]
            stride (int): Stride of the convolution
            padding (str): Padding mode ("valid" or "same")

        Returns:
            numpy.ndarray: Output tensor after convolution

        Note:
            This is a reference implementation for testing purposes.
            The actual HE implementation will be generated by the backend.
        """
        input_shape = input_tensor.shape
        filter_shape = filter_tensor.shape
        if padding == "valid":
            h_o = (input_shape[1] - filter_shape[2]) // stride + 1
            w_o = (input_shape[2] - filter_shape[3]) // stride + 1
            output_shape = [filter_shape[0], h_o, w_o]
        elif padding == "same":
            # Same padding: stride 1 -> output size = input size; stride > 1 -> ceil(H/stride) x ceil(W/stride)
            if stride == 1:
                output_shape = [filter_shape[0], input_shape[1], input_shape[2]]
            else:
                h_o = (input_shape[1] + stride - 1) // stride
                w_o = (input_shape[2] + stride - 1) // stride
                output_shape = [filter_shape[0], h_o, w_o]

        # pad indices based on padding
        if padding == "valid":
            pass
        elif padding == "same":
            # Padding so that strided conv yields exactly h_o x w_o output
            pad_top = max(
                0,
                math.floor(
                    (stride * (output_shape[1] - 1) - input_shape[1] + filter_shape[2])
                    / 2
                ),
            )
            pad_bot = max(
                0,
                math.ceil(
                    (stride * (output_shape[1] - 1) - input_shape[1] + filter_shape[2])
                    / 2
                ),
            )
            pad_left = max(
                0,
                math.floor(
                    (stride * (output_shape[2] - 1) - input_shape[2] + filter_shape[3])
                    / 2
                ),
            )
            pad_right = max(
                0,
                math.ceil(
                    (stride * (output_shape[2] - 1) - input_shape[2] + filter_shape[3])
                    / 2
                ),
            )

            padded_input_tensor = []
            for channel_tensor in input_tensor:
                padded_input_tensor.append(
                    np.pad(
                        channel_tensor,
                        pad_width=((pad_top, pad_bot), (pad_left, pad_right)),
                        mode="constant",
                        constant_values=0,
                    )
                )
            input_tensor = padded_input_tensor

        output_tensor = np.zeros(output_shape)
        for in_c in range(input_shape[0]):
            for out_c in range(output_shape[0]):
                for i in range(output_shape[1]):
                    for j in range(output_shape[2]):
                        # Define the slice of the input tensor
                        i_start = i * stride
                        j_start = j * stride
                        i_end = i_start + filter_shape[2]
                        j_end = j_start + filter_shape[3]

                        # Extract the patch and compute the dot product
                        patch = []
                        for x in range(i_start, i_end):
                            row = []
                            for y in range(j_start, j_end):
                                row.append(input_tensor[in_c][x][y])
                            patch.append(row)
                        patch = np.array(patch)
                        # Multiply patch (from input channel in_c) by filter for that channel.
                        # If filter has fewer input channels, broadcast (use last filter channel).
                        f_in_idx = min(in_c, filter_shape[1] - 1)
                        output_tensor[out_c][i][j] += np.sum(
                            patch * filter_tensor[out_c][f_in_idx]
                        )

        return output_tensor

    def eval_helper(self, env, inputs):
        """Helper method for evaluating tensor terms.

        This method handles the evaluation of individual tensor operations
        during the computation process. It's called by the main eval method
        for each term in post-order.

        Args:
            env (dict): Environment mapping terms to their computed values
            inputs (dict): Dictionary mapping input tensor names to numpy arrays

        Returns:
            numpy.ndarray: The computed value for this term

        Raises:
            NotImplementedError: If the operation is not implemented
        """
        match self.op:
            case TensorOp.TENSOR:
                shape = inputs[self.cs[0]].shape
                rounded_shape = [self.round_to_ceiling_power_of_2(s) for s in shape]
                padding = [0] * len(shape)
                for i, (a, b) in enumerate(zip(shape, rounded_shape)):
                    padding[i] = b - a

                if len(padding) == 1:
                    padded_tensor = np.pad(
                        inputs[self.cs[0]],
                        pad_width=((0, padding[0])),
                        mode="constant",
                        constant_values=0,
                    )
                elif len(padding) == 2:
                    padded_tensor = np.pad(
                        inputs[self.cs[0]],
                        pad_width=((0, padding[0]), (0, padding[1])),
                        mode="constant",
                        constant_values=0,
                    )
                else:
                    # HACK don't pad for now
                    return np.array(inputs[self.cs[0]])
                return np.array(padded_tensor)
            case TensorOp.ADD:
                return env[self.cs[0]] + env[self.cs[1]]
            case TensorOp.SUB:
                return env[self.cs[0]] - env[self.cs[1]]
            case TensorOp.MUL:
                return env[self.cs[0]] * env[self.cs[1]]
            case TensorOp.SUM:
                return np.sum(env[self.cs[0]], axis=self.cs[1], keepdims=False)
            case TensorOp.MATMUL:
                return env[self.cs[0]] @ env[self.cs[1]]
            case TensorOp.TRANSPOSE:
                return env[self.cs[0]].T
            case TensorOp.CONV2D:
                return self.eval_conv2d(
                    env[self.cs[0]], env[self.cs[1]], self.cs[2], self.cs[3]
                )
            case TensorOp.CONST:
                return self.cs[0]
            case TensorOp.INDEX:
                # Support Python-style indexing and slicing.
                #
                # `self.cs[1]` is whatever was passed to `__getitem__`, which can be:
                # - int
                # - slice
                # - tuple of (int | slice | Ellipsis | None)
                # Additionally, for some serialization / interop use-cases, we accept:
                # - list of [start, stop] or [start, stop, step] (interpreted as slice)
                # - dict {"start": .., "stop": .., "step": ..} (interpreted as slice)
                def _to_slice(obj):
                    if isinstance(obj, slice) or obj is Ellipsis or obj is None:
                        return obj
                    # Allow list-based slice specs: [start, stop] or [start, stop, step]
                    if isinstance(obj, list) and len(obj) in (2, 3):
                        return slice(*obj)
                    # Allow dict-based slice specs
                    if isinstance(obj, dict) and ("start" in obj or "stop" in obj):
                        return slice(obj.get("start"), obj.get("stop"), obj.get("step"))
                    return obj

                item = self.cs[1]
                if isinstance(item, tuple):
                    item = tuple(_to_slice(x) for x in item)
                else:
                    item = _to_slice(item)

                return env[self.cs[0]][item]
            case TensorOp.RESHAPE:
                tensor = env[self.cs[0]]
                shape = {}
                for i, s in enumerate(tensor.shape):
                    shape[i] = s
                del shape[self.cs[1]]  # drop the dimension we're reshaping
                for k, v in self.cs[2].items():
                    shape[k] = self.round_to_ceiling_power_of_2(v)
                shape_list = [shape[k] for k in sorted(shape.keys())]
                return tensor.reshape(shape_list)
            case TensorOp.PERMUTE:
                tensor = env[self.cs[0]]
                return np.moveaxis(tensor, self.cs[1].keys(), self.cs[1].values())
            case TensorOp.RESCALE:
                scale_value = 2 ** self.cs[1]
                return env[self.cs[0]] / scale_value
            case TensorOp.POLY:
                x = env[self.cs[0]]
                func = self.cs[1] if len(self.cs) > 1 else "identity"
                if callable(func):
                    return np.asarray(func(x))
                if func == "identity":
                    return x
                if func == "relu_exact":
                    return np.maximum(x, 0.0)
                if func == "silu":
                    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
                if (
                    isinstance(func, tuple)
                    and len(func) >= 5
                    and func[0] == "batchnorm"
                ):
                    _, mean_key, var_key, gamma_key, beta_key = func[:5]
                    eps = float(func[5]) if len(func) > 5 else 1e-5
                    mean = np.asarray(inputs[mean_key])
                    var = np.asarray(inputs[var_key])
                    gamma = np.asarray(inputs[gamma_key])
                    beta = np.asarray(inputs[beta_key])
                    # Infer channel dim: last for 2D (N,C), first for 3D+ (C,H,W)
                    if x.ndim >= 3:
                        ch_dim = x.shape[0]
                        bc_shape = (ch_dim,) + (1,) * (x.ndim - 1)
                    elif x.ndim == 2:
                        ch_dim = x.shape[-1]
                        bc_shape = (1, ch_dim)
                    else:
                        ch_dim = mean.size
                        bc_shape = None
                    if ch_dim > mean.size:
                        pad_len = ch_dim - mean.size
                        mean = np.concatenate(
                            [mean, np.zeros(pad_len, dtype=mean.dtype)]
                        )
                        var = np.concatenate([var, np.ones(pad_len, dtype=var.dtype)])
                        gamma = np.concatenate(
                            [gamma, np.ones(pad_len, dtype=gamma.dtype)]
                        )
                        beta = np.concatenate(
                            [beta, np.zeros(pad_len, dtype=beta.dtype)]
                        )
                    if bc_shape is not None:
                        mean = mean.reshape(bc_shape)
                        var = var.reshape(bc_shape)
                        gamma = gamma.reshape(bc_shape)
                        beta = beta.reshape(bc_shape)
                    inv_std = 1.0 / np.sqrt(var + eps)
                    return gamma * (x - mean) * inv_std + beta
                if isinstance(func, (list, tuple)) and len(func) > 0:
                    try:
                        coeffs = [float(c) for c in func]
                    except (TypeError, ValueError):
                        coeffs = None
                    if coeffs is not None:
                        out = np.zeros_like(x, dtype=np.float64)
                        for i, c in enumerate(coeffs):
                            out = out + c * (x.astype(np.float64) ** i)
                        return out
                raise NotImplementedError(
                    f"Poly func {func!r} not implemented for eval"
                )
            case TensorOp.POLY_CALL:
                x = env[self.cs[0]]
                func = self.cs[1] if len(self.cs) > 1 else "identity"
                if func == "identity":
                    return x
                if func == "relu_exact":
                    return np.maximum(x, 0.0)
                if func == "relu":
                    return np.maximum(x, 0.0)
                if func == "silu":
                    return x * (1.0 / (1.0 + np.exp(-np.clip(x, -20, 20))))
                raise NotImplementedError(
                    f"Poly call func {func!r} not implemented for eval"
                )
            case _:
                raise NotImplementedError(self.op)

    def eval(self, inputs):
        """Evaluates the tensor computation represented by this term.

        Performs a post-order traversal of the computation DAG and evaluates each term
        using the provided input values. The evaluation maintains an environment mapping
        terms to their computed values.

        Args:
            inputs (dict): Dictionary mapping input tensor names to their numpy array values

        Returns:
            numpy.ndarray: The result of evaluating the full tensor computation

        Example:
            >>> import numpy as np
            >>> a = TensorTerm.Tensor("a", [64, 64], True)
            >>> b = TensorTerm.Tensor("b", [64], False)
            >>> c = a @ b
            >>>
            >>> inputs = {"a": np.random.rand(64, 64), "b": np.random.rand(64)}
            >>> result = c.eval(inputs)
        """
        env = {}
        for term in self.post_order():
            env[term] = term.eval_helper(env, inputs)
        return env[term]
