"""
Alignment utilities for tensor operations.

This module provides functions for determining dimension alignment between
tensor operands with different layouts. Proper alignment is crucial for
ensuring that tensor operations can be performed efficiently and deterministically
in the homomorphic encryption domain.

Key functions:
- get_dim_alignment: Determines how dimensions should be aligned for operations
"""

from frontends.tensor import TensorOp


def get_dim_alignment(term, shapes):
    """Determines dimension alignment for tensor operations.

    This function analyzes a tensor operation and its input shapes to determine
    how dimensions should be aligned between operands. The alignment is crucial
    for ensuring that operations can be performed correctly in the HE domain.

    For different operation types:
    - Element-wise ops (ADD, SUB, MUL): Aligns corresponding dimensions
    - Matrix multiplication: Aligns contraction dimensions and batch dimensions
    - Block matrix multiplication: Aligns block dimensions

    Args:
        term: TensorTerm representing the operation
        shapes: List of input tensor shapes

    Returns:
        Set of tuples (dim_a, dim_b) representing dimension alignments.
        None values indicate dimensions that don't have a corresponding
        dimension in the other operand.

    Raises:
        NotImplementedError: If the operation type alignment is not implemented
    """
    alignment = set()
    match term.op:
        case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
            a_shape = shapes[0]
            b_shape = shapes[1]
            a_dims = list(range(len(a_shape)))[::-1]
            b_dims = list(range(len(b_shape)))[::-1]
            for i in range(min(len(a_dims), len(b_dims))):
                alignment.add((a_dims[i], b_dims[i]))
            if len(a_dims) > len(b_dims):
                for i in range(len(b_dims), len(a_dims)):
                    alignment.add((a_dims[i], None))
            else:
                for i in range(len(a_dims), len(b_dims)):
                    alignment.add((None, b_dims[i]))
            return alignment
        case TensorOp.MATMUL:
            a_shape = shapes[0]
            b_shape = shapes[1]
            a_dims = list(range(len(a_shape)))
            b_dims = list(range(len(b_shape)))

            # align dimensions
            if len(a_shape) == 2 and len(b_shape) == 1:
                alignment.add((0, None))
                alignment.add((1, 0))
            elif len(a_shape) == 1 and len(b_shape) == 2:
                alignment.add((None, 0))
                alignment.add((0, 1))
            elif len(a_shape) == 2 and len(b_shape) == 2:
                alignment.add((0, None))
                alignment.add((1, 0))
                alignment.add((None, 1))
            elif len(a_shape) == 3 and len(b_shape) == 2:
                alignment.add((0, None))
                alignment.add((1, None))
                alignment.add((2, 0))
                alignment.add((None, 1))
            else:
                raise NotImplementedError("alignment not implemented")
            return alignment
        case TensorOp.BLOCK_MATMUL:
            alignment.add((0, 0))
            alignment.add((1, None))
            alignment.add((2, 1))
            alignment.add((None, 2))
            return alignment
        case _:
            raise NotImplementedError(f"alignment: {term.op}")
