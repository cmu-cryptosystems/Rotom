"""
Shape analysis for tensor computations.

This module provides shape analysis to determine the shapes of tensors
throughout a computation graph. It handles shape inference, padding
to power-of-2 sizes, and shape propagation through various operations.

Key Concepts:

- Shape inference: Determining output shapes from input shapes
- Power-of-2 padding: Padding dimensions to powers of 2 for HE efficiency
- Shape propagation: How shapes flow through operations
- Dimension compatibility: Ensuring compatible shapes for operations
"""

from frontends.tensor import TensorOp
from util.util import round_to_ceiling_power_of_2, prod
from copy import deepcopy as copy

class Shape:
    def __init__(self, comp):
        self.comp = comp
        self.shapes = {}
        self.padded_shapes = {}

    def get_padded_shape(self, term):
        match term.op:
            case TensorOp.TENSOR:
                return [round_to_ceiling_power_of_2(s) for s in term.cs[1]]
            case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
                a_shape = self.padded_shapes[term.cs[0]]
                b_shape = self.padded_shapes[term.cs[1]]
                if len(a_shape) > len(b_shape):
                    return a_shape
                else:
                    return b_shape
            case TensorOp.MATMUL:
                a_shape = self.padded_shapes[term.cs[0]]
                b_shape = self.padded_shapes[term.cs[1]]

                # Result shape: all dims of A except last + all dims of B except second-to-last
                if len(a_shape) == 0 or len(b_shape) == 0:
                    raise ValueError("Cannot perform matmul on scalar tensors")
                
                # For 1D tensors, treat as row vector (1, n) or column vector (n, 1)
                if len(a_shape) == 1 and len(b_shape) == 1:
                    # Vector dot product - result is scalar, but we represent as 1D with size 1
                    assert a_shape[0] == b_shape[0]
                    c_shape = [1]
                elif len(a_shape) == 1:
                    # 1D × ND: treat 1D as (1, n), result is (1, ...)
                    assert a_shape[0] == b_shape[-2]
                    c_shape = [1] + list(b_shape[:-2]) + [b_shape[-1]]
                elif len(b_shape) == 1:
                    # ND × 1D: treat 1D as (n, 1), result is (..., 1)
                    assert a_shape[-1] == b_shape[0]
                    c_shape = list(a_shape[:-1]) + [1]
                else:
                    # ND × ND: standard batched matmul
                    assert a_shape[-1] == b_shape[-2]
                    c_shape = list(a_shape[:-1]) + list(b_shape[:-2]) + [b_shape[-1]]
                
                return c_shape
            case TensorOp.BLOCK_MATMUL:
                a_shape = self.padded_shapes[term.cs[0]]
                b_shape = self.padded_shapes[term.cs[1]]
                assert len(a_shape) == 3 and len(b_shape) == 3
                assert a_shape[0] == b_shape[0]
                assert a_shape[2] == b_shape[1]
                return [a_shape[0], a_shape[1], b_shape[2]]
            case TensorOp.TRANSPOSE:
                a_shape = self.padded_shapes[term.cs[0]]
                assert len(a_shape) == 2
                a_shape[0], a_shape[1] = a_shape[1], a_shape[0]
                return a_shape
            case TensorOp.RESHAPE:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                del a_shape[term.cs[1]]
                return a_shape + [round_to_ceiling_power_of_2(s) for s in term.cs[2].values()]
            case TensorOp.CONV2D:
                a_shape = self.padded_shapes[term.cs[0]]
                b_shape = self.padded_shapes[term.cs[1]]

                c_i = a_shape[0]
                h_i = a_shape[1]
                w_i = a_shape[2]

                c_o = b_shape[0]
                f_h = b_shape[1]
                f_w = b_shape[2]

                stride = term.cs[2]
                padding = term.cs[3]

                if padding == "valid":
                    # does not add any extra pixels to the image
                    h_o = (h_i - f_h) // stride + 1
                    w_o = (w_i - f_w) // stride + 1
                elif padding == "same":
                    h_o = h_i
                    w_o = w_i
                else:
                    raise NotImplementedError(f"unknown padding: {padding}")
                c_shape = [c_o, h_o, w_o]
                return c_shape
            case TensorOp.INDEX:
                a_shape = self.padded_shapes[term.cs[0]]
                return a_shape[1:]
            case TensorOp.PERMUTE:
                a_shape = self.padded_shapes[term.cs[0]]
                permuted_shape = [0] * len(a_shape)
                for k, v in term.cs[1].items():
                    permuted_shape[v] = a_shape[k]
                return permuted_shape
            case _:
                raise NotImplementedError(term.op)

    def get_shape(self, term):
        match term.op:
            case TensorOp.TENSOR:
                return term.cs[1]
            case TensorOp.ADD:
                a = term.cs[0]
                b = term.cs[1]
                a_shape = self.get_shape(a)
                b_shape = self.get_shape(b)
                if len(a_shape) > len(b_shape):
                    return a_shape
                else:
                    return b_shape
            case TensorOp.MATMUL:
                a = term.cs[0]
                b = term.cs[1]
                a_shape = self.get_shape(a)
                b_shape = self.get_shape(b)

                # Result shape: all dims of A except last + all dims of B except second-to-last
                if len(a_shape) == 0 or len(b_shape) == 0:
                    raise ValueError("Cannot perform matmul on scalar tensors")
                
                # For 1D tensors, treat as row vector (1, n) or column vector (n, 1)
                if len(a_shape) == 1 and len(b_shape) == 1:
                    # Vector dot product - result is scalar, but we represent as 1D with size 1
                    assert a_shape[0] == b_shape[0]
                    c_shape = [1]
                elif len(a_shape) == 1:
                    # 1D × ND: treat 1D as (1, n), result is (1, ...)
                    assert a_shape[0] == b_shape[-2]
                    c_shape = [1] + list(b_shape[:-2]) + [b_shape[-1]]
                elif len(b_shape) == 1:
                    # ND × 1D: treat 1D as (n, 1), result is (..., 1)
                    assert a_shape[-1] == b_shape[0]
                    c_shape = list(a_shape[:-1]) + [1]
                else:
                    # ND × ND: standard batched matmul
                    assert a_shape[-1] == b_shape[-2]
                    c_shape = list(a_shape[:-1]) + list(b_shape[:-2]) + [b_shape[-1]]
                
                return c_shape
            case TensorOp.TRANSPOSE:
                a = term.cs[0]
                a_shape = self.get_shape(a)
                assert len(a_shape) == 2
                return [a_shape[1], a_shape[0]]
            case TensorOp.CONV2D:
                a = term.cs[0]
                b = term.cs[1]
                a_shape = self.get_shape(a)
                b_shape = self.get_shape(b)

                c_i = a_shape[0]
                h_i = a_shape[1]
                w_i = a_shape[2]

                c_o = b_shape[0]
                f_h = b_shape[1]
                f_w = b_shape[2]

                stride = term.cs[2]
                padding = term.cs[3]

                if padding == "valid":
                    # does not add any extra pixels to the image
                    h_o = (h_i - f_h) // stride + 1
                    w_o = (w_i - f_w) // stride + 1
                elif padding == "same":
                    h_o = h_i
                    w_o = w_i
                else:
                    raise NotImplementedError(f"unknown padding: {padding}")
                c_shape = [c_o, h_o, w_o]
                return c_shape
            case TensorOp.INDEX:
                a = term.cs[0]
                a_shape = self.get_shape(a)
                return a_shape[1:]
            case TensorOp.RESHAPE:
                a = term.cs[0]
                a_shape = self.get_shape(a)
                shape_map = {}
                for i, shape in enumerate(a_shape):
                    shape_map[i] = shape
                del shape_map[term.cs[1]]
                for k, v in term.cs[2].items():
                    shape_map[k] = v
                new_shape = []
                for i in range(len(shape_map)):
                    new_shape.append(shape_map[i])
                return new_shape
            case TensorOp.PERMUTE:
                a = term.cs[0]
                a_shape = self.get_shape(a)
                shape_map = {}
                for i, shape in enumerate(a_shape):
                    shape_map[i] = shape
                new_shape = [0] * len(shape_map)
                for i in range(len(shape_map)):
                    new_shape[term.cs[1][i]] = shape_map[i]
                return new_shape
            case TensorOp.BLOCK_MATMUL:
                a = term.cs[0]
                b = term.cs[1]
                a_shape = self.get_shape(a)
                b_shape = self.get_shape(b)
                assert len(a_shape) == 3 and len(b_shape) == 3
                assert a_shape[0] == b_shape[0]
                assert a_shape[2] == b_shape[1]
                return [a_shape[0], a_shape[1], b_shape[2]]
            case _:
                raise NotImplementedError(term.op)

    def run(self):
        for term in self.comp.post_order():
            self.shapes[term] = self.get_shape(term)
            self.padded_shapes[term] = self.get_padded_shape(term)
