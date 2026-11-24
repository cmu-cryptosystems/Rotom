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

from copy import deepcopy as copy

from frontends.tensor import TensorOp
from frontends.tensor_args import (
    Conv2dArgs,
    Conv3dArgs,
    ReshapeArgs,
    TensorPlaceholderArgs,
    window_axis,
)
from util.util import round_to_ceiling_power_of_2


def _avg_pool_out_hw(
    h_i: int, w_i: int, k: int, s: int, padding: str
) -> tuple[int, int]:
    return window_axis(h_i, k, s, padding)[0], window_axis(w_i, k, s, padding)[0]


class Shape:
    def __init__(self, comp):
        self.comp = comp
        self.shapes = {}
        self.padded_shapes = {}

    def _index_shape(self, base_shape, index_spec):
        """Compute the output shape of an INDEX operation.

        Args:
            base_shape (list[int]): Input tensor shape.
            index_spec: Indexing object from TensorTerm (int, slice, or tuple).

        Returns:
            list[int]: Resulting shape after applying the index.
        """

        def _slice_len(dim_len, slc: slice) -> int:
            start = 0 if slc.start is None else slc.start
            stop = dim_len if slc.stop is None else slc.stop
            step = 1 if slc.step is None else slc.step
            if step <= 0:
                raise NotImplementedError(
                    "Shape analysis for INDEX does not yet support non-positive slice steps"
                )
            if stop < start:
                return 0
            return (stop - start + step - 1) // step

        shape = list(base_shape)

        # Simple integer index: drop the first dimension
        if isinstance(index_spec, int):
            return shape[1:]

        # Single slice applied to the first dimension
        if isinstance(index_spec, slice):
            if not shape:
                return []
            new_dim = _slice_len(shape[0], index_spec)
            return [new_dim] + shape[1:]

        # Tuple of indices/slices: apply sequentially across dimensions
        if isinstance(index_spec, tuple):
            new_shape = []
            dim_i = 0
            for idx in index_spec:
                if isinstance(idx, int):
                    # Integer index removes this dimension
                    dim_i += 1
                elif isinstance(idx, slice):
                    if dim_i >= len(shape):
                        raise IndexError(
                            "INDEX spec has more dimensions than tensor rank"
                        )
                    dim_len = shape[dim_i]
                    new_dim = _slice_len(dim_len, idx)
                    new_shape.append(new_dim)
                    dim_i += 1
                else:
                    raise NotImplementedError(
                        f"Unsupported INDEX component type in shape analysis: {type(idx)}"
                    )

            # Any remaining dimensions are carried over unchanged
            if dim_i < len(shape):
                new_shape.extend(shape[dim_i:])
            return new_shape

        # Fallback for unsupported index types
        raise NotImplementedError(
            f"Unsupported INDEX spec in shape analysis: {type(index_spec)}"
        )

    def get_padded_shape(self, term):
        match term.op:
            case TensorOp.TENSOR:
                args = TensorPlaceholderArgs.from_term(term)
                return [round_to_ceiling_power_of_2(s) for s in args.shape]
            case TensorOp.CONST:
                return None
            case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                b_shape = copy(self.padded_shapes[term.cs[1]])
                if a_shape is None:
                    return b_shape
                if b_shape is None:
                    return a_shape
                if len(a_shape) > len(b_shape):
                    return a_shape
                else:
                    return b_shape
            case TensorOp.RESCALE | TensorOp.CAST:
                # Rescale / cast preserve the shape of the input tensor
                return copy(self.padded_shapes[term.cs[0]])
            case TensorOp.MATMUL:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                b_shape = copy(self.padded_shapes[term.cs[1]])

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
                    assert a_shape[-1] == b_shape[-2], (
                        f"MATMUL padded inner dim mismatch: left padded_shape={a_shape} "
                        f"(last={a_shape[-1]}), right padded_shape={b_shape} "
                        f"(second-to-last={b_shape[-2]}). "
                        f"Left get_shape={self.get_shape(term.cs[0])}, "
                        f"right get_shape={self.get_shape(term.cs[1])}."
                    )
                    c_shape = list(a_shape[:-1]) + list(b_shape[:-2]) + [b_shape[-1]]

                return c_shape
            case TensorOp.TRANSPOSE:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                assert len(a_shape) == 2
                a_shape[0], a_shape[1] = a_shape[1], a_shape[0]
                return a_shape
            case TensorOp.RESHAPE:
                args = ReshapeArgs.from_term(term)
                a_shape = copy(self.padded_shapes[args.input])
                dim_to_del = args.dim
                del a_shape[dim_to_del]
                return a_shape + [
                    round_to_ceiling_power_of_2(s) for s in args.shape.values()
                ]
            case TensorOp.CONV2D:
                args = Conv2dArgs.from_term(term)
                # Convolution semantics use the *logical* input spatial sizes; tensor eval
                # does not power-of-2 pad rank>2 tensors. We still round the output to a
                # power-of-2 for layout compatibility.
                a_shape = copy(self.padded_shapes[args.input])
                logical_b = self.get_shape(args.filter)
                logical_a = self.get_shape(args.input)

                _c_i = a_shape[0]
                h_i = logical_a[1]
                w_i = logical_a[2]

                # Logical output channels (match layout gen / eval; filter c_out is not p2-padded).
                c_o = args.out_channels(logical_b)
                if args.depthwise:
                    # The channel is carried, in the pieces the input holds it in.
                    c_o = _c_i
                # Use logical filter spatial sizes (declared shape), not power-of-2-padded filter.
                f_h, f_w = args.taps(logical_b)

                (h_o, _, _), (w_o, _, _) = args.window([c_o, h_i, w_i], logical_b)
                # Round to power of 2 for layout compatibility
                c_shape = [
                    c_o,
                    round_to_ceiling_power_of_2(h_o),
                    round_to_ceiling_power_of_2(w_o),
                ]
                return c_shape
            case TensorOp.CONV3D:
                args = Conv3dArgs.from_term(term)
                # Use logical input spatial sizes; then round the output spatial dims to p2.
                a_shape = copy(self.padded_shapes[args.input])
                b_shape = copy(self.padded_shapes[args.filter])
                logical_b = self.get_shape(args.filter)
                logical_a = self.get_shape(args.input)

                # Logical output channels (layout gen uses unpadded filter c_out).
                c_o = args.out_channels(logical_b)
                if args.depthwise:
                    # The channel is carried, in the pieces the input holds it in.
                    c_o = a_shape[0]
                # Use logical input and filter spatial sizes, not padded ones.
                (d_o, _, _), (h_o, _, _), (w_o, _, _) = args.window(
                    logical_a, logical_b
                )

                return [
                    c_o,
                    round_to_ceiling_power_of_2(d_o),
                    round_to_ceiling_power_of_2(h_o),
                    round_to_ceiling_power_of_2(w_o),
                ]
            case TensorOp.INDEX:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                return self._index_shape(a_shape, term.cs[1])
            case TensorOp.PERMUTE:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                permuted_shape = [0] * len(a_shape)
                for k, v in term.cs[1].items():
                    permuted_shape[v] = a_shape[k]
                return permuted_shape
            case TensorOp.SUM | TensorOp.PRODUCT:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                dim_idx = term.cs[1]
                result_shape = a_shape[:dim_idx] + a_shape[dim_idx + 1 :]
                return result_shape
            case TensorOp.MEAN:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                axes = term.cs[1]
                if isinstance(axes, int):
                    axes = (axes,)
                out = copy(a_shape)
                for a in axes:
                    out[int(a)] = 1
                return out
            case TensorOp.POLY_CALL | TensorOp.HARD_SWISH:
                # Preserves the shape of the input tensor
                return copy(self.padded_shapes[term.cs[0]])
            case TensorOp.TILE:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                reps = term.cs[1]
                if len(reps) != len(a_shape):
                    raise ValueError(
                        "TILE reps rank must match input rank for padded shape"
                    )
                return [a_shape[i] * int(reps[i]) for i in range(len(a_shape))]
            case TensorOp.CONCAT:
                axis = int(term.cs[1])
                cs_shapes = [copy(self.padded_shapes[t]) for t in term.cs[0]]
                out = copy(cs_shapes[0])
                out[axis] = sum(s[axis] for s in cs_shapes)
                return out
            case TensorOp.CUMSUM:
                return copy(self.padded_shapes[term.cs[0]])
            case TensorOp.AVG_POOL2D:
                a_shape = copy(self.padded_shapes[term.cs[0]])
                k, s, pad = int(term.cs[1]), int(term.cs[2]), term.cs[3]
                h_o, w_o = _avg_pool_out_hw(
                    int(a_shape[1]), int(a_shape[2]), k, s, str(pad)
                )
                return [
                    int(a_shape[0]),
                    round_to_ceiling_power_of_2(h_o),
                    round_to_ceiling_power_of_2(w_o),
                ]
            case _:
                raise NotImplementedError(term.op)

    def get_shape(self, term):
        match term.op:
            case TensorOp.TENSOR:
                return TensorPlaceholderArgs.from_term(term).shape
            case TensorOp.CONST:
                return None
            case TensorOp.ADD | TensorOp.MUL | TensorOp.SUB:
                a = term.cs[0]
                b = term.cs[1]
                a_shape = copy(self.get_shape(a))
                b_shape = copy(self.get_shape(b))
                if a_shape is None:
                    return b_shape
                if b_shape is None:
                    return a_shape
                if len(a_shape) > len(b_shape):
                    return a_shape
                else:
                    return b_shape
            case TensorOp.RESCALE | TensorOp.CAST:
                # Rescale / cast preserve the shape of the input tensor
                return copy(self.get_shape(term.cs[0]))
            case TensorOp.MATMUL:
                a = term.cs[0]
                b = term.cs[1]
                a_shape = copy(self.get_shape(a))
                b_shape = copy(self.get_shape(b))

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
                a_shape = copy(self.get_shape(a))
                assert len(a_shape) == 2
                return [a_shape[1], a_shape[0]]
            case TensorOp.CONV2D:
                args = Conv2dArgs.from_term(term)
                a = args.input
                b = args.filter
                a_shape = copy(self.get_shape(a))
                b_shape = copy(self.get_shape(b))

                c_o = args.out_channels(b_shape)
                f_h, f_w = args.taps(b_shape)
                h_i = a_shape[1]
                w_i = a_shape[2]

                (h_o, _, _), (w_o, _, _) = args.window([c_o, h_i, w_i], b_shape)
                c_shape = [c_o, h_o, w_o]
                return c_shape
            case TensorOp.CONV3D:
                args = Conv3dArgs.from_term(term)
                a = args.input
                b = args.filter
                a_shape = copy(self.get_shape(a))
                b_shape = copy(self.get_shape(b))

                c_o = args.out_channels(b_shape)
                (d_o, _, _), (h_o, _, _), (w_o, _, _) = args.window(a_shape, b_shape)

                return [c_o, d_o, h_o, w_o]
            case TensorOp.INDEX:
                a = term.cs[0]
                a_shape = copy(self.get_shape(a))
                return self._index_shape(a_shape, term.cs[1])
            case TensorOp.RESHAPE:
                args = ReshapeArgs.from_term(term)
                a = args.input
                a_shape = copy(self.get_shape(a))
                shape_map = {}
                for i, shape in enumerate(a_shape):
                    shape_map[i] = shape
                dim_to_del = args.dim
                del shape_map[dim_to_del]
                for k, v in args.shape.items():
                    shape_map[k] = v
                new_shape = [shape_map[k] for k in sorted(shape_map.keys())]
                return new_shape
            case TensorOp.PERMUTE:
                a = term.cs[0]
                a_shape = copy(self.get_shape(a))
                shape_map = {}
                for i, shape in enumerate(a_shape):
                    shape_map[i] = shape
                new_shape = [0] * len(shape_map)
                for i in range(len(shape_map)):
                    new_shape[term.cs[1][i]] = shape_map[i]
                return new_shape
            case TensorOp.SUM | TensorOp.PRODUCT:
                a = term.cs[0]
                a_shape = copy(self.get_shape(a))
                dim_idx = term.cs[1]
                result_shape = a_shape[:dim_idx] + a_shape[dim_idx + 1 :]
                return result_shape
            case TensorOp.MEAN:
                a_shape = copy(self.get_shape(term.cs[0]))
                axes = term.cs[1]
                if isinstance(axes, int):
                    axes = (axes,)
                out = copy(a_shape)
                for ax in axes:
                    out[int(ax)] = 1
                return out
            case TensorOp.POLY_CALL | TensorOp.HARD_SWISH:
                return copy(self.get_shape(term.cs[0]))
            case TensorOp.TILE:
                a_shape = copy(self.get_shape(term.cs[0]))
                reps = term.cs[1]
                if len(reps) != len(a_shape):
                    raise ValueError("TILE reps rank must match input rank")
                return [a_shape[i] * int(reps[i]) for i in range(len(a_shape))]
            case TensorOp.CONCAT:
                axis = int(term.cs[1])
                cs_shapes = [copy(self.get_shape(t)) for t in term.cs[0]]
                out = copy(cs_shapes[0])
                out[axis] = sum(s[axis] for s in cs_shapes)
                return out
            case TensorOp.CUMSUM:
                return copy(self.get_shape(term.cs[0]))
            case TensorOp.AVG_POOL2D:
                a_shape = copy(self.get_shape(term.cs[0]))
                k, s, pad = int(term.cs[1]), int(term.cs[2]), term.cs[3]
                h_o, w_o = _avg_pool_out_hw(
                    int(a_shape[1]), int(a_shape[2]), k, s, str(pad)
                )
                return [int(a_shape[0]), h_o, w_o]
            case _:
                raise NotImplementedError(term.op)

    def run(self):
        for term in self.comp.post_order():
            self.shapes[term] = self.get_shape(term)
            self.padded_shapes[term] = self.get_padded_shape(term)
