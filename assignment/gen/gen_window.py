"""
Layout generation shared by windowed operators (conv2d, conv3d, avg_pool2d).

The roles of every dim come from the operator's spec
(``assignment.alignment.conv_spec``), not from dim positions:

- a filter label the input does not hold (the output channel, the taps) is
  replicated onto the input
- a label both hold (the input channel) aligns the two layouts
- a label the output drops (the input channel, the taps) leaves a gap
- a label defined by a window relation ``h = stride * h_out + kh`` becomes its
  output dim, and a stride above 1 frees the positions between outputs

Key functions:
- replication_dims: the replication the input needs, and the filter dims it faces
- filter_dims: the filter layout that faces a replicated input layout
- output_dims: the output layout of a replicated input and its filter
"""

from ir.dim import Dim, DimType
from util.util import prod


def replication_dims(spec, a_shape, b_shape):
    """Replication pieces for the filter labels the input does not hold.

    Returns:
        (replicated_dims, b_dims): the pieces to prepend to the input layout,
        outermost first, and the filter pieces that face them
    """
    replicated_dims = []
    b_dims = []
    for b_dim, label in enumerate(spec.rhs):
        if label in spec.lhs or b_shape[b_dim] <= 1:
            continue
        relation = spec.relation_of(label)
        if relation is None:
            # Not a tap: one copy of the whole input per index.
            stride = prod(a_shape)
        else:
            # A tap strides over the input axes inside its window's axis.
            stride = prod(a_shape[spec.lhs.index(relation.label) + 1 :])
        replicated_dims.append(Dim(None, b_shape[b_dim], int(stride)))
        b_dims.append(Dim(b_dim, b_shape[b_dim], 1))
    return replicated_dims, b_dims


def filter_dims(spec, a_dims, b_dims):
    """Extends `b_dims` with the filter pieces that face the input's own pieces.

    Returns:
        The filter's pieces, or None when the two layouts cannot face each
        other piece by piece.
    """
    b_dims = list(b_dims)
    for dim in a_dims:
        if dim.dim is not None:
            label = spec.lhs[dim.dim]
            if label in spec.rhs:
                b_dims.append(Dim(spec.rhs.index(label), dim.extent, dim.stride))
            else:
                b_dims.append(Dim(None, dim.extent, 1))
        elif dim.dim_type == DimType.EMPTY:
            b_dims.append(Dim(None, dim.extent, 1, DimType.EMPTY))
    if len(b_dims) != len(a_dims):
        return None
    return b_dims


def output_dims(spec, a_dims, b_dims, out_extents):
    """The output layout's pieces for a replicated input and its filter.

    Args:
        spec: the operator's spec
        a_dims, b_dims: the input's and the filter's pieces, facing each other
        out_extents: the extent of each window output label

    Returns:
        The output pieces, or None when a windowed dim is split across pieces.
    """
    lhs_out = spec.lhs_out_dims()
    rhs_out = spec.rhs_out_dims()
    out = []
    windows_done = set()
    for a_dim, b_dim in zip(a_dims, b_dims):
        a_summed = a_dim.dim is not None and lhs_out[a_dim.dim] is None
        b_summed = b_dim.dim is not None and rhs_out[b_dim.dim] is None
        if a_dim.dim_type == DimType.EMPTY or a_summed or b_summed:
            out.append(Dim(None, a_dim.extent, 1, DimType.EMPTY))
        elif a_dim.dim is None and b_dim.dim is not None:
            # A label only the filter holds and the output keeps.
            out.append(Dim(rhs_out[b_dim.dim], b_dim.extent, b_dim.stride))
        elif (
            a_dim.dim is not None
            and a_dim.dim_type == DimType.FILL
            and b_dim.dim is None
            and spec.relation_of(spec.lhs[a_dim.dim]) is not None
        ):
            relation = spec.relation_of(spec.lhs[a_dim.dim])
            if relation.label in windows_done:
                return None
            windows_done.add(relation.label)
            out.append(Dim(lhs_out[a_dim.dim], out_extents[relation.base], 1))
            if relation.stride == 2:
                # The window keeps every other position; the rest are freed.
                out.append(Dim(None, relation.stride, 1, DimType.EMPTY))
            elif relation.stride != 1:
                raise NotImplementedError(f"stride not supported: {relation.stride}")
        else:
            dim = a_dim.dim if a_dim.dim is None else lhs_out[a_dim.dim]
            out.append(Dim(dim, a_dim.extent, 1, DimType.FILL))
    return out


def materialized_shape(layout):
    """The tensor shape a layout holds: the product of extents per dim."""
    shape = {}
    for dim in layout.get_dims():
        if dim.dim_type == DimType.EMPTY or dim.dim is None:
            continue
        shape[dim.dim] = shape.get(dim.dim, 1) * dim.extent
    if not shape:
        return []
    return [shape.get(i, 1) for i in range(max(shape) + 1)]
