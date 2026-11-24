"""
Layout generation for unary operators, read from their operator spec.

A permute, transpose, reshape, sum, product or index takes each piece of the
input layout to a place in the output layout that its spec decides
(``assignment.alignment``):

- a label the output keeps: the piece moves to the output's dim for it
- a label the output drops (a sum, or a dim an index fixes): the piece
  becomes a gap
- a label a split divides, ``d = inner*d.0 + d.1``: the piece goes to the
  outer or the inner dim, or is cut in two where it crosses ``inner``
- a label a slice reads, ``d = step*d_out + start``: the piece holds the
  output dim, shrunk to the output's extent

Key functions:
- output_pieces: each input piece as the output holds it
- unary_layout: the output layout of one input layout
- gen_unary: the kernels of a unary term
"""

from copy import deepcopy as copy

from assignment.alignment import spec_for_term
from assignment.gen.gen_compaction import find_compaction
from frontends.tensor import TensorOp
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.layout_utils import dimension_merging
from ir.roll import Roll
from util.shape_util import get_term_shape
from util.util import round_to_ceiling_power_of_2

KERNEL_OPS = {
    TensorOp.TRANSPOSE: KernelOp.TRANSPOSE,
    TensorOp.PERMUTE: KernelOp.PERMUTE,
    TensorOp.RESHAPE: KernelOp.RESHAPE,
    TensorOp.SUM: KernelOp.SUM,
    TensorOp.PRODUCT: KernelOp.PRODUCT,
    TensorOp.INDEX: KernelOp.INDEX,
}


def _gap(piece):
    return Dim(None, piece.extent, piece.stride, DimType.EMPTY)


def _split(piece, relation, spec):
    """A piece of a dim that ``d = inner*outer + offset`` divides."""
    outer = spec.out.index(relation.base)
    offset = spec.out.index(relation.offset)
    inner = round_to_ceiling_power_of_2(relation.stride)
    if piece.stride * piece.extent <= inner:
        return [Dim(offset, piece.extent, piece.stride, piece.dim_type)]
    if inner != relation.stride:
        # The layout pads the inner dim, so the outer index is not a digit.
        return None
    if piece.stride >= inner:
        return [Dim(outer, piece.extent, piece.stride // inner, piece.dim_type)]
    high = piece.stride * piece.extent // inner
    return [
        Dim(outer, high, 1, piece.dim_type),
        Dim(offset, piece.extent // high, piece.stride, piece.dim_type),
    ]


def _sliced(dims, relation, spec, out_shape):
    """The pieces of a dim a slice reads, shrunk to the output's extent.

    The output dim is dense, so the pieces keep their strides and the
    coarsest ones give up the extent the slice removes.
    """
    dim = spec.lhs.index(relation.label)
    out_dim = spec.out.index(relation.base)
    pieces = [piece for piece in dims if piece.dim == dim]
    held = 1
    for piece in pieces:
        held *= piece.extent
    shrink = held // round_to_ceiling_power_of_2(out_shape[out_dim])
    extents = {}
    for piece in sorted(pieces, key=lambda piece: -piece.stride):
        cut = min(max(shrink, 1), piece.extent)
        extents[id(piece)] = piece.extent // cut
        shrink //= cut
    return {
        id(piece): (
            [Dim(out_dim, extents[id(piece)], piece.stride, piece.dim_type)]
            if extents[id(piece)] > 1 or len(pieces) == 1
            else []
        )
        for piece in pieces
    }


def output_pieces(spec, dims, out_shape):
    """Each input piece as the output holds it.

    Args:
        spec: the unary operator's spec
        dims: the input layout's pieces
        out_shape: the output's shape, for the extent a slice leaves

    Returns:
        A list parallel to `dims` of the output pieces that take each input
        piece's place: one piece, two where a split cuts it, a gap where the
        output drops its dim. None if the layout cannot carry the operator.
    """
    sliced = {}
    for relation in spec.relations:
        if relation.base is not None and relation.offset is None:
            sliced.update(_sliced(dims, relation, spec, out_shape))

    pieces = []
    for piece in dims:
        if piece.dim is None:
            pieces.append([copy(piece)])
            continue
        if piece.dim >= len(spec.lhs):
            return None
        label = spec.lhs[piece.dim]
        relation = spec.relation_of(label)
        if label in spec.out:
            held = [
                Dim(spec.out.index(label), piece.extent, piece.stride, piece.dim_type)
            ]
        elif relation is None or relation.base is None:
            # summed away, or fixed by an index
            held = [_gap(piece)]
        elif relation.offset is not None:
            held = _split(piece, relation, spec)
        else:
            held = sliced[id(piece)]
        if held is None:
            return None
        pieces.append(held)
    return pieces


def _output_rolls(rolls, dims, pieces):
    """The input's rolls on the output's pieces, or None if one cannot be kept."""
    out_rolls = []
    for roll in rolls:
        rolled, rolled_by = (pieces[i] for i in roll.roll_index(dims))
        if len(rolled) != 1 or len(rolled_by) != 1:
            return None
        rolled, rolled_by = rolled[0], rolled_by[0]
        if rolled.dim is None:
            # Every element of a dropped dim is read, in whatever order.
            continue
        if rolled_by.dim is None:
            # The kept dim is shifted by the dropped one: its elements at one
            # position come from different indices.
            return None
        out_rolls.append(Roll(rolled, rolled_by))
    return out_rolls


def unary_layout(spec, term, layout, out_shape):
    """The output layout of a unary operator on one input layout.

    Returns:
        (layout, dropped): `dropped` holds the input pieces whose dim the
        output drops. None if the input layout cannot carry the operator.
    """
    dims = layout.get_dims()
    moves_data = any(relation.offset is None for relation in spec.relations)
    if moves_data and layout.rolls:
        # An index or slice is lowered slot by slot from unrolled layouts.
        return None
    pieces = output_pieces(spec, dims, out_shape)
    if pieces is None:
        return None
    rolls = _output_rolls(layout.rolls, dims, pieces)
    if rolls is None:
        return None
    dropped = [
        copy(piece)
        for piece, held in zip(dims, pieces)
        if piece.dim is not None and len(held) == 1 and held[0].dim is None
    ]
    out_dims = [piece for held in pieces for piece in held]
    out_layout = dimension_merging(
        Layout(term, rolls, out_dims, layout.n, layout.secret)
    )
    return out_layout, dropped


def gen_unary(term, kernels):
    """Generates the kernels of a unary term, one per input layout.

    Args:
        term: a TRANSPOSE, PERMUTE, RESHAPE, SUM, PRODUCT or INDEX term
        kernels: the input's candidate kernels

    Returns:
        Set of kernels. A SUM or PRODUCT kernel also carries the input pieces
        it reduces; an INDEX kernel is compacted.
    """
    op = KERNEL_OPS[term.op]
    spec = spec_for_term(term, [get_term_shape(term.cs[0])])
    out_shape = get_term_shape(term)

    output_kernels = set()
    for kernel in kernels:
        made = unary_layout(spec, term, kernel.layout, out_shape)
        if made is None:
            continue
        layout, dropped = made
        cs = [Kernel(KernelOp.CS, [0], kernel.layout)]
        if op in (KernelOp.SUM, KernelOp.PRODUCT):
            cs.append(tuple(dropped))
        out_kernel = Kernel(op, cs, layout)
        if op == KernelOp.INDEX:
            out_kernel = find_compaction(out_kernel)
        output_kernels.add(out_kernel)
    return output_kernels
