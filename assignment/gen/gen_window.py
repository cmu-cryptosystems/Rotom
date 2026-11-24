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
- direct_dims: the intermediate and output layouts of a direct kernel
"""

import math

from ir.dim import Dim, DimType
from util.util import prod, round_to_ceiling_power_of_2


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


#: where a direct kernel puts the labels only the filter holds (the output
#: channel): on ciphertexts, in the positions a stride frees, or in every gap
FILLS = ("none", "strides", "gaps")


def _window_pieces(pieces, out_extent, stride):
    """How the pieces of a windowed dim hold its output dim.

    The dim shrinks from the product of `pieces` to `out_extent`: by its
    stride inside each output position, which frees the finest positions, and
    by what valid padding trims, which leaves a gap at the coarse end.

    Returns:
        One (gap, extent, stride, freed) per piece, in the order given: the
        piece becomes a gap, the output piece, then the freed positions. None
        when the pieces cannot be divided that way.
    """
    held = prod([piece.extent for piece in pieces])
    if held % out_extent:
        return None
    freed_left = math.gcd(int(held // out_extent), stride)
    out_left = out_extent
    out_stride = 1
    plan = {}
    for at in sorted(range(len(pieces)), key=lambda i: pieces[i].stride):
        extent = pieces[at].extent
        freed = math.gcd(extent, freed_left)
        freed_left //= freed
        extent //= freed
        take = math.gcd(extent, out_left)
        out_left //= take
        plan[at] = (extent // take, take, out_stride, freed)
        out_stride *= take
    if freed_left != 1 or out_left != 1:
        return None
    return [plan[at] for at in range(len(pieces))]


def direct_dims(window, a_layout, fill):
    """The layout of a direct kernel's intermediate ``Z`` and of its output.

    ``Z`` follows the input layout piece by piece, so that most pairs share a
    rotation: a label both hold keeps its pieces, and a windowed label becomes
    its output label, which frees the positions between outputs when the
    window strides. `fill` says which free positions take the output channel;
    the rest of the channel goes on ciphertexts.

    Args:
        window: the operator (``util.window.Window``)
        a_layout: the input's layout, without rolls
        fill: one of ``FILLS``

    Returns:
        (z_dims, out_dims), or None when a windowed dim cannot hold its output.
    """
    spec = window.spec
    z_dim = {label: i for i, label in enumerate(window.z_labels)}
    num_ct_pieces = len(a_layout.ct_dims)
    pieces = a_layout.ct_dims + a_layout.slot_dims

    windows = {}
    for relation in spec.relations:
        lhs_dim = spec.lhs.index(relation.label)
        at = [i for i, piece in enumerate(pieces) if piece.dim == lhs_dim]
        plan = _window_pieces(
            [pieces[i] for i in at],
            round_to_ceiling_power_of_2(window.extent(relation.base)),
            relation.stride,
        )
        if plan is None:
            return None
        windows.update({i: (z_dim[relation.base], p) for i, p in zip(at, plan)})

    # (piece, how it may be filled, whether it is a ciphertext piece)
    dims = []
    for at, piece in enumerate(pieces):
        in_ct = at < num_ct_pieces
        if piece.dim is None:
            # A gap stays free. Replication is read at its first copy, so a
            # replicated slot piece is free too and replicated ciphertexts
            # are not carried.
            if not in_ct:
                dims.append((Dim(None, piece.extent, 1, DimType.EMPTY), "gaps", False))
        elif at not in windows:
            dims.append(
                (
                    Dim(z_dim[spec.lhs[piece.dim]], piece.extent, piece.stride),
                    None,
                    in_ct,
                )
            )
        else:
            base, (gap, extent, stride, freed) = windows[at]
            if gap > 1 and not in_ct:
                dims.append((Dim(None, gap, 1, DimType.EMPTY), "gaps", False))
            if extent > 1:
                dims.append((Dim(base, extent, stride), None, in_ct))
            if freed > 1 and not in_ct:
                dims.append((Dim(None, freed, 1, DimType.EMPTY), "strides", False))

    # Labels only the filter holds: free pieces first, innermost at stride 1,
    # and what is left on a leading ciphertext piece.
    leading = []
    wanted = {"none": (), "strides": ("strides",), "gaps": ("strides", "gaps")}[fill]
    for label in spec.out:
        if label in spec.lhs or window.window_of(label) is not None:
            continue
        extent = window.extent(label)
        room = round_to_ceiling_power_of_2(extent)
        stride = 1
        for at in range(len(dims) - 1, -1, -1):
            piece, free, _ = dims[at]
            if free not in wanted or room == 1:
                continue
            take = min(piece.extent, room)
            filled = [(Dim(z_dim[label], take, stride), None, False)]
            if take < piece.extent:
                gap = Dim(None, piece.extent // take, 1, DimType.EMPTY)
                filled.insert(0, (gap, None, False))
            dims[at : at + 1] = filled
            room //= take
            stride *= take
        if math.ceil(extent / stride) > 1:
            leading.append(
                (Dim(z_dim[label], math.ceil(extent / stride), stride), None, True)
            )

    summed = set(window.summed)
    z_dims = []
    out_dims = []
    for piece, _, in_ct in leading + dims:
        z_dims.append(piece)
        if piece.dim not in summed:
            out_dims.append(piece)
        elif not in_ct:
            out_dims.append(Dim(None, piece.extent, 1, DimType.EMPTY))
    return z_dims, out_dims
