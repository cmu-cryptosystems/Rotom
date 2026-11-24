"""
Helpers for binary-op layout alignment, roll-based alignment, and output layout
construction.
"""

from copy import deepcopy as copy

from assignment.alignment import paired_pieces
from frontends.tensor import TensorOp
from ir.dim import Dim, DimType
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from ir.roll import Roll
from util.layout_util import (
    align_dimension_extents,
    dimension_merging,
    get_dim_map,
    match_dims,
    merge_dims,
)
from util.util import prod, split_dim


def check_extent_alignment(alignment, a_dims, b_dims):
    a_extent_map = {}
    for dim in a_dims:
        if dim.dim not in a_extent_map:
            a_extent_map[dim.dim] = []
        a_extent_map[dim.dim].append((dim.extent, dim.stride))

    b_extent_map = {}
    for dim in b_dims:
        if dim.dim not in b_extent_map:
            b_extent_map[dim.dim] = []
        b_extent_map[dim.dim].append((dim.extent, dim.stride))

    for k, v in alignment:
        if k in a_extent_map and sorted(a_extent_map[k]) != sorted(b_extent_map[v]):
            return False
    return True


def get_fill_len(dims):
    return int(prod([dim.extent for dim in dims if dim.dim_type == DimType.FILL]))


def match_kernel_dims(a_kernel, b_kernel):
    if len(a_kernel) < len(b_kernel):
        diff = len(b_kernel) // len(a_kernel)
        stride = 1
        for dim in a_kernel.layout.get_dims():
            if dim.dim_type == DimType.EMPTY:
                stride *= dim.extent
        dim = Dim(None, diff, stride, DimType.EMPTY)
        new_dims = a_kernel.layout.get_dims()
        new_dims.insert(0, dim)
        return new_dims, b_kernel.layout.get_dims()
    elif len(b_kernel) < len(a_kernel):
        diff = len(a_kernel) // len(b_kernel)
        stride = 1
        for dim in b_kernel.layout.get_dims():
            if dim.dim_type == DimType.EMPTY:
                stride *= dim.extent

        dim = Dim(None, diff, stride, DimType.EMPTY)
        new_dims = b_kernel.layout.get_dims()
        new_dims.insert(0, dim)
        return a_kernel.layout.get_dims(), new_dims
    return a_kernel.layout.get_dims(), b_kernel.layout.get_dims()


def replicate_dimensions(a_kernel, b_kernel, shapes, alignment):
    # match kernel lengths
    a_dims, b_dims = match_kernel_dims(a_kernel, b_kernel)

    # figure out how much replication is needed
    a_fill_len = get_fill_len(a_dims)
    b_fill_len = get_fill_len(b_dims)
    fill_len = max(a_fill_len, b_fill_len)

    # replication should be based on the alignment of the term
    a_shape = shapes[0]
    b_shape = shapes[1]
    replicated_shape = []
    for a_dim, b_dim in alignment:
        if a_dim is not None:
            replicated_shape.append(a_shape[a_dim])
        elif b_dim is not None:
            replicated_shape.append(b_shape[b_dim])

    # figure out how much each term needs to be replicated to match alignment
    replicated_kernels = []
    for kernel in [a_kernel, b_kernel]:
        shape_len = max(fill_len, prod(replicated_shape))
        kernel_len = int(
            prod(
                [
                    dim.extent
                    for dim in kernel.layout.get_dims()
                    if dim.dim_type is DimType.FILL
                ]
            )
        )

        shape_len //= kernel_len
        if shape_len <= 1:
            replicated_kernels.append(kernel)
            continue

        new_dims = []
        replicated = 1
        for dim in kernel.layout.get_dims()[::-1]:
            if shape_len > 1 and dim.dim_type == DimType.EMPTY:
                if dim.extent <= shape_len:
                    shape_len //= dim.extent
                    new_dim = Dim(dim.dim, dim.extent, replicated, DimType.FILL)
                    new_dims.insert(0, new_dim)
                    replicated *= dim.extent
                else:
                    split_dim_1, split_dim_2 = split_dim(dim, dim.extent // shape_len)
                    shape_len //= dim.extent
                    split_dim_1.dim_type = DimType.EMPTY
                    split_dim_2.dim_type = DimType.FILL
                    split_dim_2.stride = replicated
                    replicated *= dim.extent
                    new_dims.insert(0, split_dim_2)
                    new_dims.insert(0, split_dim_1)
            else:
                new_dim = copy(dim)
                new_dims.insert(0, new_dim)

        if shape_len > 1:
            new_dims.insert(0, Dim(None, shape_len, replicated))

        layout = dimension_merging(
            Layout(
                kernel.layout.term,
                kernel.layout.rolls,
                new_dims,
                kernel.layout.n,
                kernel.layout.secret,
            )
        )
        replicated_kernels.append(Kernel(KernelOp.REPLICATE, [kernel], layout))

    a_fill_len = get_fill_len(replicated_kernels[0].layout.get_dims())
    b_fill_len = get_fill_len(replicated_kernels[1].layout.get_dims())
    assert a_fill_len == b_fill_len
    return replicated_kernels


def check_dim_len_eq(a_dims, b_dims):
    a_dim_len = int(
        prod([dim.extent for dim in a_dims if dim.dim_type is DimType.FILL])
    )
    b_dim_len = int(
        prod([dim.extent for dim in b_dims if dim.dim_type is DimType.FILL])
    )
    return a_dim_len == b_dim_len


def check_ct_dim_alignment(alignment, a_kernel, b_kernel):
    a_ct_dims = a_kernel.layout.ct_dims
    b_ct_dims = b_kernel.layout.ct_dims

    a_ct_dims_map = {}
    for ct_dim in a_ct_dims:
        if ct_dim.dim not in a_ct_dims_map:
            a_ct_dims_map[ct_dim.dim] = 1
        a_ct_dims_map[ct_dim.dim] *= ct_dim.extent
    b_ct_dims_map = {}
    for ct_dim in b_ct_dims:
        if ct_dim.dim not in b_ct_dims_map:
            b_ct_dims_map[ct_dim.dim] = 1
        b_ct_dims_map[ct_dim.dim] *= ct_dim.extent

    for a, b in alignment:
        if a is None or b is None:
            continue
        if a in a_ct_dims_map and b not in b_ct_dims_map:
            return False
        elif a not in a_ct_dims_map and b in b_ct_dims_map:
            return False
        elif (
            a in a_ct_dims_map
            and b in b_ct_dims_map
            and a_ct_dims_map[a] != b_ct_dims_map[b]
        ):
            return False

    a_stride = 1
    for ct_dim in a_ct_dims:
        if ct_dim.dim is not None:
            a_stride *= ct_dim.stride
    b_stride = 1
    for ct_dim in b_ct_dims:
        if ct_dim.dim is not None:
            b_stride *= ct_dim.stride
    return True


def check_slot_dim_alignment(alignment, a_kernel, b_kernel):
    a_dims = a_kernel.layout.slot_dims
    b_dims = b_kernel.layout.slot_dims

    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0
    while a_next < len(a_dims) and b_next < len(b_dims):
        a_dim = a_dims[a_next]
        b_dim = b_dims[b_next]
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        if a_dim.dim_type != b_dim.dim_type:
            return False

        if (a_dim.dim, b_dim.dim) not in alignment:
            return False

        a_extent = prod([dim.extent for dim in a_dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in b_dims[b_start : b_next + 1]])
        if a_extent == b_extent:
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
        elif a_extent > b_extent:
            b_next += 1
        else:
            a_next += 1
    return True


def check_ct_roll_alignment(alignment, a_kernel, b_kernel):
    alignment_map = {}
    rev_alignment_map = {}
    for k, v in alignment:
        alignment_map[k] = v
        rev_alignment_map[v] = k

    a_rolls = a_kernel.layout.rolls
    b_rolls = b_kernel.layout.rolls

    a_roll_set = set()
    for a_roll in a_rolls:
        a_roll_set.add((a_roll.dim_to_roll.dim, a_roll.dim_to_roll.extent))
    b_roll_set = set()
    for b_roll in b_rolls:
        b_roll_set.add((b_roll.dim_to_roll.dim, b_roll.dim_to_roll.extent))

    for a in a_roll_set:
        if a[0] not in alignment_map:
            return False
        if (alignment_map[a[0]], a[1]) not in b_roll_set:
            return False
    for b in b_roll_set:
        if b[0] not in rev_alignment_map:
            return False
        if (rev_alignment_map[b[0]], b[1]) not in a_roll_set:
            return False
    return True


def _digit_strides(pieces):
    """The index stride of each binary digit the pieces hold, outermost first.

    ``[0:4:2]`` holds two digits, worth 4 and 2. A replicated piece holds no
    index, so its digits are None.
    """
    strides = []
    for piece in pieces:
        extent = piece.extent
        while extent > 1:
            extent //= 2
            strides.append(None if piece.dim is None else piece.stride * extent)
    return strides


def _same_digits(a_pieces, b_pieces):
    """True when two runs of pieces put the same index digit at each position.

    ``[0:2:1][0:2:2]`` and ``[0:4:1]`` cover the same extent of dim 0 but hold
    its elements in a different order.
    """
    return all(
        a is None or b is None or a == b
        for a, b in zip(_digit_strides(a_pieces), _digit_strides(b_pieces))
    )


def check_dim_alignment(alignment, a_kernel, b_kernel):
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    assert check_dim_len_eq(a_dims, b_dims)

    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0
    while a_next < len(a_dims) and b_next < len(b_dims):
        a_dim = a_dims[a_next]
        b_dim = b_dims[b_next]
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        if a_dim.dim_type != b_dim.dim_type:
            return False

        if (a_dim.dim, b_dim.dim) not in alignment:
            return False

        a_extent = prod([dim.extent for dim in a_dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in b_dims[b_start : b_next + 1]])
        if a_extent == b_extent:
            if not _same_digits(
                a_dims[a_start : a_next + 1], b_dims[b_start : b_next + 1]
            ):
                return False
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
        elif a_extent > b_extent:
            b_next += 1
        else:
            a_next += 1
    return True


def _sum_dim(spec, side):
    """The dim of operand `side` (0 lhs, 1 rhs) that the operator sums away."""
    dims = spec.lhs_sum_dims() if side == 0 else spec.rhs_sum_dims()
    return dims[0] if len(dims) == 1 else None


def rolled_spec(spec, a_kernel, b_kernel):
    """`spec` over the pieces and rolls of the two operand layouts."""
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    return spec.with_rolls(
        [(d.dim, d.extent, d.stride) for d in a_dims],
        [(d.dim, d.extent, d.stride) for d in b_dims],
        [roll.roll_index(a_dims) for roll in a_kernel.layout.rolls],
        [roll.roll_index(b_dims) for roll in b_kernel.layout.rolls],
    )


def check_roll_alignment(spec, alignment, a_kernel, b_kernel):
    """True when the pieces and rolls of the two layouts give both operands
    one traversal."""
    return rolled_spec(spec, a_kernel, b_kernel).aligned()


def check_alignment(spec, alignment, a_kernel, b_kernel):
    # An elementwise operator pairs the operands ciphertext by ciphertext.
    if not spec.contracted() and (a_kernel.layout.num_ct() != b_kernel.layout.num_ct()):
        return False
    return check_dim_alignment(alignment, a_kernel, b_kernel) and check_roll_alignment(
        spec, alignment, a_kernel, b_kernel
    )


def _facing_dims(dims, other_dims, partners, side):
    """`dims` relabeled with the dims of the other operand that face them.

    Args:
        dims: the pieces of one operand's layout
        other_dims: the pieces of the other operand's layout
        partners: for each dim of this operand, the other operand's dims the
            alignment pairs it with (None for replication)
        side: "a" or "b", the operand `dims` belongs to, for error messages

    Returns:
        The other operand laid out like `dims`: one piece per piece, holding
        the facing dim, with gaps kept as gaps.

    Raises:
        KeyError: If a piece has no facing dim, or faces more than one
    """
    other = "b" if side == "a" else "a"
    partners = {dim: list(found) for dim, found in partners.items()}
    remaining = {}
    stride = {}
    for other_dim in other_dims:
        if other_dim.dim not in remaining:
            remaining[other_dim.dim] = other_dim.extent
            stride[other_dim.dim] = other_dim.extent
        else:
            remaining[other_dim.dim] *= other_dim.extent
            stride[other_dim.dim] *= other_dim.extent

    facing = []
    for piece in dims:
        if piece.dim_type == DimType.EMPTY:
            facing.append(copy(piece))
            continue
        extent = piece.extent
        dim = copy(piece)
        if piece.dim not in partners or len(partners[piece.dim]) == 0:
            raise KeyError(f"missing alignment for {side}-dim {piece.dim}")
        _MISSING = object()
        picked = _MISSING
        for cand in partners[piece.dim]:
            if cand in remaining:
                picked = cand
                break
        if picked is _MISSING:
            raise KeyError(
                f"no remaining aligned {other}-dim for {side}-dim {piece.dim}; "
                f"candidates={partners[piece.dim]}, remaining={list(remaining.keys())}"
            )
        dim.dim = picked
        if remaining[dim.dim] <= extent:
            dim.extent = remaining[dim.dim]
            dim.stride = stride[dim.dim] // dim.extent
            extent //= remaining[dim.dim]
            del remaining[dim.dim]
            stride[dim.dim] //= dim.extent
            partners[piece.dim].remove(dim.dim)
        else:
            dim.extent = extent
            dim.stride = stride[dim.dim] // dim.extent
            remaining[dim.dim] //= extent
            stride[dim.dim] //= dim.extent
            extent = 1
        if extent != 1:
            # The piece faces more than one dim, and one piece holds one.
            raise KeyError(f"{side}-piece {piece} faces more than one {other}-dim")
        facing.append(dim)

    for piece, dim in zip(dims, facing):
        if piece.dim is not None and dim.dim is not None:
            dim.stride = piece.stride
    return facing


def _partners(alignment):
    """`(a_partners, b_partners)`: the dims each operand's dims are paired with."""
    a_partners = {}
    b_partners = {}
    for a, b in alignment:
        a_partners.setdefault(a, []).append(b)
        b_partners.setdefault(b, []).append(a)
    return a_partners, b_partners


def get_aligned_dimensions(alignment, a_dims, b_dims):
    """`(aligned_b_dims, aligned_a_dims)`: each operand laid out like the
    other, piece for piece.

    Raises:
        KeyError: If either operand cannot be laid out like the other
    """
    a_partners, b_partners = _partners(alignment)
    aligned_b_dims = _facing_dims(a_dims, b_dims, a_partners, "a")
    aligned_a_dims = _facing_dims(b_dims, a_dims, b_partners, "b")
    return aligned_b_dims, aligned_a_dims


def plaintext_side(alignment, a_kernel, b_kernel):
    """The side (0 or 1) of an operand that is a plaintext in the other
    operand's layout, or None.

    That is a public operand, not replicated, whose every dim faces a dim of
    the other operand. It needs no layout of its own and no replication.
    """
    if (
        not b_kernel.layout.secret
        and b_kernel.op == KernelOp.CS
        and all(a is not None for a, _ in alignment)
    ):
        return 1
    if (
        not a_kernel.layout.secret
        and a_kernel.op == KernelOp.CS
        and all(b is not None for _, b in alignment)
    ):
        return 0
    return None


def _dims_like(dims, partners):
    """A public operand's layout, read off the other operand's `dims`: each
    piece holds the dim it faces, or replication where the public operand has
    an axis of extent 1, whatever extents the two operands were padded to."""
    out = []
    for piece in dims:
        if piece.dim_type == DimType.EMPTY or piece.dim is None:
            out.append(copy(piece))
            continue
        (partner,) = partners[piece.dim]
        out.append(Dim(partner, piece.extent, piece.stride, DimType.FILL))
    return out


def match_public_kernel(alignment, a_kernel, b_kernel, left):
    """The public operand (the lhs if `left`) laid out like the other one.

    Only that direction has to exist: a replicated piece of the public
    operand may face several dims of the other, which one piece of the other
    could not be laid out like.
    """
    a_partners, b_partners = _partners(alignment)
    a_dims, b_dims = a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
    try:
        side = plaintext_side(alignment, a_kernel, b_kernel)
        if left and side == 0:
            aligned_a_dims = _dims_like(b_dims, b_partners)
        elif left:
            aligned_a_dims = _facing_dims(b_dims, a_dims, b_partners, "b")
        elif side == 1:
            aligned_b_dims = _dims_like(a_dims, a_partners)
        else:
            aligned_b_dims = _facing_dims(a_dims, b_dims, a_partners, "a")
    except KeyError:
        return None

    if left:
        roll_indices = []
        for roll in b_kernel.layout.rolls:
            roll_indices.append(roll.roll_index(b_kernel.layout.get_dims()))

        new_rolls = []
        for roll_index in roll_indices:
            new_rolls.append(
                Roll(aligned_a_dims[roll_index[0]], aligned_a_dims[roll_index[1]])
            )

        aligned_a_layout = Layout(
            a_kernel.layout.term,
            new_rolls,
            aligned_a_dims,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        aligned_a_kernel = Kernel(
            KernelOp.TENSOR,
            [],
            aligned_a_layout,
        )
        return (aligned_a_kernel, b_kernel)
    else:
        roll_indices = []
        for roll in a_kernel.layout.rolls:
            roll_indices.append(roll.roll_index(a_kernel.layout.get_dims()))

        new_rolls = []
        for roll_index in roll_indices:
            new_rolls.append(
                Roll(aligned_b_dims[roll_index[0]], aligned_b_dims[roll_index[1]])
            )
        aligned_b_layout = Layout(
            b_kernel.layout.term,
            new_rolls,
            aligned_b_dims,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        aligned_b_kernel = Kernel(
            KernelOp.TENSOR,
            [],
            aligned_b_layout,
        )
        return (a_kernel, aligned_b_kernel)


def conv_dimensions(alignment, kernels):
    a_kernel = kernels[0]
    b_kernel = kernels[1]

    aligned_kernels = []
    if (not a_kernel.layout.secret) or (a_kernel.op == KernelOp.TENSOR):
        mk = match_public_kernel(alignment, a_kernel, b_kernel, True)
        if mk is not None:
            aligned_kernels.append(mk)
    if (not b_kernel.layout.secret) or (b_kernel.op == KernelOp.TENSOR):
        mk = match_public_kernel(alignment, a_kernel, b_kernel, False)
        if mk is not None:
            aligned_kernels.append(mk)

    if a_kernel.layout.secret and b_kernel.layout.secret:
        extent_a_dims, extent_b_dims = align_dimension_extents(
            a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
        )
        try:
            aligned_b_dims, aligned_a_dims = get_aligned_dimensions(
                alignment, extent_a_dims, extent_b_dims
            )
        except KeyError:
            return []

        a_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls,
            extent_a_dims,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        a_extent_kernel = Kernel(
            a_kernel.op,
            a_kernel.cs,
            a_layout,
        )

        conv_b_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls,
            aligned_b_dims,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )

        conv_b_kernel = Kernel(
            KernelOp.CONVERSION,
            [tuple(extent_b_dims), tuple(aligned_b_dims), b_kernel],
            conv_b_layout,
        )
        aligned_kernels.append((a_extent_kernel, conv_b_kernel))

        b_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls,
            extent_b_dims,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        b_extent_kernel = Kernel(
            b_kernel.op,
            b_kernel.cs,
            b_layout,
        )
        conv_a_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls,
            aligned_a_dims,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        conv_a_kernel = Kernel(
            KernelOp.CONVERSION,
            [tuple(extent_a_dims), tuple(aligned_a_dims), a_kernel],
            conv_a_layout,
        )
        aligned_kernels.append((conv_a_kernel, b_extent_kernel))

    return aligned_kernels


def conv_ct_dimensions(alignment, kernels):
    aligned_kernels = []
    a_kernel = kernels[0]
    b_kernel = kernels[1]

    try:
        aligned_b_dims, aligned_a_dims = get_aligned_dimensions(
            alignment, a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
        )
    except KeyError:
        return []
    conv_b_layout = Layout(
        b_kernel.layout.term,
        b_kernel.layout.rolls,
        aligned_b_dims,
        b_kernel.layout.n,
        b_kernel.layout.secret,
    )
    conv_b_kernel = Kernel(
        KernelOp.CONVERSION,
        [tuple(b_kernel.layout.get_dims()), tuple(aligned_b_dims), b_kernel],
        conv_b_layout,
    )

    aligned_kernels.append((a_kernel, conv_b_kernel))

    conv_a_layout = Layout(
        a_kernel.layout.term,
        a_kernel.layout.rolls,
        aligned_a_dims,
        a_kernel.layout.n,
        a_kernel.layout.secret,
    )
    conv_a_kernel = Kernel(
        KernelOp.CONVERSION,
        [tuple(a_kernel.layout.get_dims()), tuple(aligned_a_dims), a_kernel],
        conv_a_layout,
    )

    aligned_kernels.append((conv_a_kernel, b_kernel))

    return aligned_kernels


def get_dim_from_alignment(alignment, dim, index, remaining_dims):
    if dim.dim_type == DimType.EMPTY:
        return dim

    for aligned_dims in alignment:
        if aligned_dims[index] == dim.dim:
            for remaining_dim in remaining_dims:
                if aligned_dims[index ^ 1] == remaining_dim.dim:
                    remaining_dims.remove(remaining_dim)
                    return remaining_dim

    return remaining_dims[0]


def apply_sum_roll(spec, kernel, side):
    sum_dim = _sum_dim(spec, side)

    layout = kernel.layout

    if any(dim.dim is None for dim in layout.ct_dims) and any(
        dim.dim == sum_dim for dim in layout.slot_dims
    ):
        for roll in layout.rolls:
            if roll.dim_to_roll.dim == sum_dim or roll.dim_to_roll_by.dim == sum_dim:
                return

        empty_ct_dim_index = 0
        count = 0
        for i, ct_dim in enumerate(merge_dims(layout.ct_dims, set())):
            if ct_dim.dim is None:
                empty_ct_dim_index = i
                count += 1

        assert count == 1

        slot_dim_index = 0
        slot_dim_len = 0
        for i, slot_dim in enumerate(layout.slot_dims):
            if slot_dim.dim == sum_dim and slot_dim.extent > slot_dim_len:
                slot_dim_index = i
                slot_dim_len = slot_dim.extent

        ct_dim_to_roll = layout.ct_dims[empty_ct_dim_index]
        slot_dim_to_roll = layout.slot_dims[slot_dim_index]
        if slot_dim_to_roll.extent == ct_dim_to_roll.extent:
            new_ct_dims = layout.ct_dims
            new_slot_dims = layout.slot_dims
        elif slot_dim_to_roll.extent <= ct_dim_to_roll.extent:
            split_extent_1 = slot_dim_to_roll.extent
            split_extent_2 = ct_dim_to_roll.extent // slot_dim_to_roll.extent
            split_dim = Dim(ct_dim_to_roll.dim, split_extent_1, ct_dim_to_roll.stride)
            split_dim_rep = Dim(
                ct_dim_to_roll.dim,
                split_extent_2,
                split_extent_1 * ct_dim_to_roll.stride,
            )

            new_ct_dims = (
                layout.ct_dims[:empty_ct_dim_index]
                + [split_dim, split_dim_rep]
                + layout.ct_dims[empty_ct_dim_index + 1 :]
            )
            new_slot_dims = layout.slot_dims
        else:
            ct_extent = ct_dim_to_roll.extent
            if (
                slot_dim_index + 1 < len(layout.slot_dims)
                and layout.slot_dims[slot_dim_index + 1].dim is None
            ):
                split_extent_1 = slot_dim_to_roll.extent // ct_extent
                split_extent_2 = ct_extent
            else:
                split_extent_1 = ct_extent
                split_extent_2 = slot_dim_to_roll.extent // ct_extent

            split_dim = Dim(
                slot_dim_to_roll.dim,
                split_extent_1,
                split_extent_2 * slot_dim_to_roll.stride,
            )
            split_dim_rep = Dim(
                slot_dim_to_roll.dim, split_extent_2, slot_dim_to_roll.stride
            )

            new_ct_dims = layout.ct_dims
            new_slot_dims = (
                layout.slot_dims[:slot_dim_index]
                + [split_dim, split_dim_rep]
                + layout.slot_dims[slot_dim_index + 1 :]
            )

        new_dims = new_ct_dims + new_slot_dims
        (
            new_dims[empty_ct_dim_index],
            new_dims[len(new_ct_dims) + slot_dim_index],
        ) = (
            new_slot_dims[slot_dim_index],
            new_ct_dims[empty_ct_dim_index],
        )

        dim_to_roll = new_dims[empty_ct_dim_index]
        dim_to_roll_by = new_dims[len(new_ct_dims) + slot_dim_index]

        # Guard against constructing invalid rolls; if extents don't match,
        # this layout candidate is not roll-alignable under the original
        # semantics, so we skip applying a sum roll.
        if dim_to_roll.extent != dim_to_roll_by.extent:
            return None

        new_roll = Roll(dim_to_roll, dim_to_roll_by)

        roll_indices = [roll.roll_index(layout.get_dims()) for roll in layout.rolls]
        updated_rolls = []
        for roll, index in zip(layout.rolls, roll_indices):
            updated_rolls.append(Roll(roll.dim_to_roll, new_dims[index[1]]))

        if new_roll in updated_rolls:
            updated_layout = Layout(
                kernel.layout.term,
                updated_rolls,
                new_dims,
                kernel.layout.n,
                kernel.layout.secret,
            )
            updated_kernel = copy(kernel)
            updated_kernel.layout = updated_layout
            return updated_kernel
        else:
            new_layout = Layout(
                kernel.layout.term,
                updated_rolls + [new_roll],
                new_dims,
                kernel.layout.n,
                kernel.layout.secret,
            )
            return Kernel(KernelOp.ROLL, [new_roll, kernel], new_layout)


def roll_dim_alignment(spec, alignment, a_kernel, b_kernel):
    aligned_kernels = []

    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    extent_a_dims, extent_b_dims = align_dimension_extents(
        a_kernel.layout.get_dims(), b_kernel.layout.get_dims()
    )

    a_sum_dim = _sum_dim(spec, 0)
    b_sum_dim = _sum_dim(spec, 1)

    for a_dim, b_dim in zip(extent_a_dims, extent_b_dims):
        if a_dim.dim == a_sum_dim and b_dim.dim != b_sum_dim:
            return []
        if b_dim.dim == b_sum_dim and a_dim.dim != a_sum_dim:
            return []
        if (
            a_dim.dim == a_sum_dim
            and b_dim.dim == b_sum_dim
            and a_dim.stride != b_dim.stride
        ):
            return []
        if (
            b_dim.dim == b_sum_dim
            and a_dim.dim == a_sum_dim
            and a_dim.stride != b_dim.stride
        ):
            return []

    a_sum_rolls = set()
    for roll in a_kernel.layout.rolls:
        if roll.dim_to_roll.dim == a_sum_dim:
            a_sum_rolls.add(roll.roll_index(a_dims))
    b_sum_rolls = set()
    for roll in b_kernel.layout.rolls:
        if roll.dim_to_roll.dim == b_sum_dim:
            b_sum_rolls.add(roll.roll_index(b_dims))

    if a_sum_rolls != b_sum_rolls:
        return []

    aligned_b_dims, aligned_a_dims = get_aligned_dimensions(alignment, a_dims, b_dims)
    for a_dim in a_dims:
        if a_dim not in aligned_a_dims:
            return []
    for b_dim in b_dims:
        if b_dim not in aligned_b_dims:
            return []

    b_dim_map = get_dim_map(b_dims)
    aligned_b_dim_map = get_dim_map(aligned_b_dims)

    b_rolls = []
    for k, v in aligned_b_dim_map.items():
        if k.dim is not None and v != b_dim_map[k]:
            if k.extent != b_dims[v].extent:
                return []
            b_rolls.append(Roll(k, b_dims[v]))

    b_dims = copy(b_dims)
    rolled_b_kernel = b_kernel
    for roll in b_rolls:
        if roll in b_kernel.layout.rolls:
            continue

        roll_idx = roll.roll_index(b_dims)

        b_dims[roll_idx[0]], b_dims[roll_idx[1]] = (
            b_dims[roll_idx[1]],
            b_dims[roll_idx[0]],
        )

        b_rolls = []
        for existing_roll in b_kernel.layout.rolls:
            update = existing_roll.roll_update(b_kernel.layout.get_dims())
            dim_to_roll = update[0]
            dim_to_roll_by = b_dims[update[1]]
            og_dim_to_roll_by = existing_roll.dim_to_roll_by
            if dim_to_roll != dim_to_roll_by:
                b_rolls.append(Roll(dim_to_roll, dim_to_roll_by))
            else:
                b_rolls.append(Roll(dim_to_roll, og_dim_to_roll_by))

        b_rolls.append(roll)
        rolled_b_layout = Layout(
            b_kernel.layout.term,
            b_rolls,
            b_dims,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        rolled_b_kernel = Kernel(
            KernelOp.ROLL,
            [roll, rolled_b_kernel],
            rolled_b_layout,
        )
    if check_alignment(spec, alignment, a_kernel, rolled_b_kernel):
        aligned_kernels.append((a_kernel, rolled_b_kernel))

    a_dim_map = get_dim_map(a_dims)
    aligned_a_dim_map = get_dim_map(aligned_a_dims)

    a_rolls = []
    for k, v in aligned_a_dim_map.items():
        if k.dim is not None and v != a_dim_map[k]:
            a_rolls.append(Roll(k, a_dims[v]))

    a_dims = copy(a_dims)
    rolled_a_kernel = a_kernel
    for roll in a_rolls:
        roll_idx = roll.roll_index(a_dims)

        a_dims[roll_idx[0]], a_dims[roll_idx[1]] = (
            a_dims[roll_idx[1]],
            a_dims[roll_idx[0]],
        )

        a_rolls = []
        for existing_roll in a_kernel.layout.rolls:
            update = existing_roll.roll_update(a_kernel.layout.get_dims())
            dim_to_roll = update[0]
            dim_to_roll_by = a_dims[update[1]]
            og_dim_to_roll_by = existing_roll.dim_to_roll_by
            if dim_to_roll != dim_to_roll_by:
                a_rolls.append(Roll(dim_to_roll, dim_to_roll_by))
            else:
                a_rolls.append(Roll(dim_to_roll, og_dim_to_roll_by))

        if roll in a_rolls:
            updated_a_layout = Layout(
                a_kernel.layout.term,
                a_rolls,
                a_dims,
                a_kernel.layout.n,
                a_kernel.layout.secret,
            )
            updated_a_kernel = copy(rolled_a_kernel)
            updated_a_kernel.layout = updated_a_layout
            rolled_a_kernel = updated_a_kernel
        else:
            a_rolls.append(roll)
            rolled_a_layout = Layout(
                a_kernel.layout.term,
                a_rolls,
                a_dims,
                a_kernel.layout.n,
                a_kernel.layout.secret,
            )
            rolled_a_kernel = Kernel(
                KernelOp.ROLL,
                [roll, rolled_a_kernel],
                rolled_a_layout,
            )
    if check_alignment(spec, alignment, rolled_a_kernel, b_kernel):
        aligned_kernels.append((rolled_a_kernel, b_kernel))
    return aligned_kernels


def _shared_dims(spec):
    """The dims of each operand whose label the other operand also holds."""
    lhs = {d for d, label in enumerate(spec.lhs) if label in spec.rhs}
    rhs = {d for d, label in enumerate(spec.rhs) if label in spec.lhs}
    return lhs, rhs


def roll_roll_alignment(spec, a_kernel, b_kernel):
    """Mirrors each roll of a shared label onto the operand that lacks it."""
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    a_shared, b_shared = _shared_dims(spec)

    if len(a_dims) > len(b_dims):
        b_dims = match_dims(b_dims, a_dims)
    elif len(b_dims) > len(a_dims):
        a_dims = match_dims(a_dims, b_dims)

    a_sum_rolls = []
    b_sum_rolls = []
    for roll in a_kernel.layout.rolls:
        if roll.dim_to_roll.dim in a_shared:
            dim_to_roll_idx = a_dims.index(roll.dim_to_roll)
            dim_to_roll_by_idx = a_dims.index(roll.dim_to_roll_by)
            a_sum_rolls.append((dim_to_roll_idx, dim_to_roll_by_idx))
    for roll in b_kernel.layout.rolls:
        if roll.dim_to_roll.dim in b_shared:
            dim_to_roll_idx = b_dims.index(roll.dim_to_roll)
            dim_to_roll_by_idx = b_dims.index(roll.dim_to_roll_by)
            b_sum_rolls.append((dim_to_roll_idx, dim_to_roll_by_idx))

    a_roll_indices = []
    b_roll_indices = []
    for a_roll in a_sum_rolls:
        if a_roll not in b_sum_rolls:
            b_roll_indices.append(a_roll)
    for b_roll in b_sum_rolls:
        if b_roll not in a_sum_rolls:
            a_roll_indices.append(b_roll)

    a_rolls = []
    for roll in a_roll_indices:
        a_rolls.append(Roll(a_dims[roll[0]], a_dims[roll[1]]))
    b_rolls = []
    for roll in b_roll_indices:
        b_rolls.append(Roll(b_dims[roll[0]], b_dims[roll[1]]))

    # A ROLL kernel applies one roll.
    if len(a_rolls) > 1 or len(b_rolls) > 1:
        return None

    if b_rolls:
        rolled_b_layout = Layout(
            b_kernel.layout.term,
            b_kernel.layout.rolls + b_rolls,
            b_dims,
            b_kernel.layout.n,
            b_kernel.layout.secret,
        )
        rolled_b_kernel = Kernel(
            KernelOp.ROLL,
            [b_rolls[0], b_kernel],
            rolled_b_layout,
        )
    else:
        rolled_b_kernel = b_kernel

    if a_rolls:
        rolled_a_layout = Layout(
            a_kernel.layout.term,
            a_kernel.layout.rolls + a_rolls,
            a_dims,
            a_kernel.layout.n,
            a_kernel.layout.secret,
        )
        rolled_a_kernel = Kernel(
            KernelOp.ROLL,
            [a_rolls[0], a_kernel],
            rolled_a_layout,
        )

    else:
        rolled_a_kernel = a_kernel
    return (rolled_a_kernel, rolled_b_kernel)


def apply_roll_conversion(spec, alignment, kernels):
    """Adds the rolls that align two layouts.

    Layouts whose dims already align need at most the missing rolls mirrored
    (roll_roll_alignment). Layouts whose dims do not align are aligned by
    rolling a dim into place (roll_dim_alignment). Either result is checked by
    the caller.
    """
    if not check_dim_alignment(alignment, kernels[0], kernels[1]):
        return roll_dim_alignment(spec, alignment, kernels[0], kernels[1])
    if check_roll_alignment(spec, alignment, kernels[0], kernels[1]):
        return [kernels]
    mirrored = roll_roll_alignment(spec, kernels[0], kernels[1])
    return [mirrored] if mirrored is not None else []


def roll_dimensions(spec, alignment, kernels):
    a_kernel = kernels[0]
    b_kernel = kernels[1]

    aligned_kernels = []
    if (not a_kernel.layout.secret) or (a_kernel.op == KernelOp.TENSOR):
        mk = match_public_kernel(alignment, a_kernel, b_kernel, True)
        if mk is not None:
            aligned_kernels.append(mk)
    if (not b_kernel.layout.secret) or (b_kernel.op == KernelOp.TENSOR):
        mk = match_public_kernel(alignment, a_kernel, b_kernel, False)
        if mk is not None:
            aligned_kernels.append(mk)

    if a_kernel.layout.secret and b_kernel.layout.secret:
        ct_dim_alignment = check_ct_dim_alignment(alignment, a_kernel, b_kernel)
        slot_dim_alignment = check_slot_dim_alignment(alignment, a_kernel, b_kernel)
        ct_roll_alignment = check_ct_roll_alignment(alignment, a_kernel, b_kernel)

        if ct_dim_alignment and slot_dim_alignment and ct_roll_alignment:
            a_rolls = {}
            for roll in kernels[0].layout.rolls:
                roll_index = roll.roll_index(kernels[0].layout.get_dims())
                if roll_index[0] not in a_rolls:
                    a_rolls[roll_index[0]] = []
                a_rolls[roll_index[0]].append(roll_index[1])

            b_rolls = {}
            for roll in kernels[1].layout.rolls:
                roll_index = roll.roll_index(kernels[1].layout.get_dims())
                if roll_index[0] not in b_rolls:
                    b_rolls[roll_index[0]] = []
                b_rolls[roll_index[0]].append(roll_index[1])

            for k, v in a_rolls.items():
                if k in b_rolls and v != b_rolls[k]:
                    return []
            for k, v in b_rolls.items():
                if k in a_rolls and v != a_rolls[k]:
                    return []

            try:
                ct_kernels = conv_ct_dimensions(alignment, kernels)
                for kernels in ct_kernels:
                    skip = False
                    for roll in kernels[0].layout.rolls:
                        if (
                            roll.dim_to_roll not in kernels[0].layout.get_dims()
                            or roll.dim_to_roll_by not in kernels[0].layout.get_dims()
                        ):
                            skip = True
                    for roll in kernels[1].layout.rolls:
                        if (
                            roll.dim_to_roll not in kernels[1].layout.get_dims()
                            or roll.dim_to_roll_by not in kernels[1].layout.get_dims()
                        ):
                            skip = True
                    if skip:
                        continue
                    aligned_kernels += apply_roll_conversion(spec, alignment, kernels)
            except Exception:
                aligned_kernels += apply_roll_conversion(spec, alignment, kernels)
        else:
            try:
                aligned_kernels += apply_roll_conversion(spec, alignment, kernels)
            except Exception:
                aligned_kernels += []
    return aligned_kernels


def match_layout(spec, kernels, alignment, roll_flag):
    if check_alignment(spec, alignment, kernels[0], kernels[1]):
        return [kernels]

    matched_layouts = []

    if not kernels[0].layout.rolls and not kernels[1].layout.rolls:
        matched_layouts += conv_dimensions(alignment, kernels)

    if roll_flag:
        matched_layouts += roll_dimensions(spec, alignment, kernels)

    matched = []
    for a_kernel, b_kernel in matched_layouts:
        if not check_alignment(spec, alignment, a_kernel, b_kernel):
            continue
        matched.append((a_kernel, b_kernel))
    return matched


def _renumbered(dim, out_dims):
    """`dim` with the output dim id its operand dim becomes under the spec."""
    if dim.dim is None or out_dims[dim.dim] == dim.dim:
        return dim
    return Dim(out_dims[dim.dim], dim.extent, dim.stride, dim.dim_type)


def _paired_dims(a_dims, b_dims):
    """`paired_pieces` over the dims of two aligned layouts.

    Returns:
        (pairs, a_at, b_at): the (a dim, b dim) pairs, and for each dim given
        the index of its pair, or None if the dim was cut.
    """

    def pieces(dims):
        return [(dim.dim, dim.extent, dim.stride) for dim in dims]

    def cut_dims(dims, cut):
        made = []
        cut = list(cut)
        for dim in dims:
            held = 1
            while held < dim.extent:
                d, extent, stride = cut.pop(0)
                made.append(Dim(d, extent, stride, dim.dim_type))
                held *= extent
        return made

    paired = paired_pieces(pieces(a_dims), pieces(b_dims))
    assert paired is not None, "aligned layouts hold the same positions"
    a_cut, b_cut, a_at, b_at = paired
    pairs = list(zip(cut_dims(a_dims, a_cut), cut_dims(b_dims, b_cut)))
    return pairs, a_at, b_at


def output_layout(term, spec, a_kernel, b_kernel):
    """The output kernel of a binary operator over two aligned operand kernels.

    The operator's structure is read off `spec`: which dims align, which are
    summed away, and the output dim id each operand dim becomes.
    """
    lhs_out = spec.lhs_out_dims()
    rhs_out = spec.rhs_out_dims()
    match term.op:
        case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
            # Each piece of the output is the piece of the operand that holds
            # a dim there. Replication faces the other operand's pieces, which
            # may be several, so the two layouts are cut where either begins.
            a_dims, b_dims = match_kernel_dims(a_kernel, b_kernel)
            pairs, _, _ = _paired_dims(a_dims, b_dims)
            # A shorter lhs was brought to the rhs's length by a leading gap,
            # which the output holds only where the rhs has a dim.
            lead, held = 0, len(a_kernel)
            while held < len(b_kernel) and pairs[lead][1].dim is None:
                held *= pairs[lead][0].extent
                lead += 1
            if held >= len(b_kernel):
                pairs = pairs[lead:]
            output_dims = []
            for a_dim, b_dim in pairs:
                if a_dim.dim is not None:
                    output_dims.append(_renumbered(a_dim, lhs_out))
                elif b_dim.dim is not None:
                    output_dims.append(_renumbered(b_dim, rhs_out))
                else:
                    output_dims.append(a_dim)

            output_layout = dimension_merging(
                Layout(
                    term,
                    a_kernel.layout.rolls,
                    output_dims,
                    a_kernel.layout.n,
                    a_kernel.layout.secret or b_kernel.layout.secret,
                )
            )
            match term.op:
                case TensorOp.ADD:
                    layout_op = KernelOp.ADD
                case TensorOp.SUB:
                    layout_op = KernelOp.SUB
                case TensorOp.MUL:
                    layout_op = KernelOp.MUL
            output_kernel = Kernel(layout_op, [a_kernel, b_kernel], output_layout)
            return output_kernel
        case TensorOp.MATMUL:
            a_layout = a_kernel.layout
            b_layout = b_kernel.layout
            output_dims = []

            (a_sum_dim,) = spec.lhs_sum_dims()
            (b_sum_dim,) = spec.rhs_sum_dims()

            # The operands hold the same positions, but a dim one holds in
            # two pieces can face a single replicated piece of the other.
            pairs, a_at, b_at = _paired_dims(a_layout.get_dims(), b_layout.get_dims())
            for a_dim, b_dim in pairs:
                if a_dim.dim is not None and b_dim.dim is None:
                    output_dims.append(_renumbered(a_dim, lhs_out))
                elif a_dim.dim is None and b_dim.dim is not None:
                    output_dims.append(_renumbered(b_dim, rhs_out))
                else:
                    output_dims.append(
                        Dim(None, a_dim.extent, a_dim.stride, DimType.EMPTY)
                    )

            a_rolls = [
                roll for roll in a_layout.rolls if roll.dim_to_roll.dim != a_sum_dim
            ]
            b_rolls = [
                roll
                for roll in b_kernel.layout.rolls
                if roll.dim_to_roll.dim != b_sum_dim
            ]

            roll_indices = []
            for roll in a_rolls:
                rolled, rolled_by = roll.roll_index(a_kernel.layout.get_dims())
                roll_indices.append((a_at[rolled], a_at[rolled_by]))
            for roll in b_rolls:
                rolled, rolled_by = roll.roll_index(b_kernel.layout.get_dims())
                roll_indices.append((b_at[rolled], b_at[rolled_by]))

            new_rolls = []
            for roll_index in roll_indices:
                new_rolls.append(
                    Roll(output_dims[roll_index[0]], output_dims[roll_index[1]])
                )

            output_layout = dimension_merging(
                Layout(
                    term,
                    new_rolls,
                    output_dims,
                    a_kernel.layout.n,
                    a_kernel.layout.secret or b_kernel.layout.secret,
                )
            )

            output_kernel = Kernel(KernelOp.MATMUL, [a_kernel, b_kernel], output_layout)
            return output_kernel
        case _:
            raise NotImplementedError(term.op)


def apply_sum_rolls(spec, replicated_kernels):
    rolled = []
    for kernels in replicated_kernels:
        rolled_a_kernel = apply_sum_roll(spec, copy(kernels[0]), 0)
        rolled_b_kernel = apply_sum_roll(spec, copy(kernels[1]), 1)

        if rolled_a_kernel and rolled_b_kernel:
            rolled.append(tuple([rolled_a_kernel, rolled_b_kernel]))
        if rolled_b_kernel:
            rolled.append(tuple([kernels[0], rolled_b_kernel]))
        if rolled_a_kernel:
            rolled.append(tuple([rolled_a_kernel, kernels[1]]))
    return rolled
