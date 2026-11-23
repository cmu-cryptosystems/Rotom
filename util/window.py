"""
A windowed operator (a convolution) as masked rotations between two layouts.

The operator's spec relates every element of an intermediate tensor ``Z`` to
the input elements it reads and the filter element that weights each one.
``Z`` holds the output labels and the labels the operator sums over that both
operands hold (the input channel), so ``Z[c_out, h_out, w_out, c_in]`` is the
sum over the taps of ``filter[c_out, c_in, kh, kw] * input[c_in, h, w]``.

Given a layout for the input and a layout for ``Z``, each related pair moves
an input slot to a ``Z`` slot between two ciphertexts. The pairs with the same
(``Z`` ciphertext, input ciphertext, slot shift) form one *group*: one rotation
of the input ciphertext and one multiply by a plaintext holding the pairs'
filter elements at their ``Z`` slots. A ``Z`` ciphertext is the sum of its
groups, and the output is ``Z`` summed over the labels the output drops.

A layout places each piece of each dim independently, so a pair's group is a
sum of one term per label. The groups are the sumset of the per-label terms,
which is how they are counted without enumerating the pairs.

Key functions:
- dim_positions: the position of each index of one dim under a layout
- position_indices: the tensor index held at each position of a layout
- window_keys: the groups of a window under two layouts
- window_ops: the HE operation counts of the kernel
- GroupWeights: the filter elements one group multiplies by
"""

import math
from dataclasses import dataclass

import numpy as np

from ir.dim import DimType


@dataclass(frozen=True)
class Window:
    """A windowed operator's spec, true shapes and padding.

    Attributes:
        spec: the operator's spec (``assignment.alignment.conv_spec``)
        in_shape, filter_shape, out_shape: unpadded extents, in label order
        pads: the padding before each windowed axis, in relation order
    """

    spec: object
    in_shape: tuple
    filter_shape: tuple
    out_shape: tuple
    pads: tuple

    @property
    def z_labels(self):
        """The output labels, then the labels both operands hold and the
        output drops."""
        spec = self.spec
        shared = tuple(
            label for label in spec.lhs if label in spec.rhs and label not in spec.out
        )
        return tuple(spec.out) + shared

    @property
    def z_shape(self):
        return tuple(self.extent(label) for label in self.z_labels)

    @property
    def summed(self):
        """The dims of ``Z`` the output drops."""
        return tuple(range(len(self.spec.out), len(self.z_labels)))

    def extent(self, label):
        spec = self.spec
        if label in spec.out:
            return int(self.out_shape[spec.out.index(label)])
        if label in spec.lhs:
            return int(self.in_shape[spec.lhs.index(label)])
        return int(self.filter_shape[spec.rhs.index(label)])

    def pad(self, relation):
        return int(self.pads[self.spec.relations.index(relation)])

    def window_of(self, base):
        """The relation whose output label is `base`, if any."""
        for relation in self.spec.relations:
            if relation.base == base:
                return relation
        return None


def _position_strides(dims):
    """The position stride of each piece within its ciphertext or slot block."""
    strides = []
    stride = 1
    for dim in reversed(dims):
        strides.append(stride)
        stride *= dim.extent
    return strides[::-1]


def dim_positions(layout, dim, extent):
    """The (ciphertext, slot) each index of `dim` sits at under `layout`.

    A layout places an element at the sum of its dims' positions. Gap and
    replication pieces sit at 0, so this is the element's first copy.

    Returns:
        (ct, slot): two arrays of length `extent`, -1 where the layout does not
        hold the index
    """
    index = np.zeros(1, dtype=np.int64)
    ct = np.zeros(1, dtype=np.int64)
    slot = np.zeros(1, dtype=np.int64)
    for dims, in_ct in ((layout.ct_dims, True), (layout.slot_dims, False)):
        for piece, stride in zip(dims, _position_strides(dims)):
            if piece.dim != dim or piece.dim_type != DimType.FILL:
                continue
            digits = np.arange(piece.extent, dtype=np.int64)
            index = (index[:, None] + digits * piece.stride).ravel()
            ct = (ct[:, None] + digits * (stride if in_ct else 0)).ravel()
            slot = (slot[:, None] + digits * (0 if in_ct else stride)).ravel()
    out_ct = np.full(extent, -1, dtype=np.int64)
    out_slot = np.full(extent, -1, dtype=np.int64)
    # Written last to first, so the first copy of an index wins.
    keep = np.nonzero(index < extent)[0][::-1]
    out_ct[index[keep]] = ct[keep]
    out_slot[index[keep]] = slot[keep]
    return out_ct, out_slot


_INDEX_CACHE = {}


def position_indices(layout, shape):
    """The tensor index held at each position of `layout`.

    Returns:
        A ``(num_ct * n, len(shape))`` array, row ``ct * n + slot``. A row is
        -1 where the position holds no element (a gap, padding past `shape`)
        or holds a copy other than the first.
    """
    shape = tuple(int(s) for s in shape)
    key = (layout.layout_str(), int(layout.n), shape)
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    n = int(layout.n)
    positions = np.arange(layout.num_ct() * n, dtype=np.int64)
    index = np.zeros((positions.shape[0], len(shape)), dtype=np.int64)
    held = np.ones(positions.shape[0], dtype=bool)
    for dims, coord in (
        (layout.ct_dims, positions // n),
        (layout.slot_dims, positions % n),
    ):
        for piece, stride in zip(dims, _position_strides(dims)):
            digit = (coord // stride) % piece.extent
            if piece.dim is None:
                held &= digit == 0
            else:
                index[:, piece.dim] += digit * piece.stride
    held &= (index < np.asarray(shape, dtype=np.int64)).all(axis=1)
    index[~held] = -1
    if len(_INDEX_CACHE) >= 16:
        _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = index
    return index


def _label_terms(window, in_layout, z_layout, z_dim):
    """One label's share of a pair's group: every (``Z`` ciphertext, input
    ciphertext, slot shift) its indices contribute."""
    spec = window.spec
    label = window.z_labels[z_dim]
    z_ct, z_slot = dim_positions(z_layout, z_dim, window.z_shape[z_dim])
    relation = window.window_of(label)
    if label in spec.lhs:
        # The input holds the label: the same index on both sides.
        lhs_dim = spec.lhs.index(label)
        in_ct, in_slot = dim_positions(in_layout, lhs_dim, window.in_shape[lhs_dim])
        z_at = np.arange(window.z_shape[z_dim])
        in_at = z_at
    elif relation is not None:
        # The label is a window's output: one input index per tap in range.
        lhs_dim = spec.lhs.index(relation.label)
        in_ct, in_slot = dim_positions(in_layout, lhs_dim, window.in_shape[lhs_dim])
        out = np.arange(window.z_shape[z_dim])[:, None]
        tap = np.arange(window.extent(relation.offset))[None, :]
        source = relation.stride * out + tap + relation.const - window.pad(relation)
        in_range = (source >= 0) & (source < window.in_shape[lhs_dim])
        z_at = np.broadcast_to(out, source.shape)[in_range]
        in_at = source[in_range]
    else:
        # Only the filter holds the label.
        if (z_ct < 0).any():
            raise ValueError(f"layout does not hold every index of {label}")
        return np.stack([z_ct, np.zeros_like(z_ct), -z_slot], axis=1)
    if (z_ct[z_at] < 0).any() or (in_ct[in_at] < 0).any():
        raise ValueError(f"layout does not hold every index of {label}")
    return np.stack([z_ct[z_at], in_ct[in_at], in_slot[in_at] - z_slot[z_at]], axis=1)


def _unique_keys(keys, n, num_in_ct):
    keys[:, 2] %= n
    code = (keys[:, 0] * num_in_ct + keys[:, 1]) * n + keys[:, 2]
    code = np.unique(code)
    return np.stack(
        [code // (num_in_ct * n), (code // n) % num_in_ct, code % n], axis=1
    )


_KEY_CACHE = {}


def window_keys(window, in_layout, z_layout):
    """The groups of `window` under an input layout and a layout of ``Z``.

    Returns:
        A ``(groups, 3)`` array of (``Z`` ciphertext, input ciphertext, shift),
        sorted. ``Z`` slot ``s`` reads input slot ``(s + shift) mod n``.
    """
    key = (window, in_layout.layout_str(), z_layout.layout_str(), int(z_layout.n))
    if key in _KEY_CACHE:
        return _KEY_CACHE[key]
    spec = window.spec
    for label in spec.lhs:
        if label not in window.z_labels and spec.relation_of(label) is None:
            raise NotImplementedError(
                f"input label {label} is neither kept nor windowed"
            )
    n = int(z_layout.n)
    num_in_ct = max(int(in_layout.num_ct()), 1)
    keys = np.zeros((1, 3), dtype=np.int64)
    for z_dim in range(len(window.z_labels)):
        terms = _unique_keys(
            _label_terms(window, in_layout, z_layout, z_dim), n, num_in_ct
        )
        keys = (keys[:, None, :] + terms[None, :, :]).reshape(-1, 3)
        keys = _unique_keys(keys, n, num_in_ct)
    if len(_KEY_CACHE) >= 4096:
        _KEY_CACHE.clear()
    _KEY_CACHE[key] = keys
    return keys


def summed_pieces(window, z_layout):
    """The pieces of ``Z`` the output sums away.

    Returns:
        (ct_pieces, slot_pieces): each a list of (piece position in its block,
        extent, position stride)
    """
    summed = set(window.summed)
    blocks = []
    for dims in (z_layout.ct_dims, z_layout.slot_dims):
        blocks.append(
            [
                (at, piece.extent, stride)
                for at, (piece, stride) in enumerate(zip(dims, _position_strides(dims)))
                if piece.dim in summed and piece.dim_type == DimType.FILL
            ]
        )
    return blocks[0], blocks[1]


def output_ct(window, z_layout, z_ct):
    """The output ciphertext each ``Z`` ciphertext is summed into."""
    z_ct = np.asarray(z_ct, dtype=np.int64)
    summed = {at for at, _, _ in summed_pieces(window, z_layout)[0]}
    out = np.zeros_like(z_ct)
    dims = z_layout.ct_dims
    for at, (piece, stride) in enumerate(zip(dims, _position_strides(dims))):
        if at not in summed:
            out = out * piece.extent + (z_ct // stride) % piece.extent
    return out


def window_ops(window, in_layout, z_layout):
    """The HE operations of the kernel: ``{"mul", "rot", "add"}``.

    - one multiply per group, and one rotation per distinct (input ciphertext,
      nonzero shift): a rotated ciphertext is shared by every group reading it
    - one add per group beyond the first of its ``Z`` ciphertext, and per ``Z``
      ciphertext beyond the first of its output ciphertext
    - per output ciphertext, a rotate-and-sum over each summed slot piece, then
      one multiply by a mask that clears the slots the sum leaves behind
    """
    keys = window_keys(window, in_layout, z_layout)
    n = int(z_layout.n)
    z_cts = np.unique(keys[:, 0])
    out_cts = np.unique(output_ct(window, z_layout, z_cts)).shape[0]
    moved = keys[keys[:, 2] != 0]
    rotations = np.unique(moved[:, 1] * n + moved[:, 2]).shape[0]
    _, slot_pieces = summed_pieces(window, z_layout)
    steps = sum(int(math.ceil(math.log2(extent))) for _, extent, _ in slot_pieces)
    return {
        "mul": int(keys.shape[0]) + (out_cts if slot_pieces else 0),
        "rot": int(rotations) + out_cts * steps,
        "add": int(keys.shape[0]) - out_cts + out_cts * steps,
    }


class GroupWeights:
    """The plaintext of one group.

    Slot ``s`` of ``Z`` ciphertext `z_ct` reads slot ``(s + shift) mod n`` of
    input ciphertext `in_ct`. Where the two slots hold a ``Z`` element and an
    input element the operator relates, the plaintext holds the filter element
    that weights the pair, and 0 elsewhere.
    """

    def __init__(self, window, in_layout, z_layout, key):
        self.window = window
        self.in_layout = in_layout
        self.z_layout = z_layout
        self.z_ct, self.in_ct, self.shift = (int(k) for k in key)
        self.n = int(z_layout.n)

    def __repr__(self):
        w = self.window
        return (
            f"weights[{w.spec} {w.in_shape} {w.filter_shape} pad {w.pads} "
            f"{self.in_layout.layout_str()} -> {self.z_layout.layout_str()} "
            f"ct {self.z_ct} <- ct {self.in_ct} << {self.shift}]"
        )

    def entries(self):
        """The slots the plaintext fills and the filter index at each.

        Returns:
            (slots, index): `index` is one array per filter dim
        """
        window = self.window
        spec = window.spec
        n = self.n
        z = position_indices(self.z_layout, window.z_shape)
        z = z[self.z_ct * n : (self.z_ct + 1) * n]
        x = position_indices(self.in_layout, window.in_shape)
        x = np.roll(x[self.in_ct * n : (self.in_ct + 1) * n], -self.shift, axis=0)

        related = (z[:, 0] >= 0) & (x[:, 0] >= 0)
        values = {label: z[:, i] for i, label in enumerate(window.z_labels)}
        for lhs_dim, label in enumerate(spec.lhs):
            if label in values:
                related &= x[:, lhs_dim] == values[label]
                continue
            relation = spec.relation_of(label)
            tap = (
                x[:, lhs_dim]
                - relation.stride * values[relation.base]
                - relation.const
                + window.pad(relation)
            )
            related &= (tap >= 0) & (tap < window.extent(relation.offset))
            values[relation.offset] = tap
        index = tuple(values[label][related] for label in spec.rhs)
        return np.nonzero(related)[0], index
