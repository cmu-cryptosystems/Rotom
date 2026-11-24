"""
Operator specs and dimension alignment for tensor operations.

An operator is described once, einsum-style, by the index labels of its
operands and of its output: matmul is ``ik,kj->ij``, a batched matmul is
``bik,kj->bij``, a sum over the last dim is ``ab->a``. A label is any string;
single letters are the shorthand that ``OperatorSpec.parse`` reads, and
operators with many roles use names (``c_in``, ``h``, ``kh``, ``h_out``).
Everything the layout assignment needs to know about the operator's structure
is derived from those labels: which operand dims align with each other, which
are summed away, and which output dim each label becomes.

Labels relate indices by equality. Indices that are related arithmetically are
described with ``Relation`` entries on top of the labels. A relation defines
one lhs label from other labels:

    label = (stride * base + offset + const) mod modulus

- a convolution window: ``h = stride * h_out + kh``
- a reshape that splits a dim: ``d1 = 64 * d1.0 + d1.1``
- an index or a slice: ``d0 = 3`` or ``d0 = 2 * d0_out + 1``
- a roll: ``k = (j + diag) mod n``

Key items:
- OperatorSpec: the spec, parsed from an einsum string plus relations
- spec_for_term: the spec of a TensorTerm
- get_dim_alignment: the (lhs dim, rhs dim) alignment set of a binary operator
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from frontends.tensor import TensorOp
from frontends.tensor_args import Conv2dArgs, Conv3dArgs, ReshapeArgs


@dataclass(frozen=True)
class Relation:
    """Defines an lhs label from other labels.

    ``label = (stride * base + offset + const) mod modulus``. ``base`` and
    ``offset`` are labels or None, and ``modulus`` is None when the index does
    not wrap. ``label`` is in the lhs only: not in the rhs and not in the
    output.
    """

    label: str
    base: Optional[str] = None
    offset: Optional[str] = None
    stride: int = 1
    const: int = 0
    modulus: Optional[int] = None

    def terms(self) -> Tuple[str, ...]:
        """The labels this relation reads."""
        return tuple(t for t in (self.base, self.offset) if t is not None)

    def __str__(self):
        parts = []
        if self.base is not None:
            parts.append(
                self.base if self.stride == 1 else f"{self.stride}*{self.base}"
            )
        if self.offset is not None:
            parts.append(self.offset)
        if self.const or not parts:
            parts.append(str(self.const))
        text = "+".join(parts)
        if self.modulus is not None:
            text = f"({text}) mod {self.modulus}"
        return f"{self.label}={text}"


@dataclass(frozen=True)
class OperatorSpec:
    """Index labels of an operator's lhs, rhs and output.

    ``rhs`` is None for a unary operator. A label is one of:
    - carried:    in the output and in one or both operands (same index)
    - contracted: in both operands and not in the output (summed away)
    - reduced:    in a unary operator's operand and not in the output
    - related:    an lhs label defined by a ``Relation``
    - a relation term: a label a relation reads, found in the rhs (a window
      tap), in the output (a window, split or slice output), or in both
    - a unit: the label of an operand axis of extent 1 that the other operand
      broadcasts over. No layout holds it, so it aligns with nothing.
    """

    lhs: Tuple[str, ...]
    rhs: Optional[Tuple[str, ...]]
    out: Tuple[str, ...]
    relations: Tuple[Relation, ...] = ()
    units: Tuple[str, ...] = ()

    def __post_init__(self):
        rhs = self.rhs or ()
        for side, labels in (("lhs", self.lhs), ("rhs", rhs), ("out", self.out)):
            if len(set(labels)) != len(labels):
                raise ValueError(f"repeated label in {side}: {' '.join(labels)}")

        related = [r.label for r in self.relations]
        if len(set(related)) != len(related):
            raise ValueError("a label is defined by more than one relation")
        terms = set()
        for r in self.relations:
            if r.stride < 1:
                raise ValueError(f"relation stride must be positive: {r.stride}")
            if r.modulus is not None and r.modulus < 1:
                raise ValueError(f"relation modulus must be positive: {r.modulus}")
            if r.label not in self.lhs or r.label in rhs or r.label in self.out:
                raise ValueError(f"related label {r.label!r} must be lhs-only")
            for term in r.terms():
                if term in self.lhs:
                    raise ValueError(f"relation term {term!r} is an lhs label")
                if term not in rhs and term not in self.out:
                    raise ValueError(f"relation term {term!r} is in no operand")
                if term in terms:
                    raise ValueError(f"label {term!r} is read by two relations")
                terms.add(term)

        for label in self.units:
            sides = (label in self.lhs) + (label in rhs)
            if sides != 1 or label in self.out or label in related or label in terms:
                raise ValueError(f"unit label {label!r} must be in one operand only")

        for label in self.out:
            if label not in self.lhs and label not in rhs and label not in terms:
                raise ValueError(f"output label {label!r} has no source")
        if self.rhs is None:
            return
        held = set(self.out) | set(self.units)
        for label in self.lhs:
            if label not in rhs and label not in held and label not in related:
                raise ValueError(f"lhs label {label!r} is summed over one operand")
        for label in rhs:
            if label not in self.lhs and label not in held and label not in terms:
                raise ValueError(f"rhs label {label!r} is summed over one operand")

    @classmethod
    def parse(cls, spec: str, relations: Sequence[Relation] = ()) -> "OperatorSpec":
        """Parses ``"ik,kj->ij"`` or the unary ``"ab->a"``; one character per label."""
        operands, arrow, out = spec.partition("->")
        if not arrow:
            raise ValueError(f"expected 'lhs,rhs->out' or 'lhs->out': {spec!r}")
        lhs, comma, rhs = operands.partition(",")
        return cls(
            tuple(lhs), tuple(rhs) if comma else None, tuple(out), tuple(relations)
        )

    def __str__(self):
        # Single-character labels print as an einsum string; names are spaced.
        labels = self.lhs + (self.rhs or ()) + self.out
        short = all(len(label) == 1 for label in labels)
        sep = "" if short else " "
        comma, arrow = (",", "->") if short else (", ", " -> ")
        text = sep.join(self.lhs)
        if self.rhs is not None:
            text += comma + sep.join(self.rhs)
        text += arrow + sep.join(self.out)
        for r in self.relations:
            text += f" [{r}]"
        return text

    @property
    def is_unary(self) -> bool:
        return self.rhs is None

    def accepts(self, shapes) -> bool:
        """True when the operand ranks match the spec."""
        if self.rhs is None:
            return len(shapes) == 1 and len(shapes[0]) == len(self.lhs)
        return (
            len(shapes) == 2
            and len(shapes[0]) == len(self.lhs)
            and len(shapes[1]) == len(self.rhs)
        )

    def relation_of(self, label: str) -> Optional[Relation]:
        """The relation that defines `label` or reads it."""
        for r in self.relations:
            if label == r.label or label in r.terms():
                return r
        return None

    def alignment(self):
        """The set of (lhs dim, rhs dim) pairs; None where the other side has no dim.

        A label in both operands aligns its two dims. A label in one operand
        aligns with replication on the other side. A unit has no dim.
        """
        if self.rhs is None:
            raise ValueError(f"a unary operator has no alignment: {self}")
        pairs = set()
        for a_dim, label in enumerate(self.lhs):
            if label in self.units:
                continue
            pairs.add((a_dim, self.rhs.index(label) if label in self.rhs else None))
        for b_dim, label in enumerate(self.rhs):
            if label not in self.lhs and label not in self.units:
                pairs.add((None, b_dim))
        return pairs

    def contracted(self) -> Tuple[str, ...]:
        """Labels in both operands that are summed away."""
        rhs = self.rhs or ()
        return tuple(
            label for label in self.lhs if label in rhs and label not in self.out
        )

    def reduced(self) -> Tuple[str, ...]:
        """A unary operator's labels that are reduced over: not in the output
        and not defined by a relation."""
        if self.rhs is not None:
            return ()
        related = {r.label for r in self.relations}
        return tuple(
            label
            for label in self.lhs
            if label not in self.out and label not in related
        )

    def lhs_sum_dims(self) -> Tuple[int, ...]:
        """Lhs dims summed away: the contracted or reduced labels."""
        return tuple(
            self.lhs.index(label) for label in self.contracted() + self.reduced()
        )

    def rhs_sum_dims(self) -> Tuple[int, ...]:
        """Rhs dims summed away: the contracted labels, then the relation terms
        the output does not keep (window taps)."""
        if self.rhs is None:
            return ()
        dims = [self.rhs.index(label) for label in self.contracted()]
        for r in self.relations:
            dims += [
                self.rhs.index(t)
                for t in r.terms()
                if t in self.rhs and t not in self.out
            ]
        return tuple(dims)

    def out_dim(self, label: str) -> Optional[int]:
        """The output dim an operand label becomes, or None if it has none.

        A related label becomes the output dim of its relation's base (a
        window's, a slice's or a split's outer output), when the output keeps
        that base.
        """
        if label in self.out:
            return self.out.index(label)
        for r in self.relations:
            if label == r.label and r.base in self.out:
                return self.out.index(r.base)
        return None

    def with_rolls(self, lhs_pieces, rhs_pieces, lhs_rolls=(), rhs_rolls=()):
        """This spec over the pieces of two rolled layouts.

        Args:
            lhs_pieces, rhs_pieces: each layout's pieces in order, as
                `(dim, extent, stride)`; `dim` is None for replication
            lhs_rolls, rhs_rolls: each layout's rolls, as `(piece, by)` piece
                positions

        Returns:
            RolledSpec: the traversal labels of each piece and the relations
            that recover the tensor indices
        """
        if self.rhs is None:
            raise ValueError(f"a unary operator has one layout: {self}")
        lhs_pieces, rhs_pieces = tuple(lhs_pieces), tuple(rhs_pieces)
        cut_rolls = []
        paired = paired_pieces(lhs_pieces, rhs_pieces)
        faces = paired is not None
        if faces:
            # Layouts of one size face each other position by position, once
            # each piece is cut where the other layout's pieces begin.
            lhs_pieces, rhs_pieces, lhs_at, rhs_at = paired
            lhs_rolls = _moved_rolls("lhs", lhs_rolls, lhs_at, cut_rolls)
            rhs_rolls = _moved_rolls("rhs", rhs_rolls, rhs_at, cut_rolls)
        lhs = _Side(lhs_pieces, self.lhs, lhs_rolls)
        rhs = _Side(rhs_pieces, self.rhs, rhs_rolls)
        if faces:
            lhs.other, rhs.other = rhs, lhs

        contracted = set(self.contracted())
        relations = []
        rolled_by_contracted = []
        named = []
        for side in (lhs, rhs):
            rolls = []
            for at, (piece, by) in enumerate(side.rolls):
                digit = side.digits[piece] or "_"
                by_label, by_mentions = side.value(by, at)
                rolls.append((digit, by_label))
                if _base(digit) not in contracted and by_mentions & contracted:
                    if side.digits[piece] is not None:
                        rolled_by_contracted.append(side.label(piece)[0])
                later = [p for p, _ in side.rolls[at + 1 :]]
                diagonal = (
                    side.label(piece)[0] if piece not in later else f"{digit}'{at}"
                )
                earlier = [i for i, (p, _) in enumerate(side.rolls[:at]) if p == piece]
                rolled = digit if not earlier else f"{digit}'{earlier[-1]}"
                relation = Relation(
                    rolled, by_label, diagonal, modulus=side.pieces[piece][1]
                )
                if relation not in relations:
                    relations.append(relation)
            named.append(tuple(rolls))
            relations += [r for r in side.relations if r not in relations]
        return RolledSpec(
            self,
            tuple(lhs.label(p)[0] for p in range(len(lhs_pieces))),
            tuple(rhs.label(p)[0] for p in range(len(rhs_pieces))),
            tuple(relations),
            faces,
            named[0],
            named[1],
            tuple(dict.fromkeys(rolled_by_contracted)),
            tuple(cut_rolls),
        )

    def lhs_out_dims(self) -> Tuple[Optional[int], ...]:
        return tuple(self.out_dim(label) for label in self.lhs)

    def rhs_out_dims(self) -> Tuple[Optional[int], ...]:
        return tuple(self.out_dim(label) for label in self.rhs or ())


@dataclass(frozen=True)
class RolledSpec:
    """An operator spec read over the pieces of two rolled layouts.

    A layout stores a tensor as an ordered list of pieces, and a roll adds the
    index of one piece to another: after ``roll(u by t)``, slot ``(d, t)``
    holds the element ``u = (t + d) mod extent``. The rolled piece is
    therefore traversed by a new index ``d``, the diagonal, and the tensor
    index is recovered by a relation.

    Two layouts of one size face each other position by position, so their
    pieces are first cut where the other layout's pieces begin
    (``paired_pieces``): ``[j:2][i:4]`` facing ``[R:8]`` reads the replication
    as ``[R:2][R:4]``. A roll is a roll of whole pieces, so one over a piece
    that had to be cut has no counterpart on the other layout.

    ``lhs`` and ``rhs`` hold one traversal label per piece, after the cuts:
    - an unrolled piece of a whole dim keeps the dim's label (``k``)
    - a strided piece or a piece of a split dim is a digit, named by its
      stride (``k@4``); the digits recover the dim through a chain of relations
    - a rolled piece is a diagonal, named ``(u-t)`` after the label of the
      piece it was rolled by at the time of the roll
    - an unrolled replication piece takes the label it faces on the other
      operand, or ``_`` when it faces none

    Two rolled layouts are aligned when the operands share one traversal: each
    position holds the same label on both, no label the output keeps is
    rolled by a contracted one, and no roll is over a piece that was cut.
    """

    spec: "OperatorSpec"
    lhs: Tuple[str, ...]
    rhs: Tuple[str, ...]
    relations: Tuple[Relation, ...]
    faces: bool
    lhs_rolls: Tuple[Tuple[str, str], ...]
    rhs_rolls: Tuple[Tuple[str, str], ...]
    output_rolled_by_contracted: Tuple[str, ...]
    cut_rolls: Tuple[str, ...] = ()

    def __str__(self):
        text = f"{' '.join(self.lhs)}, {' '.join(self.rhs)} -> "
        text += " ".join(self.spec.out)
        for r in self.relations:
            text += f" [{r}]"
        return text

    def mismatches(self) -> Tuple[str, ...]:
        """Why the two rolled layouts do not share a traversal; empty if they do."""
        why = [
            f"{label} is rolled by a contracted label"
            for label in self.output_rolled_by_contracted
        ]
        why += [
            f"{roll} is over a piece the other layout holds in parts"
            for roll in self.cut_rolls
        ]
        if self.faces:
            for position, (a, b) in enumerate(zip(self.lhs, self.rhs)):
                if a != b and "_" not in (a, b):
                    why.append(f"piece {position} is {a} on the lhs and {b} on the rhs")
            return tuple(why)
        # The layouts differ in size, so positions do not face each other:
        # compare the rolls of the labels both operands hold.
        sides = (
            ("lhs", self.lhs_rolls, self.rhs_rolls, self.spec.rhs or ()),
            ("rhs", self.rhs_rolls, self.lhs_rolls, self.spec.lhs),
        )
        for side, mine, theirs, other_labels in sides:
            for u, t in mine:
                if _base(u) in other_labels and (u, t) not in theirs:
                    why.append(f"{side} roll({u} by {t}) is not rolled on the other")
        return tuple(why)

    def aligned(self) -> bool:
        return not self.mismatches()


def paired_pieces(lhs_pieces, rhs_pieces):
    """Two layouts' pieces, cut where the other's begin, so that they pair off.

    ``[j:2][i:4]`` and ``[R:8]`` hold the same eight positions in two pieces
    and in one; cutting the second into ``[R:2][R:4]`` pairs every piece with
    the piece at its positions. The outer part of a cut piece keeps the
    coarser stride: ``[i:8:1]`` cut at 4 is ``[i:2:4][i:4:1]``.

    Args:
        lhs_pieces, rhs_pieces: each layout's pieces in order, as
            `(dim, extent, stride)`

    Returns:
        (lhs, rhs, lhs_at, rhs_at): the cut pieces of each layout, equal in
        number and extents, and for each piece given its position among them,
        or None if it was cut. None if the layouts differ in size.
    """
    rest = [list(lhs_pieces), list(rhs_pieces)]
    whole = [[True] * len(lhs_pieces), [True] * len(rhs_pieces)]
    index = [0, 0]
    cut = ([], [])
    at = ([None] * len(lhs_pieces), [None] * len(rhs_pieces))
    while rest[0] and rest[1]:
        extent = min(rest[0][0][1], rest[1][0][1])
        for side in (0, 1):
            dim, held, stride = rest[side][0]
            if held == extent:
                if whole[side][index[side]]:
                    at[side][index[side]] = len(cut[side])
                cut[side].append((dim, held, stride))
                rest[side].pop(0)
                index[side] += 1
            else:
                inner = held // extent
                cut[side].append((dim, extent, stride * inner))
                rest[side][0] = (dim, inner, stride)
                whole[side][index[side]] = False
    if rest[0] or rest[1]:
        return None
    return tuple(cut[0]), tuple(cut[1]), tuple(at[0]), tuple(at[1])


def _moved_rolls(side, rolls, at, cut_rolls):
    """`rolls` on the positions of the cut pieces.

    A roll over a piece that was cut is dropped and described in `cut_rolls`.
    """
    moved = []
    for piece, by in rolls:
        if at[piece] is None or at[by] is None:
            cut_rolls.append(f"{side} roll({piece} by {by})")
        else:
            moved.append((at[piece], at[by]))
    return moved


def _base(label: str) -> str:
    """The spec label a digit label is a digit of."""
    return label.partition("@")[0]


class _Side:
    """The pieces and rolls of one operand's layout."""

    def __init__(self, pieces, own, rolls):
        self.pieces = pieces
        self.rolls = tuple(rolls)
        strides = {}
        for dim, _, stride in pieces:
            if dim is not None:
                strides.setdefault(dim, []).append(stride)
        self.digits = []
        for dim, _, stride in pieces:
            if dim is None:
                self.digits.append(None)
            elif strides[dim] == [1]:
                self.digits.append(own[dim])
            else:
                self.digits.append(f"{own[dim]}@{stride}")
        self.relations = []
        for dim, dim_strides in strides.items():
            if dim_strides != [1]:
                self.relations += _digit_relations(own[dim], dim_strides)
        self.other = None
        self._labels = {}

    def label(self, piece):
        """The traversal label of a piece, and the spec labels it mentions."""
        if piece in self._labels:
            return self._labels[piece]
        self._labels[piece] = ("_", frozenset())  # a cycle resolves to "_"
        digit = self.digits[piece]
        rolls = [(i, by) for i, (p, by) in enumerate(self.rolls) if p == piece]
        if digit is None and not rolls:
            faces = self.other is not None and piece < len(self.other.pieces)
            result = self.other.label(piece) if faces else ("_", frozenset())
        else:
            text = digit or "_"
            mentions = frozenset([_base(digit)] if digit else [])
            if rolls:
                by = [self.value(t, i) for i, t in rolls]
                text = f"({text}-{'-'.join(b[0] for b in by)})"
                mentions = mentions.union(*(b[1] for b in by))
            result = (text, mentions)
        self._labels[piece] = result
        return result

    def value(self, piece, at):
        """What a piece's index holds when roll number `at` reads it."""
        own = [i for i, (p, _) in enumerate(self.rolls) if p == piece]
        if all(i > at for i in own):
            # Not rolled yet: its index is still its traversal index.
            return self.label(piece)
        digit = self.digits[piece]
        if all(i < at for i in own) and digit is not None:
            # Fully rolled: its index is the tensor index again.
            return (digit, frozenset([_base(digit)]))
        return ("_", frozenset())


def _digit_relations(label, strides):
    """Mixed-radix digits, outermost first: k = s0*k@s0 + (s1*k@s1 + ...)."""
    ordered = sorted(strides, reverse=True)
    relations = []
    rest = label
    for i, stride in enumerate(ordered[:-1]):
        if i == len(ordered) - 2 and ordered[-1] == 1:
            below = f"{label}@1"
        else:
            below = f"{label}<{stride}"
        relations.append(Relation(rest, f"{label}@{stride}", below, stride=stride))
        rest = below
    if ordered[-1] != 1:
        relations.append(Relation(rest, f"{label}@{ordered[-1]}", stride=ordered[-1]))
    return relations


def _dims(rank: int) -> Tuple[str, ...]:
    return tuple(f"d{i}" for i in range(rank))


def elementwise_spec(lhs_shape, rhs_shape) -> OperatorSpec:
    """Elementwise op with broadcasting: dims align from the right, and an
    axis of extent 1 broadcasts over the other operand's.

    Args:
        lhs_shape, rhs_shape: the operand shapes, or their ranks when no axis
            broadcasts

    ``[C, T, H, W] * [C, T, 1, 1]`` is ``d0 d1 d2 d3, d0 d1 u2 u3 -> d0 d1 d2
    d3`` with units ``u2`` and ``u3``.
    """
    shapes = [
        (None,) * shape if isinstance(shape, int) else tuple(shape)
        for shape in (lhs_shape, rhs_shape)
    ]
    rank = max(len(shape) for shape in shapes)
    out = _dims(rank)
    operands = [list(out[rank - len(shape) :]) for shape in shapes]
    units = []
    for side, other in ((0, 1), (1, 0)):
        for i in range(1, len(shapes[side]) + 1):
            if i > len(shapes[other]):
                break
            if shapes[side][-i] == 1 and shapes[other][-i] != 1:
                operands[side][-i] = f"u{rank - i}"
                units.append(operands[side][-i])
    return OperatorSpec(
        tuple(operands[0]), tuple(operands[1]), out, units=tuple(sorted(units))
    )


def matmul_spec(lhs_rank: int, rhs_rank: int) -> OperatorSpec:
    """Matmul: the last lhs dim contracts with the first rhs dim."""
    specs = {
        (2, 1): "ik,k->i",
        (1, 2): "k,kj->j",
        (2, 2): "ik,kj->ij",
        (3, 2): "bik,kj->bij",
    }
    if (lhs_rank, rhs_rank) not in specs:
        raise NotImplementedError("alignment not implemented")
    return OperatorSpec.parse(specs[(lhs_rank, rhs_rank)])


def conv_spec(strides: Sequence[int], depthwise: bool = False) -> OperatorSpec:
    """Convolution over ``len(strides)`` spatial axes, one stride per axis.

    Input is ``[C_in, *spatial]``, filter is ``[C_out, C_in, *taps]`` and
    output is ``[C_out, *spatial_out]``. The input channel is contracted. For
    depthwise, the filter is ``[*taps, C]`` (TFLite's ``[1, K_h, K_w, C]``
    without its unit axis) and the channel is carried.

    conv2d is ``c_in h w, c_out c_in kh kw -> c_out h_out w_out`` with
    ``h = s*h_out + kh`` and ``w = s*w_out + kw``.
    """
    if not 1 <= len(strides) <= 3:
        raise NotImplementedError(f"conv over {len(strides)} spatial axes")
    spatial = ("d", "h", "w")[-len(strides) :]
    taps = tuple(f"k{axis}" for axis in spatial)
    outs = tuple(f"{axis}_out" for axis in spatial)
    windows = tuple(
        Relation(label=i, base=o, offset=t, stride=s)
        for i, o, t, s in zip(spatial, outs, taps, strides)
    )
    if depthwise:
        return OperatorSpec(("c",) + spatial, taps + ("c",), ("c",) + outs, windows)
    return OperatorSpec(
        ("c_in",) + spatial, ("c_out", "c_in") + taps, ("c_out",) + outs, windows
    )


def identity_spec(rank: int) -> OperatorSpec:
    """A slot-wise function of one tensor."""
    return OperatorSpec(_dims(rank), None, _dims(rank))


def permute_spec(perm) -> OperatorSpec:
    """Input dim ``i`` becomes output dim ``perm[i]``."""
    lhs = _dims(len(perm))
    out = [None] * len(perm)
    for i, label in enumerate(lhs):
        out[perm[i]] = label
    if None in out:
        raise ValueError(f"not a permutation: {perm}")
    return OperatorSpec(lhs, None, tuple(out))


def reduce_spec(rank: int, dim: int) -> OperatorSpec:
    """A sum or product over one dim, which the output drops."""
    lhs = _dims(rank)
    return OperatorSpec(lhs, None, lhs[:dim] + lhs[dim + 1 :])


def split_spec(rank: int, dim: int, outer: int, inner: int) -> OperatorSpec:
    """A reshape that splits ``dim`` into an outer and an inner dim.

    Splitting ``d1`` of extent 768 into (12, 64) is ``d1 = 64*d1.0 + d1.1``.
    """
    if outer < 1 or inner < 1:
        raise ValueError(f"split extents must be positive: {outer}, {inner}")
    lhs = _dims(rank)
    parts = (f"{lhs[dim]}.0", f"{lhs[dim]}.1")
    relation = Relation(label=lhs[dim], base=parts[0], offset=parts[1], stride=inner)
    return OperatorSpec(lhs, None, lhs[:dim] + parts + lhs[dim + 1 :], (relation,))


def index_spec(rank: int, index) -> OperatorSpec:
    """Indexing: an int fixes a dim and drops it, a slice keeps it."""
    items = index if isinstance(index, tuple) else (index,)
    if len(items) > rank:
        raise ValueError(f"too many indices for rank {rank}: {index}")
    lhs = _dims(rank)
    out, relations = [], []
    for dim, label in enumerate(lhs):
        item = items[dim] if dim < len(items) else slice(None)
        if isinstance(item, int):
            if item < 0:
                raise NotImplementedError(f"negative index: {item}")
            relations.append(Relation(label=label, const=item))
        elif isinstance(item, slice):
            start = 0 if item.start is None else item.start
            step = 1 if item.step is None else item.step
            if step <= 0 or start < 0:
                raise NotImplementedError(f"slice not supported: {item}")
            if start == 0 and step == 1:
                # A slice that only trims the end reads the same indices.
                out.append(label)
            else:
                kept = f"{label}_out"
                relations.append(
                    Relation(label=label, base=kept, stride=step, const=start)
                )
                out.append(kept)
        else:
            raise NotImplementedError(f"index not supported: {item!r}")
    return OperatorSpec(lhs, None, tuple(out), tuple(relations))


def spec_for_term(term, shapes) -> OperatorSpec:
    """The operator spec of a TensorTerm, given its operand shapes.

    Raises:
        NotImplementedError: If the operation has no spec
    """
    match term.op:
        case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
            return elementwise_spec(shapes[0], shapes[1])
        case TensorOp.MATMUL:
            return matmul_spec(len(shapes[0]), len(shapes[1]))
        case TensorOp.CONV2D:
            args = Conv2dArgs.from_term(term)
            if args.groups not in (1, "depthwise"):
                raise NotImplementedError(f"conv2d groups: {args.groups}")
            return conv_spec(args.strides, depthwise=args.depthwise)
        case TensorOp.CONV3D:
            args = Conv3dArgs.from_term(term)
            return conv_spec(args.strides, depthwise=args.depthwise)
        case (
            TensorOp.POLY_CALL
            | TensorOp.HARD_SWISH
            | TensorOp.CAST
            | TensorOp.RESCALE
        ):
            return identity_spec(len(shapes[0]))
        case TensorOp.TRANSPOSE:
            return permute_spec({0: 1, 1: 0})
        case TensorOp.PERMUTE:
            return permute_spec(term.cs[1])
        case TensorOp.SUM | TensorOp.PRODUCT:
            return reduce_spec(len(shapes[0]), term.cs[1])
        case TensorOp.RESHAPE:
            args = ReshapeArgs.from_term(term)
            if list(args.shape) == [args.dim]:
                # One part: the dim keeps its place.
                return identity_spec(len(shapes[0]))
            # The term replaces `dim` by new dim ids. It is a split when those
            # ids are `dim` and `dim + 1` and no later input dim keeps an id
            # they take.
            if (
                sorted(args.shape) != [args.dim, args.dim + 1]
                or args.dim != len(shapes[0]) - 1
            ):
                raise NotImplementedError(f"reshape is not a split: {args.shape}")
            return split_spec(
                len(shapes[0]),
                args.dim,
                args.shape[args.dim],
                args.shape[args.dim + 1],
            )
        case TensorOp.INDEX:
            index = term.cs[1] if isinstance(term.cs[1], tuple) else (term.cs[1],)
            # An int counts from the end when negative.
            index = tuple(
                item + shapes[0][dim] if isinstance(item, int) and item < 0 else item
                for dim, item in enumerate(index)
            )
            return index_spec(len(shapes[0]), index)
        case _:
            raise NotImplementedError(f"alignment: {term.op}")


def get_dim_alignment(term, shapes):
    """Determines dimension alignment for a binary tensor operation.

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
    return spec_for_term(term, shapes).alignment()
