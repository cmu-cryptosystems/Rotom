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
    """

    lhs: Tuple[str, ...]
    rhs: Optional[Tuple[str, ...]]
    out: Tuple[str, ...]
    relations: Tuple[Relation, ...] = ()

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

        for label in self.out:
            if label not in self.lhs and label not in rhs and label not in terms:
                raise ValueError(f"output label {label!r} has no source")
        if self.rhs is None:
            return
        for label in self.lhs:
            if label not in rhs and label not in self.out and label not in related:
                raise ValueError(f"lhs label {label!r} is summed over one operand")
        for label in rhs:
            if label not in self.lhs and label not in self.out and label not in terms:
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
        aligns with replication on the other side.
        """
        if self.rhs is None:
            raise ValueError(f"a unary operator has no alignment: {self}")
        pairs = set()
        for a_dim, label in enumerate(self.lhs):
            pairs.add((a_dim, self.rhs.index(label) if label in self.rhs else None))
        for b_dim, label in enumerate(self.rhs):
            if label not in self.lhs:
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

    def lhs_out_dims(self) -> Tuple[Optional[int], ...]:
        return tuple(self.out_dim(label) for label in self.lhs)

    def rhs_out_dims(self) -> Tuple[Optional[int], ...]:
        return tuple(self.out_dim(label) for label in self.rhs or ())


def _dims(rank: int) -> Tuple[str, ...]:
    return tuple(f"d{i}" for i in range(rank))


def elementwise_spec(lhs_rank: int, rhs_rank: int) -> OperatorSpec:
    """Elementwise op with rank broadcasting: dims align from the right."""
    rank = max(lhs_rank, rhs_rank)
    out = _dims(rank)
    return OperatorSpec(out[rank - lhs_rank :], out[rank - rhs_rank :], out)


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
    depthwise, the filter is ``[C, *taps]`` and the channel is carried.

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
        return OperatorSpec(("c",) + spatial, ("c",) + taps, ("c",) + outs, windows)
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
            return elementwise_spec(len(shapes[0]), len(shapes[1]))
        case TensorOp.MATMUL:
            return matmul_spec(len(shapes[0]), len(shapes[1]))
        case TensorOp.CONV2D:
            args = Conv2dArgs.from_term(term)
            # Depthwise terms carry a TFLite-style [1, Kh, Kw, C] filter, which
            # is not the [C, Kh, Kw] filter of conv_spec(depthwise=True).
            if args.groups != 1:
                raise NotImplementedError(f"conv2d groups: {args.groups}")
            return conv_spec([args.stride] * 2)
        case TensorOp.CONV3D:
            return conv_spec([Conv3dArgs.from_term(term).stride] * 3)
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
            return index_spec(len(shapes[0]), term.cs[1])
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
