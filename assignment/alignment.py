"""
Operator specs and dimension alignment for tensor operations.

An operator is described once, einsum-style, by the index labels of its two
operands and of its output: matmul is ``ik,kj->ij``, block matmul is
``bik,bkj->bij``. A label is any string; single letters are the shorthand that
``OperatorSpec.parse`` reads, and operators with many roles use names
(``c_in``, ``h``, ``kh``, ``h_out``). Everything the layout
assignment needs to know about the operator's structure is derived from those
labels: which operand dims align with each other, which are contracted
(summed away), and which output dim each label becomes.

Labels relate indices by equality. Operators that relate indices
arithmetically are described with ``Window`` entries on top of the labels: a
convolution's output index ``h_out`` reads input index ``stride * h_out + kh``, so the
output label, the input label and the filter-tap label are three different
labels tied together by one window.

Key items:
- OperatorSpec: the spec, parsed from an einsum string plus windows
- spec_for_term: the spec of a TensorTerm
- get_dim_alignment: the (lhs dim, rhs dim) alignment set derived from the spec
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

from frontends.tensor import TensorOp
from frontends.tensor_args import Conv2dArgs, Conv3dArgs


@dataclass(frozen=True)
class Window:
    """Ties an output label to an input label through a filter-tap label.

    Output index ``o`` reads input index ``stride * o + tap`` (before padding).
    ``out`` appears only in the output, ``inp`` only in the lhs, and ``tap``
    only in the rhs.
    """

    out: str
    inp: str
    tap: str
    stride: int = 1


@dataclass(frozen=True)
class OperatorSpec:
    """Index labels of a binary operator's lhs, rhs and output.

    A label is one of:
    - carried:    in the output and in one or both operands (same index)
    - contracted: in both operands and not in the output (summed away)
    - window in:  in the lhs only, read at ``stride * out + tap``
    - window tap: in the rhs only, summed away over the window
    - window out: in the output only, produced by a window
    """

    lhs: Tuple[str, ...]
    rhs: Tuple[str, ...]
    out: Tuple[str, ...]
    windows: Tuple[Window, ...] = ()

    def __post_init__(self):
        for side, labels in (("lhs", self.lhs), ("rhs", self.rhs), ("out", self.out)):
            if len(set(labels)) != len(labels):
                raise ValueError(f"repeated label in {side}: {''.join(labels)}")

        win_out = [w.out for w in self.windows]
        win_inp = [w.inp for w in self.windows]
        win_tap = [w.tap for w in self.windows]
        for names in (win_out, win_inp, win_tap):
            if len(set(names)) != len(names):
                raise ValueError("a label is used by more than one window")
        for w in self.windows:
            if w.stride < 1:
                raise ValueError(f"window stride must be positive: {w.stride}")
            if w.out not in self.out or w.out in self.lhs or w.out in self.rhs:
                raise ValueError(f"window output label {w.out!r} must be output-only")
            if w.inp not in self.lhs or w.inp in self.rhs or w.inp in self.out:
                raise ValueError(f"window input label {w.inp!r} must be lhs-only")
            if w.tap not in self.rhs or w.tap in self.lhs or w.tap in self.out:
                raise ValueError(f"window tap label {w.tap!r} must be rhs-only")

        for label in self.out:
            if label not in self.lhs and label not in self.rhs and label not in win_out:
                raise ValueError(f"output label {label!r} has no source")
        for label in self.lhs:
            if label not in self.rhs and label not in self.out and label not in win_inp:
                raise ValueError(f"lhs label {label!r} is summed over one operand")
        for label in self.rhs:
            if label not in self.lhs and label not in self.out and label not in win_tap:
                raise ValueError(f"rhs label {label!r} is summed over one operand")

    @classmethod
    def parse(cls, spec: str, windows: Sequence[Window] = ()) -> "OperatorSpec":
        """Parses ``"ik,kj->ij"``; each character is one label."""
        operands, arrow, out = spec.partition("->")
        lhs, comma, rhs = operands.partition(",")
        if not arrow or not comma:
            raise ValueError(f"expected 'lhs,rhs->out': {spec!r}")
        return cls(tuple(lhs), tuple(rhs), tuple(out), tuple(windows))

    def __str__(self):
        # Single-character labels print as an einsum string; names are spaced.
        short = all(len(label) == 1 for label in self.lhs + self.rhs + self.out)
        sep = "" if short else " "
        comma, arrow = (",", "->") if short else (", ", " -> ")
        text = (
            sep.join(self.lhs) + comma + sep.join(self.rhs) + arrow + sep.join(self.out)
        )
        for w in self.windows:
            text += f" [{w.out}={w.stride}*{w.inp}+{w.tap}]"
        return text

    def accepts(self, shapes) -> bool:
        """True when the operand ranks match the spec."""
        return len(shapes[0]) == len(self.lhs) and len(shapes[1]) == len(self.rhs)

    def alignment(self):
        """The set of (lhs dim, rhs dim) pairs; None where the other side has no dim.

        A label in both operands aligns its two dims. A label in one operand
        aligns with replication on the other side.
        """
        pairs = set()
        for a_dim, label in enumerate(self.lhs):
            pairs.add((a_dim, self.rhs.index(label) if label in self.rhs else None))
        for b_dim, label in enumerate(self.rhs):
            if label not in self.lhs:
                pairs.add((None, b_dim))
        return pairs

    def contracted(self) -> Tuple[str, ...]:
        """Labels in both operands that are summed away."""
        return tuple(
            label for label in self.lhs if label in self.rhs and label not in self.out
        )

    def lhs_sum_dims(self) -> Tuple[int, ...]:
        """Lhs dims summed away: the contracted labels."""
        return tuple(self.lhs.index(label) for label in self.contracted())

    def rhs_sum_dims(self) -> Tuple[int, ...]:
        """Rhs dims summed away: the contracted labels, then the window taps."""
        dims = [self.rhs.index(label) for label in self.contracted()]
        dims += [self.rhs.index(w.tap) for w in self.windows]
        return tuple(dims)

    def window_of(self, label: str) -> Optional[Window]:
        """The window that `label` takes part in, as output, input or tap."""
        for w in self.windows:
            if label in (w.out, w.inp, w.tap):
                return w
        return None

    def out_dim(self, label: str) -> Optional[int]:
        """The output dim a lhs or rhs label becomes, or None if it is summed away.

        A window's input label becomes the window's output dim.
        """
        if label in self.out:
            return self.out.index(label)
        for w in self.windows:
            if label == w.inp:
                return self.out.index(w.out)
        return None

    def lhs_out_dims(self) -> Tuple[Optional[int], ...]:
        return tuple(self.out_dim(label) for label in self.lhs)

    def rhs_out_dims(self) -> Tuple[Optional[int], ...]:
        return tuple(self.out_dim(label) for label in self.rhs)


def elementwise_spec(lhs_rank: int, rhs_rank: int) -> OperatorSpec:
    """Elementwise op with rank broadcasting: dims align from the right."""
    rank = max(lhs_rank, rhs_rank)
    out = tuple(f"d{i}" for i in range(rank))
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


def block_matmul_spec() -> OperatorSpec:
    """Block matmul: a shared leading block dim, then a matmul per block."""
    return OperatorSpec.parse("bik,bkj->bij")


def conv_spec(strides: Sequence[int], depthwise: bool = False) -> OperatorSpec:
    """Convolution over ``len(strides)`` spatial axes, one stride per axis.

    Input is ``[C_in, *spatial]``, filter is ``[C_out, C_in, *taps]`` and
    output is ``[C_out, *spatial_out]``. The input channel is contracted. For
    depthwise, the filter is ``[C, *taps]`` and the channel is carried.

    conv2d is ``c_in h w, c_out c_in kh kw -> c_out h_out w_out`` with windows
    ``h_out = s*h + kh`` and ``w_out = s*w + kw``.
    """
    if not 1 <= len(strides) <= 3:
        raise NotImplementedError(f"conv over {len(strides)} spatial axes")
    spatial = ("d", "h", "w")[-len(strides) :]
    taps = tuple(f"k{axis}" for axis in spatial)
    outs = tuple(f"{axis}_out" for axis in spatial)
    windows = tuple(
        Window(out=o, inp=i, tap=t, stride=s)
        for o, i, t, s in zip(outs, spatial, taps, strides)
    )
    if depthwise:
        return OperatorSpec(("c",) + spatial, ("c",) + taps, ("c",) + outs, windows)
    return OperatorSpec(
        ("c_in",) + spatial, ("c_out", "c_in") + taps, ("c_out",) + outs, windows
    )


def spec_for_term(term, shapes) -> OperatorSpec:
    """The operator spec of a binary TensorTerm.

    Raises:
        NotImplementedError: If the operation has no spec
    """
    match term.op:
        case TensorOp.ADD | TensorOp.SUB | TensorOp.MUL:
            return elementwise_spec(len(shapes[0]), len(shapes[1]))
        case TensorOp.MATMUL:
            return matmul_spec(len(shapes[0]), len(shapes[1]))
        case TensorOp.BLOCK_MATMUL:
            return block_matmul_spec()
        case TensorOp.CONV2D:
            args = Conv2dArgs.from_term(term)
            # Depthwise terms carry a TFLite-style [1, Kh, Kw, C] filter, which
            # is not the [C, Kh, Kw] filter of conv_spec(depthwise=True).
            if args.groups != 1:
                raise NotImplementedError(f"conv2d groups: {args.groups}")
            return conv_spec([args.stride] * 2)
        case TensorOp.CONV3D:
            return conv_spec([Conv3dArgs.from_term(term).stride] * 3)
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
