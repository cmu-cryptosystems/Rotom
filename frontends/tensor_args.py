"""
Structured views of TensorTerm wire format (term.cs).

This module defines dataclasses and helpers to read tensor operation arguments
from terms without relying on positional indices. The wire format is still
defined where terms are built (e.g. frontends.tensor); this module only
interprets it.

Import from here when you need to read term.cs in a type-safe way without
pulling in the full tensor frontend (e.g. lower, assignment, ir/analysis).
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def axis_strides(stride: Any, axes: int) -> Tuple[int, ...]:
    """A conv stride as one int per spatial axis.

    An int applies to every axis; a sequence gives one stride per axis.
    """
    if isinstance(stride, int):
        strides = (stride,) * axes
    else:
        strides = tuple(int(s) for s in stride)
    if len(strides) != axes or any(s < 1 for s in strides):
        raise ValueError(f"expected {axes} positive strides, got {stride!r}")
    return strides


def window_axis(
    extent: int, kernel: int, stride: int, padding: str
) -> Tuple[int, int, int]:
    """One spatial axis of a windowed operator (conv, pooling).

    Returns:
        (out_extent, pad_before, pad_after)

    ``"same"`` with stride 1 keeps the extent and splits ``kernel - 1`` of
    padding with the extra cell after. ``"same"`` with a larger stride pads
    ``kernel // 2`` on both sides, as ``torch.nn.Conv2d`` does.
    """
    if padding == "valid":
        return (extent - kernel) // stride + 1, 0, 0
    if padding != "same":
        raise NotImplementedError(f"unknown padding: {padding!r}")
    if stride == 1:
        total = max(0, kernel - 1)
        return extent, total // 2, total - total // 2
    pad = kernel // 2
    return (extent + 2 * pad - kernel) // stride + 1, pad, pad


@dataclass(frozen=True)
class PolyCallArgs:
    """Structured view of a POLY_CALL term (term.cs = [input, name, lower_bound, upper_bound])."""

    name: str
    lower_bound: float
    upper_bound: float

    @classmethod
    def from_term(cls, term: Any) -> "PolyCallArgs":
        return cls(
            name=term.cs[1],
            lower_bound=float(term.cs[2]),
            upper_bound=float(term.cs[3]),
        )


@dataclass(frozen=True)
class Conv2dArgs:
    """Structured view of a CONV2D term (term.cs = [input, filter, stride, padding]).
    After assignment, term.cs may have computed padding at index 4; use get_computed_padding(term).
    """

    input: Any
    filter: Any  # noqa: A003
    stride: Any
    padding: str
    groups: Any

    @property
    def strides(self) -> Tuple[int, int]:
        """The stride of the H and W axes."""
        return axis_strides(self.stride, 2)

    @classmethod
    def from_term(cls, term: Any) -> "Conv2dArgs":
        groups = 1
        if len(term.cs) > 4:
            cs4 = term.cs[4]
            if isinstance(cs4, (int, str)):
                groups = cs4
        return cls(
            input=term.cs[0],
            filter=term.cs[1],
            stride=term.cs[2],
            padding=term.cs[3],
            groups=groups,
        )

    @staticmethod
    def get_computed_padding(term: Any) -> Optional[List[int]]:
        """Return [pad_top, pad_bottom, pad_left, pad_right] if set by assignment (term.cs[4])."""
        if len(term.cs) > 5 and isinstance(term.cs[5], list):
            return term.cs[5]
        if len(term.cs) > 4 and isinstance(term.cs[4], list):
            return term.cs[4]
        return None


@dataclass(frozen=True)
class Conv3dArgs:
    """Structured view of a CONV3D term (term.cs = [input, filter, stride, padding]).

    After assignment, term.cs may have computed padding at index 4; for Conv3D this is
    `[pad_front, pad_back, pad_top, pad_bottom, pad_left, pad_right]`.
    """

    input: Any
    filter: Any  # noqa: A003
    stride: Any
    padding: str

    @property
    def strides(self) -> Tuple[int, int, int]:
        """The stride of the D, H and W axes."""
        return axis_strides(self.stride, 3)

    @classmethod
    def from_term(cls, term: Any) -> "Conv3dArgs":
        return cls(
            input=term.cs[0],
            filter=term.cs[1],
            stride=term.cs[2],
            padding=term.cs[3],
        )

    @staticmethod
    def get_computed_padding(term: Any) -> Optional[List[int]]:
        """Return [pf, pb, pt, pbot, pl, pr] if set by assignment (term.cs[4])."""
        if len(term.cs) > 4:
            return term.cs[4]
        return None


@dataclass(frozen=True)
class ReshapeArgs:
    """Structured view of a RESHAPE term (term.cs = [input, dim, shape])."""

    input: Any
    dim: int
    shape: Dict[int, int]

    @classmethod
    def from_term(cls, term: Any) -> "ReshapeArgs":
        return cls(input=term.cs[0], dim=term.cs[1], shape=term.cs[2])


@dataclass(frozen=True)
class TensorPlaceholderArgs:
    """Structured view of a TENSOR placeholder term (term.cs = [name, shape, secret])."""

    name: str
    shape: List[int]
    secret: bool

    @classmethod
    def from_term(cls, term: Any) -> "TensorPlaceholderArgs":
        return cls(name=term.cs[0], shape=term.cs[1], secret=term.cs[2])
