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
from typing import Any, Dict, List, Optional


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
    stride: int
    padding: str
    groups: Any

    @classmethod
    def from_term(cls, term: Any) -> "Conv2dArgs":
        groups = 1
        if len(term.cs) > 4:
            groups = term.cs[4]
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
        if len(term.cs) > 4:
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
    stride: int
    padding: str

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
