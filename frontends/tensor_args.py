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
from typing import Any, Dict, List, Tuple


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


def axis_paddings(padding: Any, axes: int) -> Tuple[Any, ...]:
    """A conv padding as one entry per spatial axis.

    ``"valid"`` or ``"same"`` applies to every axis; a sequence gives each
    axis its own mode or its own ``(before, after)`` cells of zero padding.
    """
    if isinstance(padding, str):
        return (padding,) * axes
    pads = tuple(p if isinstance(p, str) else (int(p[0]), int(p[1])) for p in padding)
    if len(pads) != axes:
        raise ValueError(f"expected {axes} paddings, got {padding!r}")
    return pads


def window_axis(
    extent: int, kernel: int, stride: int, padding: Any
) -> Tuple[int, int, int]:
    """One spatial axis of a windowed operator (conv, pooling).

    Returns:
        (out_extent, pad_before, pad_after)

    ``"same"`` with stride 1 keeps the extent and splits ``kernel - 1`` of
    padding with the extra cell after. ``"same"`` with a larger stride pads
    ``kernel // 2`` on both sides, as ``torch.nn.Conv2d`` does. A pair is the
    padding itself: ``(kernel - 1, 0)`` is causal, and TensorFlow's ``SAME``
    is ``tf_same_padding``.
    """
    if not isinstance(padding, str):
        before, after = padding
        return (extent + before + after - kernel) // stride + 1, before, after
    if padding == "valid":
        return (extent - kernel) // stride + 1, 0, 0
    if padding != "same":
        raise NotImplementedError(f"unknown padding: {padding!r}")
    if stride == 1:
        total = max(0, kernel - 1)
        return extent, total // 2, total - total // 2
    pad = kernel // 2
    return (extent + 2 * pad - kernel) // stride + 1, pad, pad


def tf_same_padding(extent: int, kernel: int, stride: int) -> Tuple[int, int]:
    """TensorFlow's ``SAME`` padding of one axis: ``ceil(extent / stride)``
    outputs, with the extra cell after."""
    out = -(-extent // stride)
    total = max((out - 1) * stride + kernel - extent, 0)
    return total // 2, total - total // 2


class _ConvArgs:
    """What Conv2dArgs and Conv3dArgs share: a window over the axes after the
    channel."""

    #: number of spatial axes
    axes = 0

    @property
    def strides(self) -> Tuple[int, ...]:
        """The stride of each spatial axis."""
        return axis_strides(self.stride, self.axes)

    @property
    def depthwise(self) -> bool:
        """True when each channel is convolved on its own.

        The filter is then TFLite's ``[1, *taps, C * multiplier]``, not
        ``[C_out, C_in, *taps]``.
        """
        return self.groups == "depthwise"

    def taps(self, filter_shape) -> Tuple[int, ...]:
        """The kernel extent of each spatial axis, for a filter of this shape."""
        if self.depthwise:
            return tuple(int(k) for k in filter_shape[1 : 1 + self.axes])
        return tuple(int(k) for k in filter_shape[-self.axes :])

    def out_channels(self, filter_shape) -> int:
        """The number of output channels of a filter of the given shape."""
        return int(filter_shape[-1] if self.depthwise else filter_shape[0])

    def window(self, in_shape, filter_shape):
        """``(out_extent, pad_before, pad_after)`` of each spatial axis."""
        return [
            window_axis(int(extent), tap, stride, padding)
            for extent, tap, stride, padding in zip(
                in_shape[1:],
                self.taps(filter_shape),
                self.strides,
                axis_paddings(self.padding, self.axes),
            )
        ]


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
class Conv2dArgs(_ConvArgs):
    """Structured view of a CONV2D term (term.cs = [input, filter, stride, padding, groups])."""

    input: Any
    filter: Any  # noqa: A003
    stride: Any
    padding: Any
    groups: Any

    axes = 2

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


@dataclass(frozen=True)
class Conv3dArgs(_ConvArgs):
    """Structured view of a CONV3D term (term.cs = [input, filter, stride, padding, groups])."""

    input: Any
    filter: Any  # noqa: A003
    stride: Any
    padding: Any
    groups: Any = 1

    axes = 3

    @classmethod
    def from_term(cls, term: Any) -> "Conv3dArgs":
        return cls(
            input=term.cs[0],
            filter=term.cs[1],
            stride=term.cs[2],
            padding=term.cs[3],
            groups=term.cs[4] if len(term.cs) > 4 else 1,
        )


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
