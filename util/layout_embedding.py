"""Expand Rotom layouts into physical ciphertext lanes.

This module is value-free: it mirrors ``apply_layout`` but records only where
each tensor index lives.  Both diagonal scoring and diagonal lowering use this
same map, so the cost model and emitted circuit agree on the physical meaning
of a layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from ir.dim import DimType
from ir.layout import Layout
from util.layout_util import add_vec, get_dim_indices, mul


Index = Tuple[int, ...]
MaybeIndex = Tuple[Optional[int], ...]


@dataclass(frozen=True)
class PhysicalLane:
    """One ciphertext slot and the tensor index stored there, if any."""

    ct: int
    slot: int
    index: Optional[Index]


@dataclass(frozen=True)
class LayoutEmbedding:
    """A full physical expansion of a layout."""

    layout: Layout
    tensor_shape: Index
    lanes: Tuple[PhysicalLane, ...]
    by_index: Dict[Index, Tuple[PhysicalLane, ...]]


def layout_embedding(layout: Layout, tensor_shape: Iterable[int]) -> LayoutEmbedding:
    """Map each physical lane to its logical tensor index.

    The arithmetic is the same as ``apply_layout``:
    - dimensions create local coordinate vectors;
    - rolls add one coordinate into another modulo that dimension extent;
    - strides turn split dimensions into mixed-radix tensor coordinates;
    - nonzero EMPTY coordinates invalidate padded/gap lanes.
    """

    tensor_shape = tuple(int(dim) for dim in tensor_shape)
    rank = len(tensor_shape)
    layout_len = max(len(layout), layout.n)
    dims = layout.get_dims()
    dim_indices = get_dim_indices(dims)

    for roll in layout.rolls:
        roll_index = roll.roll_index(dims)
        dim_indices[roll_index[0]] = [
            (dim_indices[roll_index[0]][i] + dim_indices[roll_index[1]][i])
            % roll.dim_to_roll.extent
            for i in range(layout_len)
        ]

    for i, dim in enumerate(dims):
        dim_indices[i] = mul(dim_indices[i], dim.stride)

    for i, dim in enumerate(dims):
        if dim.dim_type == DimType.EMPTY:
            dim_indices[i] = [value if not value else None for value in dim_indices[i]]

    indices_map: Dict[int, List[Optional[int]]] = {}
    for i, dim in enumerate(dims):
        if dim.dim is None:
            continue
        if dim.dim in indices_map:
            indices_map[dim.dim] = add_vec(indices_map[dim.dim], dim_indices[i])
        else:
            indices_map[dim.dim] = dim_indices[i]

    for i, dim in enumerate(dims):
        if dim.dim is None and dim.dim_type == DimType.EMPTY:
            for pertinent_dim in indices_map:
                indices_map[pertinent_dim] = add_vec(
                    indices_map[pertinent_dim], dim_indices[i]
                )

    lanes = []
    by_index: Dict[Index, List[PhysicalLane]] = {}
    zero_indices = [0] * layout_len
    for flat_idx in range(layout_len):
        maybe_index: MaybeIndex = tuple(
            indices_map.get(dim, zero_indices)[flat_idx] for dim in range(rank)
        )
        index = valid_index(maybe_index, tensor_shape)
        lane = PhysicalLane(
            ct=flat_idx // layout.n,
            slot=flat_idx % layout.n,
            index=index,
        )
        lanes.append(lane)
        if index is not None:
            by_index.setdefault(index, []).append(lane)

    return LayoutEmbedding(
        layout=layout,
        tensor_shape=tensor_shape,
        lanes=tuple(lanes),
        by_index={index: tuple(index_lanes) for index, index_lanes in by_index.items()},
    )


def valid_index(index: MaybeIndex, tensor_shape: Index) -> Optional[Index]:
    if any(dim is None for dim in index):
        return None
    concrete_index = tuple(int(dim) for dim in index if dim is not None)
    if not within_shape(concrete_index, tensor_shape):
        return None
    return concrete_index


def within_shape(index: Index, shape: Index) -> bool:
    return len(index) == len(shape) and all(
        0 <= value < shape[axis] for axis, value in enumerate(index)
    )
