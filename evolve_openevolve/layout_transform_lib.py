"""
Safe layout transformation primitives for OpenEvolve search.

Layouts use a traversal dimension list ``layout.dims`` (see ``ir.layout.Layout``).
Rolls reference ``Dim`` objects by identity; permutations reorder the same objects.

These helpers preserve object identity for dimensions so existing ``Roll`` entries
remain valid. New rolls must use dimensions with matching ``extent`` (``Roll`` invariant).
"""

from __future__ import annotations

from typing import Sequence

from ir.layout import Layout
from ir.roll import Roll


def apply_permute_traversal_dims(layout: Layout, perm: Sequence[int]) -> Layout:
    """
    Reorder ``layout.dims`` by permutation ``perm`` where ``new_dims[i] = old_dims[perm[i]]``.

    ``perm`` must be a permutation of ``range(len(layout.dims))`` (includes gap ``G:`` dims).
    Existing rolls keep referencing the same ``Dim`` instances.
    """
    old = list(layout.dims)
    n = len(old)
    if len(perm) != n or set(perm) != set(range(n)):
        raise ValueError(f"perm must be a permutation of range({n}), got {tuple(perm)!r}")
    new_dims = [old[perm[i]] for i in range(n)]
    new_rolls = [Roll(r.dim_to_roll, r.dim_to_roll_by) for r in layout.rolls]
    if len(new_rolls) != len(set(new_rolls)):
        raise ValueError("duplicate roll after permute")
    return Layout(layout.term, new_rolls, new_dims, layout.n, layout.secret)


def apply_append_roll(layout: Layout, dim_to_roll_idx: int, dim_to_roll_by_idx: int) -> Layout:
    """
    Append ``Roll(layout.dims[dim_to_roll_idx], layout.dims[dim_to_roll_by_idx])``.

    Requires equal extents on the two dimensions (``Roll`` constructor).
    """
    dims = list(layout.dims)
    if not (0 <= dim_to_roll_idx < len(dims) and 0 <= dim_to_roll_by_idx < len(dims)):
        raise IndexError("roll dimension index out of range")
    if dim_to_roll_idx == dim_to_roll_by_idx:
        raise ValueError("roll indices must differ")
    new_r = Roll(dims[dim_to_roll_idx], dims[dim_to_roll_by_idx])
    new_rolls = list(layout.rolls) + [new_r]
    if len(new_rolls) != len(set(new_rolls)):
        raise ValueError("duplicate roll")
    return Layout(layout.term, new_rolls, dims, layout.n, layout.secret)


def clone_layout_identity(layout: Layout) -> Layout:
    """Copy rolls with fresh ``Roll`` objects; same ``dims`` order."""
    new_rolls = [Roll(r.dim_to_roll, r.dim_to_roll_by) for r in layout.rolls]
    return Layout(layout.term, new_rolls, list(layout.dims), layout.n, layout.secret)
