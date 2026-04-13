from copy import copy as copy
import hashlib
import os
import pickle
import uuid
from pathlib import Path

import numpy as np

from ir.dim import Dim, DimType
from ir.layout import Layout
from ir.roll import Roll
from util.util import prod, split_dim


def transpose(rows):
    return list(map(list, zip(*rows)))


def swap_rolls(layout, roll):
    # swap rolls
    new_rolls = []
    for base_roll in layout.rolls:
        if base_roll == roll:
            new_rolls.append(Roll(roll.dim_to_roll_by, roll.dim_to_roll))
        elif base_roll.dim_to_roll == roll.dim_to_roll:
            new_rolls.append(Roll(roll.dim_to_roll_by, base_roll.dim_to_roll_by))
        else:
            new_rolls.append(base_roll.copy())

    # swap dimensions
    dims = layout.dims
    new_dims = copy(dims)
    new_dims[roll.dim_to_roll] = dims[roll.dim_to_roll_by].copy()
    new_dims[roll.dim_to_roll_by] = dims[roll.dim_to_roll].copy()

    # create new layout
    eq_layout = Layout(layout.term, new_rolls, new_dims, layout.n, layout.secret)
    return eq_layout


def merge_dims(dims, in_roll):
    """Given a list of dimensions, try and merge the dimensions"""
    merged_dims = []
    if not dims:
        return merged_dims

    base_dim = copy(dims[0])
    for next_dim in dims[1:]:
        # don't merge dimensions that are used in rolls
        if base_dim in in_roll or next_dim in in_roll:
            merged_dims.append(base_dim)
            base_dim = next_dim
            continue
        if (
            base_dim.dim == next_dim.dim
            and base_dim.dim_type == next_dim.dim_type
            and base_dim.stride == next_dim.extent * next_dim.stride
        ):
            base_dim.extent *= next_dim.extent
            base_dim.stride = next_dim.stride
        elif (
            base_dim.dim is None
            and next_dim.dim is None
            and base_dim.dim_type == next_dim.dim_type
        ):
            base_dim.extent *= next_dim.extent
            base_dim.stride = 1
        else:
            merged_dims.append(base_dim)
            base_dim = next_dim
    merged_dims.append(base_dim)
    return merged_dims


def dimension_merging(layout):
    """Merge similar dimensions, filled or empty, together
    HACK: does not adjust rollutations on the layout
    QUESTION: can rolls cross the slot, ct dimensions?
    """
    merged_dims = []

    # find original rollutation indices
    roll_map = {}
    in_roll = set()
    for roll in layout.rolls:
        if roll.dim_to_roll not in roll_map:
            roll_map[roll.dim_to_roll] = []
        roll_map[roll.dim_to_roll].append(roll.dim_to_roll_by)
        in_roll.add(roll.dim_to_roll)
        in_roll.add(roll.dim_to_roll_by)

    # filter for only filled ct_dims
    # or if the dimension is in a roll
    layout_ct_dims = [
        copy(dim)
        for dim in layout.ct_dims
        if dim.dim_type == DimType.FILL or dim in in_roll
    ]

    merged_ct_dims = merge_dims(copy(layout_ct_dims), in_roll)
    merged_dims += merged_ct_dims

    # merge slot dims
    merged_slot_dims = merge_dims(
        [copy(slot_dim) for slot_dim in layout.slot_dims], in_roll
    )
    merged_dims += merged_slot_dims
    new_rolls = []
    for k, v in roll_map.items():
        for r in v:
            assert k in merged_dims
            assert r in merged_dims
            new_rolls.append(Roll(k, r))

    return Layout(layout.term, new_rolls, merged_dims, layout.n, layout.secret)


def align_dimension_extents(a_dims, b_dims):
    a_dim_len = prod([a_dim.extent for a_dim in a_dims])
    b_dim_len = prod([b_dim.extent for b_dim in b_dims])
    if a_dim_len > b_dim_len:
        b_dims = [Dim(None, a_dim_len // b_dim_len, 1, DimType.EMPTY)] + b_dims
    elif a_dim_len < b_dim_len:
        a_dims = [Dim(None, b_dim_len // a_dim_len, 1, DimType.EMPTY)] + a_dims

    a_dim_len = prod([a_dim.extent for a_dim in a_dims])
    b_dim_len = prod([b_dim.extent for b_dim in b_dims])
    assert a_dim_len == b_dim_len

    # find smallest dimension extent
    # min extent is set to 2
    min_extent = 2

    split_a = []
    for a_dim in a_dims:
        while a_dim.extent > min_extent:
            split_1, split_2 = split_dim(a_dim, min_extent)
            split_a.append(split_1)
            a_dim = split_2
        split_a.append(a_dim)

    split_b = []
    for b_dim in b_dims:
        while b_dim.extent > min_extent:
            split_1, split_2 = split_dim(b_dim, min_extent)
            split_b.append(split_1)
            b_dim = split_2
        split_b.append(b_dim)

    return split_a, split_b


def get_extent_dims(dims):
    # find smallest dimension extent
    # min extent is set to 2
    min_extent = 2

    split = []
    for dim in dims:
        while dim.extent > min_extent:
            split_1, split_2 = split_dim(dim, min_extent)
            split.append(split_1)
            dim = split_2
        split.append(dim)
    return split


def align_dimension_extents_compact(a_dims, b_dims):
    a_dim_len = prod(
        [a_dim.extent for a_dim in a_dims if a_dim.dim_type == DimType.FILL]
    )
    b_dim_len = prod(
        [b_dim.extent for b_dim in b_dims if b_dim.dim_type == DimType.FILL]
    )
    assert a_dim_len == b_dim_len

    # find smallest dimension extent
    # min extent is set to 2
    min_extent = 2

    split_a = []
    for a_dim in a_dims:
        while a_dim.extent > min_extent:
            split_1, split_2 = split_dim(a_dim, min_extent)
            split_a.append(split_1)
            a_dim = split_2
        split_a.append(a_dim)

    split_b = []
    for b_dim in b_dims:
        while b_dim.extent > min_extent:
            split_1, split_2 = split_dim(b_dim, min_extent)
            split_b.append(split_1)
            b_dim = split_2
        split_b.append(b_dim)
    return split_a, split_b


def align_dimension_extents_compact_skip_empty_gaps(a_dims, b_dims):
    """Like :func:`align_dimension_extents_compact` but do not split EMPTY (gap) dims.

    Used when ``a_dims`` / ``b_dims`` come from :meth:`Layout.get_dims` so logical
    gaps are already coalesced; splitting ``[G:32]`` into many ``[G:2]`` would
    desynchronize paired lists versus a target that keeps a single gap extent.
    """
    a_dim_len = prod(
        [a_dim.extent for a_dim in a_dims if a_dim.dim_type == DimType.FILL]
    )
    b_dim_len = prod(
        [b_dim.extent for b_dim in b_dims if b_dim.dim_type == DimType.FILL]
    )
    assert a_dim_len == b_dim_len

    min_extent = 2

    split_a = []
    for a_dim in a_dims:
        if a_dim.dim_type == DimType.EMPTY:
            split_a.append(a_dim)
            continue
        while a_dim.extent > min_extent:
            split_1, split_2 = split_dim(a_dim, min_extent)
            split_a.append(split_1)
            a_dim = split_2
        split_a.append(a_dim)

    split_b = []
    for b_dim in b_dims:
        if b_dim.dim_type == DimType.EMPTY:
            split_b.append(b_dim)
            continue
        while b_dim.extent > min_extent:
            split_1, split_2 = split_dim(b_dim, min_extent)
            split_b.append(split_1)
            b_dim = split_2
        split_b.append(b_dim)
    return split_a, split_b


def match_dims(dims, to_match):
    a_start = 0
    b_start = 0
    a_next = 0
    b_next = 0

    dims = dims[::-1]
    to_match = to_match[::-1]

    matched_dims = []
    replicated_stride = 1
    while a_next < len(dims) and b_next < len(to_match):
        a_dim = dims[a_next]
        b_dim = to_match[b_next]
        # both emtpy dimensions, skip
        if (
            a_dim.dim_type == DimType.EMPTY
            and b_dim.dim_type == DimType.EMPTY
            and a_dim.extent == b_dim.extent
        ):
            a_next += 1
            b_next += 1
            continue

        a_extent = prod([dim.extent for dim in dims[a_start : a_next + 1]])
        b_extent = prod([dim.extent for dim in to_match[b_start : b_next + 1]])
        if a_extent == b_extent:
            a_next += 1
            b_next += 1
            a_start = a_next
            b_start = b_next
            dim = copy(b_dim)
            dim.dim = a_dim.dim
            if a_dim.dim is None:
                dim.stride = replicated_stride
                replicated_stride *= dim.extent
            else:
                dim.stride = a_dim.extent // b_dim.extent * a_dim.stride
            dim.dim_type = a_dim.dim_type
            matched_dims.insert(0, dim)
        elif a_extent > b_extent:
            dim = copy(b_dim)
            dim.dim = a_dim.dim
            dim.stride = a_dim.stride
            dim.dim_type = a_dim.dim_type
            matched_dims.insert(0, dim)
            b_next += 1
        else:
            a_next += 1
    return matched_dims


def index_layout(layout, indexing_map, cts):
    dim_indices = get_dim_indices(layout.ct_dims)

    # group indices together
    grouped_dim_indices = {}
    for i in range(len(cts)):
        grouped_dim_indices[i] = []
    for indices in dim_indices:
        for i, index in enumerate(indices):
            grouped_dim_indices[i].append(index)

    # map values to dim_indices:
    for i, (dim, index) in enumerate(zip(layout.ct_dims, dim_indices)):
        keys_to_remove = []
        if dim in indexing_map:
            for k, v in grouped_dim_indices.items():
                if v[i] != indexing_map[dim]:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                del grouped_dim_indices[k]

    # get corresponding ct
    ct_indices = sorted(list(grouped_dim_indices.keys()))
    indexed_cts = [cts[i] for i in ct_indices]

    indexed_cts_dims = copy(layout.ct_dims)
    for dim in indexing_map:
        indexed_cts_dims.remove(dim)
    return indexed_cts, indexed_cts_dims


def convert_layout_to_mask(layout):
    slot_dims = layout.slot_dims
    slot_dim_indices = get_dim_indices(slot_dims)
    mask = [1] * layout.n
    for dim, indices in zip(slot_dims, slot_dim_indices):
        if dim.dim_type != DimType.EMPTY:
            continue
        for i, index in enumerate(indices):
            if index != 0:
                mask[i] = 0
    return mask


def layout_to_ct_indices(layout):
    dims = layout.ct_dims
    dim_indices = get_dim_indices(dims)

    # keep only the pertinent dims
    indices_map = {}
    for i, dim in enumerate(dims):
        if dim.dim is not None:
            dim_indices[i] = mul(dim_indices[i], dim.stride)

            if dim.dim not in indices_map:
                indices_map[dim.dim] = dim_indices[i]
            else:
                indices_map[dim.dim] = add_vec(indices_map[dim.dim], dim_indices[i])

    if len(indices_map) == 1:
        dim = [dim.dim for dim in dims if dim.dim is not None][0]
        return indices_map[dim]
    elif len(indices_map) == 2:
        keys = sorted(indices_map.keys())
        return [(a, b) for a, b in zip(indices_map[keys[0]], indices_map[keys[1]])]
    else:
        raise NotImplementedError("more than 2 dimensions")


def get_dim_map(dims):
    dim_map = {}
    for i, dim in enumerate(dims):
        dim_map[dim] = i
    return dim_map


def dim_list_index(dim, dims):
    """Index of ``dim`` in ``dims`` for segment / layout math.

    ``Dim.__eq__`` is hash-based on ``str(dim)``, so distinct gap dimensions like
    two ``[G:2]`` slots compare equal and a plain ``dict`` from :func:`get_dim_map`
    collapses them to one index. That breaks :func:`get_segment` and any pass that
    needs the correct physical position in ``dims``. Prefer object identity, then
    fall back to equality only when it is unambiguous.
    """
    for i, d in enumerate(dims):
        if d is dim:
            return i
    matches = [i for i, d in enumerate(dims) if d == dim]
    if len(matches) == 1:
        return matches[0]
    raise ValueError(
        f"dim {dim!r} is missing or ambiguous in dims (eq matches={matches}); "
        "pass the exact Dim instance from this layout's ct_dims/slot_dims list."
    )


def get_segments(dims):
    n = 1
    for dim in dims:
        n *= dim.extent

    segments = {}
    for i in range(len(dims)):
        segment_len = int(prod([dim.extent for dim in dims[i + 1 :]]))
        extent = dims[i].extent
        count = n // segment_len // extent
        segments[i] = [count, extent, segment_len]
    return segments


def get_segment(dim, dims):
    segments = get_segments(dims)
    idx = dim_list_index(dim, dims)
    return segments[idx]


def get_cts_by_dim(layout_cts, dim):
    """Get ciphertext groups by dimension from LayoutCiphertexts.

    Args:
        layout_cts: LayoutCiphertexts object containing layout and ciphertexts
        dim: Dim object to group by

    Returns:
        List of lists of HETerm objects, grouped by the dimension
    """
    cts = layout_cts.cts
    ct_dims = layout_cts.layout.ct_dims
    assert dim in ct_dims
    ct_dim_index = dim_list_index(dim, ct_dims)

    groups = []
    dim_indices = get_dim_indices(ct_dims)
    indices = dim_indices[ct_dim_index]
    while any(i is not None for i in indices):
        group = []
        for i in range(dim.extent):
            for j in range(len(indices)):
                if indices[j] == i:
                    group.append(j)
                    indices[j] = None
                    break
        groups.append(group)

    ct_groups = []
    for group in groups:
        ct_group = []
        for g in group:
            ct_group.append(cts[g])
        ct_groups.append(ct_group)
    return ct_groups


def get_ct_idxs_by_dim(ct_dims, dim):
    assert dim in ct_dims
    ct_dim_index = dim_list_index(dim, ct_dims)

    dim_indices = get_dim_indices(ct_dims)
    indices = list(dim_indices[ct_dim_index])

    num_ct = len(indices)
    if num_ct == dim.extent:
        groups = [[j for j in range(num_ct)]]
    else:
        groups = []
        for i in range(dim.extent):
            group = [j for j in range(len(indices)) if indices[j] == i]
            groups.append(group)

    return [[g for g in group] for group in groups]


def get_dim_indices(dims):
    segments = get_segments(dims)
    all_indices = []
    for i in range(len(dims)):
        segment = segments[i]
        i_len = segment[0]
        j_len = segment[1]
        k_len = segment[2]
        indices = []
        for _ in range(i_len):
            for j in range(j_len):
                for _ in range(k_len):
                    indices.append(j)
        all_indices.append(indices)
    return all_indices


def get_dim_indices_by_dim(dims):
    dim_indices = get_dim_indices(dims)
    dim_map = {}
    for dim, dim_index in zip(dims, dim_indices):
        dim_index = mul(dim_index, dim.stride)
        if dim.dim not in dim_map:
            dim_map[dim.dim] = dim_index
        else:
            dim_map[dim.dim] = add_vec(dim_map[dim.dim], dim_index)
    return dim_map


def add_vec(a, b):
    return [x + y if x is not None and y is not None else None for x, y in zip(a, b)]


def add_vecs(a, b):
    result = []
    for x, y in zip(a, b):
        if x is not None and y is not None:
            result.append(x + y)
        elif x is not None:
            result.append(x)
        elif y is not None:
            result.append(y)
        else:
            result.append(None)
    return result


def add_vecs_of_vecs(a, b):
    return [add_vecs(x, y) for x, y in zip(a, b)]


def mul_vec(a, b):
    return [x * y if x is not None and y is not None else None for x, y in zip(a, b)]


def add(vec, n):
    return [v + n if v is not None else None for v in vec]


def mul(vec, n):
    return [v * n if v is not None else None for v in vec]


def _normalize_scalar_from_tensor(value):
    """Match legacy ``apply_layout`` scalar extraction from indexed tensor values."""
    if isinstance(value, np.ndarray) and value.ndim > 0 and value.size == 1:
        return value.item()
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return value.item()
    return value


def _ct_index_rows_to_values(pt_tensor, ct_indices):
    """Map layout index rows to plaintext tensor values (semantics match legacy loop).

    Hot path uses vectorized advanced indexing (same as ``pt_tensor[tuple(idx)]`` per
    row), which is correct for C-, F-, and non-contiguous arrays. A stride + ``ravel``
    shortcut was incorrect after ops like ``PERMUTE`` that yield Fortran-ordered views.
    """
    pt_tensor_ndim = int(np.ndim(pt_tensor))
    n = len(ct_indices)
    if n == 0:
        return []

    if pt_tensor_ndim == 0:
        out = []
        base = pt_tensor.item()
        for index in ct_indices:
            effective_index = list(index)
            while len(effective_index) < pt_tensor_ndim:
                effective_index.append(0)
            if any(effective_index[i] is None for i in range(pt_tensor_ndim)):
                out.append(0)
                continue
            if any(
                effective_index[i] >= pt_tensor.shape[i] for i in range(pt_tensor_ndim)
            ):
                out.append(0)
                continue
            out.append(_normalize_scalar_from_tensor(base))
        return out

    # Build (n, pt_tensor_ndim) index matrix.
    # Fastest case: plan provides a compact int32 array with -1 sentinel for None.
    if (
        isinstance(ct_indices, np.ndarray)
        and ct_indices.ndim == 2
        and ct_indices.dtype == np.int32
    ):
        # ct_indices is shaped (n, ndims_in_plan). When the plan was built for fewer
        # dims than the runtime tensor has (e.g. vector plan applied to (N, 1)),
        # pad missing dims with zeros to match legacy behavior.
        plan_ndim = int(ct_indices.shape[1])
        use = min(plan_ndim, pt_tensor_ndim)
        if use == pt_tensor_ndim:
            idx = ct_indices[:, :pt_tensor_ndim].astype(np.int64, copy=False)
        else:
            idx = np.zeros((n, pt_tensor_ndim), dtype=np.int64)
            idx[:, :use] = ct_indices[:, :use].astype(np.int64, copy=False)
        bad = (idx[:, :use] == -1).any(axis=1) if use else np.zeros(n, dtype=bool)
    else:
        idx = np.zeros((n, pt_tensor_ndim), dtype=np.int64)
        bad = np.zeros(n, dtype=bool)
        for i, index in enumerate(ct_indices):
            row = list(index)
            if len(row) < pt_tensor_ndim:
                row = row + [0] * (pt_tensor_ndim - len(row))
            elif len(row) > pt_tensor_ndim:
                row = row[:pt_tensor_ndim]
            for j in range(pt_tensor_ndim):
                v = row[j]
                if v is None:
                    bad[i] = True
                else:
                    idx[i, j] = int(v)

    shp = np.asarray(pt_tensor.shape, dtype=np.int64)
    oob = (idx < 0) | (idx >= shp)
    bad |= oob.any(axis=1)

    # Fast path: numeric ndarray — advanced indexing (memory-order agnostic).
    if (
        isinstance(pt_tensor, np.ndarray)
        and pt_tensor.size > 0
        and np.issubdtype(pt_tensor.dtype, np.number)
    ):
        lim = np.maximum(shp - 1, 0)
        safe = idx.astype(np.int64, copy=True)
        safe[bad] = 0
        for j in range(pt_tensor_ndim):
            safe[:, j] = np.clip(safe[:, j], 0, lim[j])
        coords = tuple(safe[:, j] for j in range(pt_tensor_ndim))
        vals = np.asarray(pt_tensor[coords], dtype=np.float64)
        vals[bad] = 0.0
        # One vectorized copy to Python floats (faster than per-element normalize).
        return vals.tolist()

    # Non-ndarray or empty: preserve exact legacy behavior.
    out = []
    for index in ct_indices:
        effective_index = list(index)
        while len(effective_index) < pt_tensor_ndim:
            effective_index.append(0)
        if any(effective_index[i] is None for i in range(pt_tensor_ndim)):
            out.append(0)
            continue
        if any(effective_index[i] >= pt_tensor.shape[i] for i in range(pt_tensor_ndim)):
            out.append(0)
            continue
        value = pt_tensor[tuple(effective_index[i] for i in range(pt_tensor_ndim))]
        out.append(_normalize_scalar_from_tensor(value))
    return out


def apply_layout(pt_tensor, layout):
    """apply a layout to a pt tensor"""
    layout_len = max(len(layout), layout.n)
    # get base_term indices
    pt_tensor_ndim = np.ndim(pt_tensor)
    plan = _get_apply_layout_plan(layout, pt_tensor_ndim, layout_len=layout_len)
    # Plans may optionally deduplicate identical ciphertext index blocks.
    # - base_indices_by_cts: (num_ct, n, ndims) int32 with -1 sentinel
    # - unique_base_indices_by_cts: (num_unique_ct, n, ndims)
    # - ct_inverse: (num_ct,) mapping each ct -> unique index
    base_indices_by_cts = plan.get(
        "unique_base_indices_by_cts", plan["base_indices_by_cts"]
    )
    ct_inverse = plan.get("ct_inverse", None)

    cts = []
    # base_indices_by_cts may be either:
    # - list[list[list[int|None]]] (legacy)
    # - np.ndarray shaped (num_ct, n, ndims) with int32 and -1 sentinel (compact)
    if isinstance(base_indices_by_cts, np.ndarray):
        for i in range(base_indices_by_cts.shape[0]):
            cts.append(_ct_index_rows_to_values(pt_tensor, base_indices_by_cts[i]))
    else:
        for ct_indices in base_indices_by_cts:
            cts.append(_ct_index_rows_to_values(pt_tensor, ct_indices))

    if ct_inverse is None:
        return cts
    # Expand unique ciphertexts back to full list.
    return [cts[int(j)] for j in ct_inverse]


_APPLY_LAYOUT_PLAN_CACHE: dict[str, dict] = {}
_APPLY_LAYOUT_PLAN_CACHE_VERSION = "5"

# Optional directory override (set by ``activate_benchmark_layout_plan_cache``). Takes
# precedence over ``ROTOM_APPLY_LAYOUT_PLAN_CACHE_DIR`` so callers do not mutate os.environ.
_APPLY_LAYOUT_PLAN_DISK_DIR_OVERRIDE: str | None = None


def set_apply_layout_plan_disk_cache_dir(path: str | None) -> None:
    """Set disk cache directory for layout plans, or clear override when ``path`` is empty."""
    global _APPLY_LAYOUT_PLAN_DISK_DIR_OVERRIDE
    if not path or not str(path).strip():
        _APPLY_LAYOUT_PLAN_DISK_DIR_OVERRIDE = None
    else:
        _APPLY_LAYOUT_PLAN_DISK_DIR_OVERRIDE = str(path).strip()


def _resolve_apply_layout_plan_disk_dir() -> str:
    if _APPLY_LAYOUT_PLAN_DISK_DIR_OVERRIDE:
        return _APPLY_LAYOUT_PLAN_DISK_DIR_OVERRIDE
    return os.environ.get("ROTOM_APPLY_LAYOUT_PLAN_CACHE_DIR", "").strip()


def _apply_layout_plan_cache_key(layout, pt_tensor_ndim: int, layout_len: int) -> str:
    # The apply-layout plan depends on the *packing geometry* (dims/rolls/ct layout),
    # not on the specific tensor term attached to the layout. `str(layout)` includes
    # `layout.term` (see `Layout.__repr__`), which would cause cache misses for
    # identical geometry across different intermediate tensors.
    payload = (
        f"{_APPLY_LAYOUT_PLAN_CACHE_VERSION}|layout_geom={layout.layout_str()}|ndim={pt_tensor_ndim}|layout_len={layout_len}|n={layout.n}"
    ).encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()


def _to_i32_with_none_sentinel(xs: list, *, none_sentinel: int = -1) -> np.ndarray:
    """Convert a Python list possibly containing None to int32 with sentinel."""
    # Use object -> int loop; this runs once per plan build (not per apply_layout call).
    out = np.empty(len(xs), dtype=np.int32)
    for i, v in enumerate(xs):
        out[i] = none_sentinel if v is None else int(v)
    return out


def _combine_index_arrays(
    a: np.ndarray, b: np.ndarray, *, none_sentinel: int = -1
) -> np.ndarray:
    """Elementwise combine like add_vecs_of_vecs with None sentinel."""
    # Rules:
    # - if both valid: a+b
    # - if one valid: that one
    # - if both none: none
    av = a != none_sentinel
    bv = b != none_sentinel
    out = np.full_like(a, none_sentinel)
    both = av & bv
    out[both] = a[both] + b[both]
    only_a = av & ~bv
    out[only_a] = a[only_a]
    only_b = ~av & bv
    out[only_b] = b[only_b]
    return out


def _get_apply_layout_plan(layout, pt_tensor_ndim: int, *, layout_len: int) -> dict:
    """Precompute the expensive layout -> indices mapping.

    This avoids rebuilding the large `get_dim_indices` / `base_indices_by_cts`
    structures on every call. The hot-loop semantics are intentionally kept
    identical to the original `apply_layout` (including padding behavior).
    """
    key = _apply_layout_plan_cache_key(layout, pt_tensor_ndim, layout_len=layout_len)
    if key in _APPLY_LAYOUT_PLAN_CACHE:
        return _APPLY_LAYOUT_PLAN_CACHE[key]

    disk_dir = _resolve_apply_layout_plan_disk_dir()
    plan = None
    if disk_dir:
        out_dir = Path(disk_dir).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / f"{key}.pkl"
        if plan_path.exists():
            with open(plan_path, "rb") as f:
                plan = pickle.load(f)

    if plan is None:
        # get base_term indices
        dims = layout.get_dims()
        dim_indices = get_dim_indices(dims)

        # apply any base term rolls
        for roll in layout.rolls:
            roll_index = roll.roll_index(dims)
            dim_indices[roll_index[0]] = [
                (dim_indices[roll_index[0]][i] + dim_indices[roll_index[1]][i])
                % roll.dim_to_roll.extent
                for i in range(layout_len)
            ]

        # update dim with strides
        for i in range(len(dim_indices)):
            dim_indices[i] = mul(dim_indices[i], dims[i].stride)

        # update dims
        for i in range(len(dims)):
            if dims[i].dim_type == DimType.EMPTY:
                dim_indices[i] = [j if not j else None for j in dim_indices[i]]

        # split indices by dimensions:
        indices_map = {}
        for i, dim in enumerate(dims):
            if dim.dim is not None:
                if dim.dim in indices_map:
                    indices_map[dim.dim] = add_vec(indices_map[dim.dim], dim_indices[i])
                else:
                    indices_map[dim.dim] = dim_indices[i]

        for i, dim in enumerate(dims):
            # used for adding gap slots into layout
            if dim.dim is None and dim.dim_type == DimType.EMPTY:
                for pertinent_dim in indices_map:
                    indices_map[pertinent_dim] = add_vec(
                        indices_map[pertinent_dim], dim_indices[i]
                    )

        # Map to pertinent dimensions, compactly.
        ndims = max(indices_map) + 1
        base = np.zeros((layout_len, ndims), dtype=np.int32)
        # Populate columns; None becomes -1 sentinel.
        for dim, indices in indices_map.items():
            base[:, int(dim)] = _to_i32_with_none_sentinel(indices, none_sentinel=-1)

        # Split by ciphertexts: (num_ct, n, ndims)
        num_ct = layout_len // layout.n
        base_by_ct = base.reshape((num_ct, layout.n, ndims))

        # combine cts if ct_dim is a gap dimension
        combined_cts = base_by_ct
        ct_dims = copy(layout.ct_dims)
        for ct_dim in layout.ct_dims:
            if ct_dim.dim_type == DimType.EMPTY:
                new_combined_cts = []
                ct_indices = get_ct_idxs_by_dim(ct_dims, ct_dim)
                for indices in ct_indices:
                    cur = combined_cts[indices[0]]
                    for index in indices[1:]:
                        cur = _combine_index_arrays(
                            cur, combined_cts[index], none_sentinel=-1
                        )
                    new_combined_cts.append(cur)
                ct_dims.remove(ct_dim)
                combined_cts = np.stack(new_combined_cts, axis=0)
        base_by_ct = combined_cts

        plan = {"base_indices_by_cts": base_by_ct}

        # Optional dedup: if some ct dimensions do not contribute to plaintext indexing
        # (e.g. replication dims `[R:…]` in `layout.ct_dims`), many ciphertext blocks are
        # identical. Detect that by grouping ciphertexts by the indices of ct-dims that
        # correspond to real tensor dimensions.
        #
        # This is especially important for ResNet conv layouts where ct replication can
        # multiply num_ct (e.g. ×16×3×3) without changing any tensor coordinates.
        if isinstance(base_by_ct, np.ndarray) and len(ct_dims) > 0:
            keep_ct_dim_idxs = [
                i for i, d in enumerate(ct_dims) if d.dim is not None and d.extent > 1
            ]
            if keep_ct_dim_idxs:
                ct_dim_indices = get_dim_indices(
                    ct_dims
                )  # list[len(ct_dims)] of [num_ct]
                key_cols = [
                    np.asarray(ct_dim_indices[i], dtype=np.int32)
                    for i in keep_ct_dim_idxs
                ]
                keys = np.stack(key_cols, axis=1)  # (num_ct, nkey)
                # Group by unique keys; pick first representative for each group.
                _, first, inv = np.unique(
                    keys, axis=0, return_index=True, return_inverse=True
                )
                if first.size < num_ct:
                    # Preserve deterministic order by sorting representatives by first appearance.
                    order = np.argsort(first)
                    first = first[order]
                    # Remap inverse to the sorted representative indices.
                    remap = np.empty(order.size, dtype=np.int32)
                    remap[order] = np.arange(order.size, dtype=np.int32)
                    inv = remap[inv]
                    plan["unique_base_indices_by_cts"] = base_by_ct[first]
                    plan["ct_inverse"] = inv.astype(np.int32, copy=False)

        if disk_dir:
            plan_path = Path(disk_dir).expanduser() / f"{key}.pkl"
            # Unique temp name: Toy multiprocessing runs many workers; a shared
            # ``*.pkl.tmp`` caused one process to ``os.replace`` another's temp.
            tmp_path = plan_path.parent / (
                f"{plan_path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with open(tmp_path, "wb") as f:
                    pickle.dump(plan, f, protocol=pickle.HIGHEST_PROTOCOL)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, plan_path)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    _APPLY_LAYOUT_PLAN_CACHE[key] = plan
    return plan


def apply_punctured_layout(pt_tensor, layout):
    """apply a layout to a pt tensor"""
    cts = apply_layout(pt_tensor, layout)
    # make the punctured matrix holey!
    masks = layout.term.cs[4]
    assert len(cts) == len(masks)

    for i in range(len(cts)):
        c_arr = np.asarray(cts[i], dtype=np.float64)
        m_arr = np.asarray(masks[i], dtype=np.float64)
        cts[i] = (c_arr * m_arr).tolist()
    return cts


def infer_n_from_layout_str(layout_str: str) -> int | None:
    """
    Infer HE vector slot count `n` from a layout string.

    Rule:
      - If there is a `;`, compute the product of extents of bracket terms after `;`.
      - Otherwise, compute the product of extents of all bracket terms.

    Extent extraction supports the layout syntax used by `Dim.parse`, including:
      - `[i:n:s]` / `[R:n:s]` -> extent is the middle `n`
      - `[R:n]` / `[G:n]` / `[i:n]` -> extent is the second `n`
      - `[n]` -> extent is the only `n`
    """
    import re

    traversal = layout_str.split(";", 1)[1] if ";" in layout_str else layout_str

    extents: list[int] = []
    for token in re.findall(r"\[([^\]]+)\]", traversal):
        parts = [p.strip() for p in token.split(":")]
        if not parts:
            continue

        extent: int | None = None
        try:
            if len(parts) == 1:
                extent = int(parts[0])
            elif len(parts) == 2:
                extent = int(parts[1])
            elif len(parts) == 3:
                extent = int(parts[1])
        except ValueError:
            continue

        if extent is not None and extent >= 1:
            extents.append(int(extent))

    if not extents:
        return None

    out = 1
    for e in extents:
        out *= e
    return out


def parse_layout(layout_str: str, n: int | None = None, secret: bool = False):
    """
    Parse a layout string into a `Layout` object.

    If `n` is omitted, infer it from the layout string (falling back to 16).
    """
    if n is None:
        inferred = infer_n_from_layout_str(layout_str)
        n = inferred if inferred is not None else 16
    return Layout.from_string(layout_str, n, secret)
