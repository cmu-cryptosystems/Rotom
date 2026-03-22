"""Segment / dim index helpers when layouts contain duplicate gap dimensions.

``Dim.__eq__`` hashes on ``str(dim)``, so two distinct ``[G:n]`` slots compare equal.
``get_dim_map`` used to collapse them to one index, breaking :func:`get_segment`
and conv2d channel reduction when ``create_layout_without_dims`` leaves two gap
dims with the same repr (e.g. after summing one channel split).
"""

import pytest

from ir.dim import Dim, DimType
from ir.layout import Layout
from lower.layout_cts import create_layout_without_dims
from lower.lower_util import find_sum_dim
from util.layout_util import dim_list_index, get_segment


def test_dim_list_index_uses_identity_for_duplicate_gap_dims():
    a = Dim(None, 2, 1, DimType.EMPTY)
    b = Dim(None, 2, 1, DimType.EMPTY)
    assert a == b
    dims = [a, b]
    assert dim_list_index(a, dims) == 0
    assert dim_list_index(b, dims) == 1


def test_dim_list_index_raises_when_equality_is_ambiguous():
    a = Dim(None, 2, 1, DimType.EMPTY)
    b = Dim(None, 2, 1, DimType.EMPTY)
    c = Dim(None, 2, 1, DimType.EMPTY)
    dims = [a, b, c]
    q = Dim(None, 2, 1, DimType.EMPTY)
    assert all(q is not d for d in dims)
    assert q == a
    with pytest.raises(ValueError, match="ambiguous"):
        dim_list_index(q, dims)


def test_get_segment_split_channel_dims_resnet_style_layout():
    """Layout matching l1_0 conv1 secret input: outer/inner channel in slot_dims."""
    lay = Layout.from_string(
        "[R:16:16384][R:3:32][R:3:1];[0:8:2][1:32:1][2:32:1][G:2][0:2:1]",
        32768,
        secret=True,
    )
    _, slot_sum_dims = find_sum_dim(lay, 0)
    assert len(slot_sum_dims) == 2
    d_outer, d_inner = slot_sum_dims[0], slot_sum_dims[1]
    s_outer = get_segment(d_outer, lay.slot_dims)
    s_inner = get_segment(d_inner, lay.slot_dims)
    assert s_outer[1] == 8 and s_outer[2] == 4096
    assert s_inner[1] == 2 and s_inner[2] == 1

    new_lay = create_layout_without_dims(lay, [d_inner])
    s_outer_after = get_segment(d_outer, new_lay.slot_dims)
    assert s_outer_after[2] == 4096
