"""Tests for optional layout simplicity bias (see ``util.layout_simplicity``)."""

from ir.dim import Dim, DimType
from ir.layout import Layout
from util.layout_simplicity import (
    conv2d_input_channel_adjacent_gap_penalty,
    layout_simplicity_penalty,
)


def test_layout_simplicity_penalty_matches_definition() -> None:
    n = 16
    layout = Layout(
        None, [], [Dim.parse("[0:4:1]"), Dim.parse("[4]"), Dim.parse("[1:4:1]")], n
    )
    g = sum(1 for d in layout.slot_dims if d.dim_type == DimType.EMPTY)
    extra_ct = max(0, len(layout.ct_dims) - 1)
    assert layout_simplicity_penalty(layout) == float(g) + 0.25 * float(extra_ct)


def test_layout_simplicity_full_slot_row_has_zero_penalty() -> None:
    n = 16
    layout = Layout(None, [], [Dim.parse("[0:16:1]")], n)
    assert layout_simplicity_penalty(layout) == 0.0


def test_conv2d_input_channel_adjacent_gap_penalty_accepts_gap_adjacent_orders() -> (
    None
):
    n = 32768
    preferred_a = Layout(
        None,
        [],
        [
            Dim.parse("[G:2]"),
            Dim.parse("[0:16:1]"),
            Dim.parse("[2:32:1]"),
            Dim.parse("[1:32:1]"),
        ],
        n,
    )
    preferred_b = Layout(
        None,
        [],
        [
            Dim.parse("[G:2]"),
            Dim.parse("[0:16:1]"),
            Dim.parse("[1:32:1]"),
            Dim.parse("[2:32:1]"),
        ],
        n,
    )
    assert conv2d_input_channel_adjacent_gap_penalty(preferred_a) == 0.0
    assert conv2d_input_channel_adjacent_gap_penalty(preferred_b) == 0.0


def test_conv2d_input_channel_adjacent_gap_penalty_rejects_nonadjacent_layout() -> None:
    n = 32768
    nonpreferred = Layout(
        None,
        [],
        [
            Dim.parse("[G:8]"),
            Dim.parse("[1:32:1]"),
            Dim.parse("[2:32:1]"),
            Dim.parse("[0:16:1]"),
        ],
        n,
    )
    assert conv2d_input_channel_adjacent_gap_penalty(nonpreferred) == 1.0


def test_conv2d_input_channel_adjacent_gap_penalty_is_neutral_without_slot_gaps() -> (
    None
):
    n = 32768
    no_gaps = Layout(
        None,
        [],
        [
            Dim.parse("[0:16:1]"),
            Dim.parse("[1:32:1]"),
            Dim.parse("[2:32:1]"),
        ],
        n,
    )
    assert conv2d_input_channel_adjacent_gap_penalty(no_gaps) == 0.0
