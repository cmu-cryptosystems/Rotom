"""Tests for optional layout simplicity bias (see ``util.layout_simplicity``)."""

from ir.dim import Dim, DimType
from ir.layout import Layout
from util.layout_simplicity import layout_simplicity_penalty


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
