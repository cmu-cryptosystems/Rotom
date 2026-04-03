"""Tests for matmul_layout_explore (alignment-respecting enumeration)."""

from __future__ import annotations

import pytest

from evolve_openevolve.matmul_layout_explore import (
    explore_matmul_kernels,
    matmul_alignment_pairs_2d2d,
    sorted_alignment_pairs_2d2d,
    dedupe_matmul_options,
)


def test_matmul_alignment_set_matches_rotom_contract():
    assert matmul_alignment_pairs_2d2d() == {(0, None), (1, 0), (None, 1)}


@pytest.mark.openevolve
def test_explore_matmul_nonempty_small():
    opts = explore_matmul_kernels(
        m=4, k=4, n_out=4, n_slots=256, roll_flag=False, network="lan"
    )
    assert len(opts) >= 1
    for o in opts:
        assert o.total_cost_lan < float("inf")
        assert o.output_layout_str
        assert o.operand_a_layout_str
        assert o.operand_b_layout_str


def test_sorted_alignment_pairs_stable():
    s = sorted_alignment_pairs_2d2d()
    assert len(s) == 3
    assert set(s) == matmul_alignment_pairs_2d2d()


def test_dedupe_shrinks_or_equal():
    opts = explore_matmul_kernels(
        m=4, k=4, n_out=4, n_slots=256, roll_flag=False, network="lan"
    )
    deduped = dedupe_matmul_options(opts)
    assert len(deduped) <= len(opts)
