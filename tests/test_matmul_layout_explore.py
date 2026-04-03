"""Tests for matmul_layout_explore (alignment-respecting enumeration)."""

from __future__ import annotations

import pytest

from ir.layout_utils import dimension_merging

from evolve_openevolve.matmul_layout_explore import (
    _operand_neighborhood_kernels,
    collect_matmul_kernel_objects,
    explore_matmul_kernels,
    local_search_matmul_kernels,
    matmul_alignment_pairs_2d2d,
    sorted_alignment_pairs_2d2d,
    dedupe_matmul_options,
    try_lower_matmul_kernel,
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
        assert o.source == "enumerate"


@pytest.mark.openevolve
def test_local_search_superset_and_sources():
    enum_opts = explore_matmul_kernels(
        m=4, k=4, n_out=4, n_slots=256, roll_flag=False, network="lan"
    )
    local_opts = local_search_matmul_kernels(
        m=4,
        k=4,
        n_out=4,
        n_slots=256,
        roll_flag=False,
        network="lan",
        max_local_attempts=5000,
    )
    assert len(local_opts) >= len(enum_opts)
    min_enum = min(o.total_cost_lan for o in enum_opts)
    min_local = min(o.total_cost_lan for o in local_opts)
    assert min_local <= min_enum
    sources = {o.source for o in local_opts}
    assert "enumerate" in sources
    assert sources & {"local_A", "local_B"}


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


@pytest.mark.openevolve
def test_operand_neighborhood_wide_is_superset_of_default():
    _term, _align, kernels = collect_matmul_kernel_objects(
        m=4, k=4, n_out=4, n_slots=256, roll_flag=False
    )
    assert kernels
    k0 = kernels[0]
    def _sig(ks):
        return {dimension_merging(x.layout).layout_str() for x in ks}

    d = list(_operand_neighborhood_kernels(k0, mode="default"))
    w = list(_operand_neighborhood_kernels(k0, mode="wide"))
    assert len(w) >= len(d)
    assert _sig(d).issubset(_sig(w))


@pytest.mark.openevolve
def test_per_seed_cap_reduces_work():
    loose = local_search_matmul_kernels(
        m=4,
        k=4,
        n_out=4,
        n_slots=256,
        roll_flag=False,
        network="lan",
        max_local_attempts=8000,
        max_attempts_per_seed=None,
    )
    tight = local_search_matmul_kernels(
        m=4,
        k=4,
        n_out=4,
        n_slots=256,
        roll_flag=False,
        network="lan",
        max_local_attempts=8000,
        max_attempts_per_seed=8,
    )
    assert len(tight) <= len(loose)


@pytest.mark.openevolve
def test_collect_kernel_refs_aligned_with_options():
    opts, kerns = local_search_matmul_kernels(
        m=4,
        k=4,
        n_out=4,
        n_slots=256,
        roll_flag=False,
        network="lan",
        max_local_attempts=800,
        collect_kernel_refs=True,
    )
    assert len(opts) == len(kerns)


@pytest.mark.openevolve
def test_try_lower_runs_on_gen_binop_kernel():
    _term, _align, kernels = collect_matmul_kernel_objects(
        m=4, k=4, n_out=4, n_slots=256, roll_flag=False
    )
    assert kernels
    for k in kernels[:5]:
        r = try_lower_matmul_kernel(k)
        assert "ok" in r


def test_matmul_layout_evaluator_smoke():
    from pathlib import Path

    from evolve_openevolve.matmul_layout_evaluator import evaluate

    prog = Path(__file__).resolve().parent.parent / "evolve_openevolve" / "matmul_layout_program.py"
    res = evaluate(str(prog))
    assert res.metrics.get("error", 0) == 0
    assert res.metrics["combined_score"] >= 1.0 - 1e-9
