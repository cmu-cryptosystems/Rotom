"""Tests for OpenEvolve layout-transform harness (TENSOR permute / roll search)."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.openevolve


def test_layout_transform_identity_evaluator():
    from evolve_openevolve.layout_transform_evaluator import evaluate
    from evolve_openevolve.layout_transform_search import baseline_program_path

    r = evaluate(str(baseline_program_path()))
    assert r.metrics["combined_score"] == pytest.approx(1.0, rel=0, abs=1e-5)
    assert r.metrics["baseline_cost"] == pytest.approx(r.metrics["evolved_cost"], rel=0, abs=1e-5)


def test_apply_permute_two_dims_semantic_and_cost():
    from ir.kernel import Kernel, KernelOp

    from evolve_openevolve.layout_transform_lib import apply_permute_traversal_dims
    from evolve_openevolve.layout_transform_search import (
        build_seed_tensor_kernel,
        evaluate_transformed_kernel,
    )

    term, seed, shape = build_seed_tensor_kernel(shape=(8, 8))

    def propose(k):
        n = len(k.layout.dims)
        # Seed may include a gap dim (e.g. [0:8][1:8][G:64]); swap the two tensor axes only.
        if n >= 3:
            perm = (1, 0) + tuple(range(2, n))
        elif n == 2:
            perm = (1, 0)
        else:
            perm = (0,)
        nl = apply_permute_traversal_dims(k.layout, perm)
        return Kernel(k.op, k.cs, nl)

    out = evaluate_transformed_kernel(term, seed, shape, propose)
    assert out.get("ok") is True, out
    assert out["semantic_ok"] is True
    assert out["total_cost"] < float("inf")


def test_apply_append_roll_invalid_extent_fails_cleanly():
    from ir.kernel import Kernel

    from evolve_openevolve.layout_transform_lib import apply_append_roll
    from evolve_openevolve.layout_transform_search import (
        build_seed_tensor_kernel,
        evaluate_transformed_kernel,
    )

    term, seed, shape = build_seed_tensor_kernel(shape=(8, 8))

    def propose(k):
        # same dim twice — invalid
        nl = apply_append_roll(k.layout, 0, 0)
        return Kernel(k.op, k.cs, nl)

    out = evaluate_transformed_kernel(term, seed, shape, propose)
    assert out.get("ok") is not True
