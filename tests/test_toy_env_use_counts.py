"""Toy backend: def-use style ``env`` eviction (incoming use counts)."""

import copy

import numpy as np
import torch

from assignment.assignment import LayoutAssignment
from backends.toy import Toy
from frontends.tensor import TensorTerm
from lower.lower import Lower
from tests.test_util import get_default_args


def test_toy_env_empty_after_run() -> None:
    """After ``run()``, ciphertext evaluation should not leave HE values in ``env``."""
    torch.manual_seed(0)
    np.random.seed(0)

    x = np.random.randn(4, 8, 8).astype(np.float64)
    inputs = {"x": x}
    t = TensorTerm.Tensor("x", [4, 8, 8], True)
    tensor_term = t.sum(dim_idx=1).sum(dim_idx=1)

    args = get_default_args()
    args.backend = "toy"
    args.n = 256
    args.rolls = False
    args.skip_toy_eval_checks = True

    kernel = LayoutAssignment(tensor_term, args).run()
    circuit_ir = Lower(kernel).run()
    toy = Toy(circuit_ir, inputs, args)
    toy.run()

    assert toy.env == {}, "per-ct use-count eviction should leave env empty"


def test_toy_post_order_includes_cs_wrapped_chain() -> None:
    """Sanity: extended post-order visits nodes under ``HEOp.CS``."""
    from ir.he import HEOp, HETerm

    from backends.toy import _toy_post_order

    # Minimal fake layout leaf: PACK won't be evaluated here; we only check order.
    inner = HETerm(HEOp.ZERO_MASK, [], False, "")
    wrapped = HETerm(HEOp.CS, [inner], True, "")
    order = _toy_post_order(wrapped)
    assert inner in order and wrapped in order
    assert order.index(inner) < order.index(wrapped)


def test_toy_post_order_deep_linear_chain_no_recursion_limit() -> None:
    """Deep unary chains must not use recursive traversal (ResNet-scale IR depth)."""
    from ir.he import HEOp, HETerm

    from backends.toy import _toy_post_order

    depth = 5_000
    t = HETerm(HEOp.ZERO_MASK, [], False, "")
    for _ in range(depth):
        t = HETerm(HEOp.ROT, [t, 0], True, "")

    order = _toy_post_order(t)
    assert len(order) == depth + 1


def test_toy_dense_eval_memoizes_per_tensor_term_identity() -> None:
    """Same TensorTerm object must not trigger repeated ``.eval`` for packing."""
    from backends.toy import Toy

    class _Term:
        eval_calls = 0

        def eval(self, inputs):
            _Term.eval_calls += 1
            return np.asarray([_Term.eval_calls], dtype=np.float64)

    term = _Term()
    toy = object.__new__(Toy)
    toy.inputs = {}
    toy._tensor_term_dense_by_id = {}

    a = Toy._dense_eval_for_tensor_term(toy, term)
    b = Toy._dense_eval_for_tensor_term(toy, term)
    assert _Term.eval_calls == 1
    assert a is b


def test_toy_parallel_ct_workers_matches_sequential() -> None:
    """``toy_ct_workers`` parallel path should match sequential results bit-for-bit."""
    torch.manual_seed(0)
    np.random.seed(0)

    x = np.random.randn(4, 8, 8).astype(np.float64)
    inputs = {"x": x}
    t = TensorTerm.Tensor("x", [4, 8, 8], True)
    tensor_term = t.sum(dim_idx=1).sum(dim_idx=1)

    args = get_default_args()
    args.backend = "toy"
    args.n = 256
    args.rolls = False
    args.skip_toy_eval_checks = True

    kernel = LayoutAssignment(tensor_term, args).run()
    circuit_ir = Lower(kernel).run()

    a_args = copy.copy(args)
    a_args.toy_ct_workers = 1
    seq = Toy(circuit_ir, inputs, a_args).run()

    b_args = copy.copy(args)
    b_args.toy_ct_workers = 0
    par = Toy(circuit_ir, inputs, b_args).run()

    assert len(seq) == len(par)
    for row_a, row_b in zip(seq, par):
        np.testing.assert_allclose(
            np.asarray(row_a, dtype=np.float64),
            np.asarray(row_b, dtype=np.float64),
        )
