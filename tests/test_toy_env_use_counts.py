"""Toy backend: def-use style ``env`` eviction (incoming use counts)."""

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


def test_toy_zero_short_circuit_mul_add_rot_sub() -> None:
    """Padded-channel / mask zeros: Toy must match full numpy but may skip inner work."""
    from ir.he import HEOp, HETerm

    from backends.toy import Toy, _toy_vec_all_zero

    n = 128
    args = get_default_args()
    args.n = n
    args.skip_toy_eval_checks = True

    rng = np.random.default_rng(42)
    a = rng.standard_normal(n).astype(np.float64)
    z = np.zeros(n, dtype=np.float64)

    assert _toy_vec_all_zero(z)
    assert not _toy_vec_all_zero(a)

    toy = Toy({}, {}, args)
    # HETerm identity is hash-based; two ZERO_MASK leaves with empty cs collide.
    inner = HETerm(HEOp.ZERO_MASK, [], False, "")
    leaf_a = HETerm(HEOp.CS, [inner], True, "wrap_a")
    leaf_z = HETerm(HEOp.CS, [inner], True, "wrap_z")
    assert leaf_a != leaf_z
    toy.env[leaf_a] = a
    toy.env[leaf_z] = z

    mul_t = HETerm(HEOp.MUL, [leaf_a, leaf_z], True, "")
    assert np.allclose(toy.eval_mul(mul_t), a * z)

    add_t = HETerm(HEOp.ADD, [leaf_a, leaf_z], True, "")
    assert np.allclose(toy.eval_add(add_t), a + z)

    sub_t = HETerm(HEOp.SUB, [leaf_a, leaf_z], True, "")
    assert np.allclose(toy.eval_sub(sub_t), a - z)

    rot_inner = HETerm(HEOp.ZERO_MASK, [], False, "rz")
    rot_wrap = HETerm(HEOp.CS, [rot_inner], True, "wrap_rot_z")
    toy.env[rot_wrap] = z
    rot_t = HETerm(HEOp.ROT, [rot_wrap, 17], True, "")
    assert np.allclose(toy.eval_rot(rot_t), np.roll(z, -17))
