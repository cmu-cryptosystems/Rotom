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


def _two_layer_circuit():
    """A program whose kernels read each other's ciphertexts."""
    rng = np.random.default_rng(0)
    inputs = {
        "x": rng.normal(size=(4, 8, 8)),
        "w1": rng.normal(size=(4, 4, 3, 3)),
        "w2": rng.normal(size=(4, 4, 3, 3)),
    }
    x = TensorTerm.Tensor("x", [4, 8, 8], True)
    w1 = TensorTerm.Tensor("w1", [4, 4, 3, 3], False)
    w2 = TensorTerm.Tensor("w2", [4, 4, 3, 3], False)
    y = TensorTerm.conv2d(x, w1, 1, "same")
    tensor_term = TensorTerm.conv2d(y, w2, 1, "same") + y

    args = get_default_args()
    args.backend = "toy"
    args.n = 256
    kernel = LayoutAssignment(tensor_term, args).run()
    return Lower(kernel).run(), inputs, args, tensor_term


def test_toy_evaluates_each_term_once() -> None:
    """Every kernel is checked, yet no term is computed twice."""
    from backends.toy import _flatten_ct_roots, _toy_post_order

    circuit_ir, inputs, args, _ = _two_layer_circuit()
    assert len(circuit_ir) > 3

    evaluated = []

    class Counting(Toy):
        def eval(self, term):
            evaluated.append(term)
            return super().eval(term)

    toy = Counting(circuit_ir, inputs, args)
    toy.run()

    seen = set()
    for cts in circuit_ir.values():
        for root in _flatten_ct_roots(cts):
            _toy_post_order(root, seen)
    assert len(evaluated) == len(set(evaluated)) == len(seen)
    assert toy.env == {}


def test_toy_evaluates_each_tensor_term_once() -> None:
    """The expected value of a kernel reuses the values of the kernels before it."""
    from frontends.tensor_evaluator import TensorEvaluator

    circuit_ir, inputs, args, tensor_term = _two_layer_circuit()
    evaluated = []

    class Counting(TensorEvaluator):
        def eval_term(self, term, env, inputs):
            evaluated.append(term)
            return super().eval_term(term, env, inputs)

    toy = Toy(circuit_ir, inputs, args)
    toy._evaluator = Counting()
    toy.run()

    program = set(tensor_term.post_order())
    counted = [term for term in evaluated if term in program]
    assert len(counted) == len(set(counted))


def test_toy_run_matches_tensor_eval() -> None:
    circuit_ir, inputs, args, tensor_term = _two_layer_circuit()
    kernel = list(circuit_ir)[-1]
    results = Toy(circuit_ir, inputs, args).run()

    from util.layout_util import apply_layout

    expected = apply_layout(tensor_term.eval(inputs), kernel.layout)
    np.testing.assert_allclose(np.asarray(results), np.asarray(expected), atol=1e-9)


def test_toy_reports_the_kernel_that_is_wrong() -> None:
    """A wrong ciphertext in the middle of the circuit fails at its own kernel."""
    import pytest
    from ir.he import HEOp, HETerm
    from ir.kernel import KernelOp

    circuit_ir, inputs, args, _ = _two_layer_circuit()
    kernel = next(k for k in circuit_ir if k.op == KernelOp.CONV_DIRECT)
    cts = circuit_ir[kernel].cts
    cts[sorted(cts)[0]] = HETerm(HEOp.ZERO_MASK, [], True, "")

    with pytest.raises(AssertionError, match="Toy mismatch at KernelOp.CONV_DIRECT"):
        Toy(circuit_ir, inputs, args).run()
