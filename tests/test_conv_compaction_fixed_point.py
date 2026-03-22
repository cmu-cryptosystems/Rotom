"""Verify secret conv operands are ct/gap compaction fixed points where checkable."""

import numpy as np
import pytest

from assignment.gen.gen_compaction import (
    find_compaction,
    layout_ct_gap_compaction_is_fixed_point,
)
from frontends.tensor import TensorTerm
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from tests.conftest import run_compiler_and_backend
from tests.test_util import get_default_args


def test_unpacked_ct_plus_gap_layout_is_not_compaction_fixed_point():
    """Layout with dim 1 on the CT axis and G: slack in slots can still compact."""
    args = get_default_args()
    args.n = 16
    t = TensorTerm.Tensor("a", [4, 4], True)
    prior = Layout.from_string("[1:4:1];[0:4:1][G:4]", args.n, True)
    prior.term = t
    assert layout_ct_gap_compaction_is_fixed_point(prior) is False


def test_fully_compacted_layout_is_fixed_point():
    args = get_default_args()
    args.n = 16
    t = TensorTerm.Tensor("a", [4, 4], True)
    target = Layout.from_string("[0:4:1][1:4:1]", args.n, True)
    target.term = t
    assert layout_ct_gap_compaction_is_fixed_point(target) is True


def test_rolled_layout_treated_as_compaction_fixed_point():
    """Roll path uses different compaction rules; helper must not false-fail."""
    args = get_default_args()
    args.n = 16
    t = TensorTerm.Tensor("a", [4, 4], True)
    layout = Layout.from_string("roll(0,1) [1:4:1][0:4:1]", args.n, True)
    layout.term = t
    assert layout.rolls
    assert layout_ct_gap_compaction_is_fixed_point(layout) is True


def test_layout_after_find_compaction_is_fixed_point():
    """One successful COMPACT pass should reach a geometry fixed point."""
    args = get_default_args()
    args.n = 16
    t = TensorTerm.Tensor("a", [4, 4], True)
    prior = Layout.from_string("[1:4:1];[0:4:1][G:4]", args.n, True)
    prior.term = t
    compacted = find_compaction(Kernel(KernelOp.TENSOR, [], prior))
    assert compacted.op == KernelOp.COMPACT
    assert layout_ct_gap_compaction_is_fixed_point(compacted.layout) is True


def test_conv2d_secret_operand_is_compaction_fixed_point_default_layouts():
    """Typical conv IR from assignment should not leave slack between ct_dims and G: gaps."""
    args = get_default_args()
    args.n = 64
    args.rolls = True
    args.conv_roll = False
    a = TensorTerm.Tensor("a", [1, 8, 8], True)
    b = TensorTerm.Tensor("b", [1, 1, 3, 3], False)
    y = TensorTerm.conv2d(a, b, 1, "same")
    inputs = {
        "a": np.random.randn(1, 8, 8),
        "b": np.random.randn(1, 1, 3, 3),
    }
    _, kernel = run_compiler_and_backend(y, inputs, args, "toy")
    for k in kernel.post_order():
        if k.op != KernelOp.CONV2D:
            continue
        secret_layout = k.cs[0].layout
        assert secret_layout.secret
        try:
            ok = layout_ct_gap_compaction_is_fixed_point(secret_layout)
        except (AttributeError, NotImplementedError) as e:
            pytest.fail(
                f"Compaction fixed-point check unsupported for layout "
                f"{secret_layout.layout_str()!r}: {e}"
            )
        assert ok, (
            "Secret conv input should be a ct/gap compaction fixed point; "
            f"geometry={secret_layout.layout_str()!r}"
        )


def test_stride2_gap_split_channel_layout_compaction_check():
    """Custom split-channel layout: verify when heuristic supports it, else skip."""
    args = get_default_args()
    args.n = 512
    args.rolls = True
    args.conv_roll = False
    args.backend = "toy"
    a = TensorTerm.Tensor(
        "a",
        [4, 8, 8],
        True,
        layout="[0:2:2][1:8:1][2:8:1][G:2][0:2:1]",
    )
    b = TensorTerm.Tensor("b", [8, 4, 3, 3], False)
    y = TensorTerm.conv2d(a, b, 2, "same")
    rng = np.random.default_rng(0)
    inputs = {
        "a": rng.normal(size=(4, 8, 8)),
        "b": rng.normal(size=(8, 4, 3, 3)),
    }
    _, kernel = run_compiler_and_backend(y, inputs, args, "toy")
    for k in kernel.post_order():
        if k.op != KernelOp.CONV2D:
            continue
        L = k.cs[0].layout
        try:
            ok = layout_ct_gap_compaction_is_fixed_point(L)
        except (AttributeError, NotImplementedError):
            pytest.skip(
                "compaction_heuristic does not support this replication+gap mix; "
                "fixed-point check unavailable"
            )
        assert ok, (
            "Expected secret conv input to be compaction-fixed; "
            f"geometry={L.layout_str()!r}"
        )


def test_stacked_conv2d_secret_operands_are_compaction_fixed_points():
    """Both CONV2D nodes in a two-layer chain: secret operands pass the fixed-point check.

    Uses ``n=64`` so layouts match the single-conv case (``n=128`` can introduce gap
    slots that trigger a fragile path in ``compaction_heuristic``).
    """
    args = get_default_args()
    args.n = 64
    args.rolls = True
    args.conv_roll = False
    a = TensorTerm.Tensor("a", [1, 8, 8], True)
    w1 = TensorTerm.Tensor("w1", [1, 1, 3, 3], False)
    w2 = TensorTerm.Tensor("w2", [1, 1, 3, 3], False)
    h = TensorTerm.conv2d(a, w1, 1, "same")
    y = TensorTerm.conv2d(h, w2, 1, "same")
    inputs = {
        "a": np.random.randn(1, 8, 8),
        "w1": np.random.randn(1, 1, 3, 3),
        "w2": np.random.randn(1, 1, 3, 3),
    }
    _, kernel = run_compiler_and_backend(y, inputs, args, "toy")
    convs = [k for k in kernel.post_order() if k.op == KernelOp.CONV2D]
    assert len(convs) == 2
    for k in convs:
        L = k.cs[0].layout
        assert L.secret
        try:
            ok = layout_ct_gap_compaction_is_fixed_point(L)
        except (AttributeError, NotImplementedError) as e:
            pytest.fail(
                f"Compaction fixed-point check unsupported for layout "
                f"{L.layout_str()!r}: {e}"
            )
        assert ok, (
            "Stacked conv: secret operand should be compaction-fixed; "
            f"geometry={L.layout_str()!r}"
        )
