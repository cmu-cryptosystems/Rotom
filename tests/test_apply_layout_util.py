"""Robust tests for ``apply_layout`` and ``apply_punctured_layout``.

The golden reference mirrors the legacy element-wise gather semantics (padding,
``None`` / OOB handling, 0-d tensors) without depending on the optimized
implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from util.layout_util import (
    _get_apply_layout_plan,
    apply_layout,
    apply_punctured_layout,
    parse_layout,
)


class _TermWithMasks:
    """Minimal ``term`` stub: ``apply_punctured_layout`` reads ``cs[4]``."""

    __slots__ = ("cs",)

    def __init__(self, masks):
        self.cs = [None, None, None, None, masks]


def _golden_apply_layout(pt_tensor, layout) -> list[list]:
    layout_len = max(len(layout), layout.n)
    pt_tensor_ndim = int(np.ndim(pt_tensor))
    plan = _get_apply_layout_plan(layout, pt_tensor_ndim, layout_len=layout_len)
    base_indices_by_cts = plan["base_indices_by_cts"]

    def _normalize_scalar(value):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.size == 1:
            return value.item()
        if isinstance(value, np.ndarray) and value.ndim == 0:
            return value.item()
        return value

    cts = []
    for ct_indices in base_indices_by_cts:
        ct = []
        for index in ct_indices:
            effective_index = list(index)
            while len(effective_index) < pt_tensor_ndim:
                effective_index.append(0)

            if any(effective_index[i] is None for i in range(pt_tensor_ndim)):
                ct.append(0)
                continue

            if any(
                effective_index[i] >= pt_tensor.shape[i] for i in range(pt_tensor_ndim)
            ):
                ct.append(0)
                continue

            if pt_tensor_ndim == 0:
                value = pt_tensor.item()
            else:
                value = pt_tensor[
                    tuple(effective_index[i] for i in range(pt_tensor_ndim))
                ]
            ct.append(_normalize_scalar(value))
        cts.append(ct)
    return cts


@pytest.mark.parametrize(
    "layout_str,pt_builder",
    [
        ("[R:2:1][0:2:1]", lambda: np.array([1, 2], dtype=np.int32)),
        (
            "[2:3:1][3:3:1];[1:4:1][R:16:1]",
            lambda: np.arange(1, 28, dtype=np.int32).reshape(1, 3, 3, 3),
        ),
        (
            "[0:4:1][1:4:1]",
            lambda: np.random.default_rng(0).standard_normal((4, 4)),
        ),
        (
            "[0:2:1][1:2:1][2:2:1]",
            lambda: np.random.default_rng(1).random((2, 2, 2)).astype(np.float32),
        ),
    ],
)
def test_apply_layout_matches_golden_reference(layout_str: str, pt_builder) -> None:
    layout = parse_layout(layout_str, secret=False)
    pt = pt_builder()
    got = apply_layout(pt, layout)
    want = _golden_apply_layout(pt, layout)
    assert len(got) == len(want)
    for a, b in zip(got, want):
        assert len(a) == len(b)
        np.testing.assert_allclose(
            np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        )


def test_apply_layout_float32_int64_and_fortran_order_match_golden() -> None:
    layout = parse_layout("[0:3:1][1:3:1]", n=8, secret=False)
    for order in "CF":
        pt = np.asarray(
            np.random.default_rng(2).random((3, 3)),
            dtype=np.float32,
            order=order,
        )
        got = apply_layout(pt, layout)
        want = _golden_apply_layout(pt, layout)
        np.testing.assert_allclose(
            np.asarray(got[0], dtype=np.float64),
            np.asarray(want[0], dtype=np.float64),
        )


def test_apply_layout_non_contiguous_transpose_matches_golden() -> None:
    """Regression: stride+ravel(C) fast path broke F-ordered / permuted views (see BERT toy)."""
    layout = parse_layout("[0:2:1][2:4:1][1:4:1]", n=32, secret=False)
    base = np.arange(2 * 4 * 4, dtype=np.float64).reshape(2, 4, 4)
    pt = base.transpose(1, 0, 2)  # non-contiguous
    assert not pt.flags["C_CONTIGUOUS"]
    got = apply_layout(pt, layout)
    want = _golden_apply_layout(pt, layout)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
    )

    pt_i = np.random.default_rng(3).integers(-5, 5, size=(3, 3), dtype=np.int64)
    got_i = apply_layout(pt_i, layout)
    want_i = _golden_apply_layout(pt_i, layout)
    np.testing.assert_equal(np.asarray(got_i[0]), np.asarray(want_i[0]))


def test_apply_layout_scalar_zero_dimensional() -> None:
    layout = parse_layout("[0:1:1]", n=4, secret=False)
    pt = np.array(3.25, dtype=np.float64)
    got = apply_layout(pt, layout)
    want = _golden_apply_layout(pt, layout)
    np.testing.assert_allclose(
        np.asarray(got, dtype=np.float64),
        np.asarray(want, dtype=np.float64),
    )


def test_apply_punctured_layout_all_ones_equals_apply_layout() -> None:
    layout = parse_layout("[0:2:1][1:2:1]", n=4, secret=False)
    pt = np.array([[1.5, -2.0], [0.25, 8.0]], dtype=np.float64)
    base = apply_layout(pt, layout)
    ones = [[1.0] * len(ct) for ct in base]
    layout.term = _TermWithMasks(ones)
    punc = apply_punctured_layout(pt, layout)
    for a, b in zip(punc, base):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_apply_punctured_layout_elementwise_mask() -> None:
    layout = parse_layout("[0:2:1][1:2:1]", n=4, secret=False)
    pt = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    base = apply_layout(pt, layout)
    masks = []
    for i, ct in enumerate(base):
        masks.append([1.0 if (i + j) % 2 == 0 else 0.0 for j in range(len(ct))])
    layout.term = _TermWithMasks(masks)
    got = apply_punctured_layout(pt, layout)
    expected = [
        [base[i][j] * masks[i][j] for j in range(len(base[i]))]
        for i in range(len(base))
    ]
    assert got == expected


def test_apply_punctured_layout_wrong_num_ct_masks_asserts() -> None:
    layout = parse_layout("[0:2:1][1:2:1]", n=4, secret=False)
    pt = np.eye(2, dtype=np.float64)
    base = apply_layout(pt, layout)
    assert len(base) == 1
    # Two mask rows but only one packed ciphertext.
    layout.term = _TermWithMasks([[1.0] * len(base[0]), [1.0] * len(base[0])])
    with pytest.raises(AssertionError):
        apply_punctured_layout(pt, layout)


def test_apply_punctured_layout_wrong_slot_length_raises() -> None:
    layout = parse_layout("[0:2:1][1:2:1]", n=4, secret=False)
    pt = np.eye(2, dtype=np.float64)
    base = apply_layout(pt, layout)
    bad_masks = [[1.0] * (len(base[0]) - 1)]  # one slot short
    layout.term = _TermWithMasks(bad_masks)
    with pytest.raises(ValueError):
        apply_punctured_layout(pt, layout)


def test_apply_layout_plan_cache_stable_across_calls() -> None:
    layout = parse_layout("[0:2:1][1:2:1]", n=4, secret=False)
    pt1 = np.ones((2, 2), dtype=np.float64)
    pt2 = np.full((2, 2), 7.0, dtype=np.float64)
    a = apply_layout(pt1, layout)
    b = apply_layout(pt2, layout)
    assert not np.allclose(a[0], b[0])
    assert np.allclose(a[0], [1.0] * len(a[0]))
    assert np.allclose(b[0], [7.0] * len(b[0]))
