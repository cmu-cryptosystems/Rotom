"""Bridge TensorTerm.layout strings to layout assignment candidate filtering."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List

from ir.layout import Layout
from ir.layout_utils import dimension_merging

if TYPE_CHECKING:
    from frontends.tensor import TensorTerm


def filter_kernels_by_frontend_layout(
    kernels: List[Any],
    term: "TensorTerm",
    n: int,
    secret: bool,
) -> List[Any]:
    """Keep only kernels whose root layout matches ``term.layout`` when it is set.

    Frontend layout strings are parsed with :meth:`Layout.from_string` using the same
    ``n`` and ``secret`` as assignment, then compared to each candidate via
    :func:`dimension_merging` and :meth:`Layout.layout_str` for a stable string form.

    When ``term.layout`` is ``None`` or blank, ``kernels`` is returned unchanged.
    """
    layout_str = getattr(term, "layout", None)
    if layout_str is None or not str(layout_str).strip():
        return kernels

    layout_str = str(layout_str).strip()
    ref = Layout.from_string(layout_str, n, secret)
    ref.term = term
    target = dimension_merging(ref).layout_str()

    filtered = [
        k for k in kernels if dimension_merging(k.layout).layout_str() == target
    ]

    if not filtered:
        avail = sorted({dimension_merging(k.layout).layout_str() for k in kernels})
        preview = avail[:25]
        suffix = " ..." if len(avail) > 25 else ""
        raise ValueError(
            f"Frontend layout {layout_str!r} for {term!r} does not match any "
            f"candidate layout (n={n}, secret={secret}). "
            f"Expected merged layout_str {target!r}. "
            f"Available ({len(avail)}): {preview}{suffix}"
        )

    return filtered
