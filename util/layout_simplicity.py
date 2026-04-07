"""Optional layout-assignment bias toward less fragmented ciphertext packings.

``Layout`` automatically inserts ``[G:k]`` **EMPTY** slot dimensions when the
logical tensor does not fill all ``n`` SIMD slots (see ``ir.layout.Layout``).
Those fillers are not optional "slop" in the optimizer sense—they are required
for a fixed ``n``—but **multiple** ``G`` segments or extra ciphertext axes often
correlate with more complex packings. A small additive penalty lets you
**explore** tradeoffs without changing the core cost model.

Use ``args.layout_simplicity_weight`` or environment variable
``ROTOM_LAYOUT_SIMPLICITY_WEIGHT`` (see ``LayoutAssignment``).
"""

from __future__ import annotations

from ir.dim import DimType
from ir.layout import Layout


def channel_dim_leading_gap_alignment_penalty(layout: Layout) -> float:
    """Penalty for 3D (CHW-style) packings where dim ``0`` is not the first **data** axis after leading ``G`` gaps.

    Convolution sums over input channels (dim ``0``). Packing channel **immediately** after
    the initial ``DimType.EMPTY`` (``[G:*]``) run often lines that reduction axis up with SIMD
    padding and avoids channel-last orderings like ``[G:8][1:32:1][2:32:1][0:4:1]`` when
    the global cost model **ties** across input permutations.

    Returns ``0.0`` when the first non-``EMPTY`` dimension in ``ct_dims + slot_dims`` is
    logical dim ``0``, else ``1.0``. Non-applicable layouts return ``0.0``.
    """
    seq = layout.ct_dims + layout.slot_dims
    i = 0
    while i < len(seq) and seq[i].dim_type == DimType.EMPTY:
        i += 1
    if i >= len(seq):
        return 0.0
    d = seq[i]
    if d.dim_type == DimType.EMPTY or d.dim is None:
        return 0.0
    return 0.0 if d.dim == 0 else 1.0


def embedded_secret_3d_channel_gap_penalty(
    kernel, secret_map: dict, padded_shapes: dict
) -> float:
    """Sum :func:`channel_dim_leading_gap_alignment_penalty` over embedded ``TENSOR`` kernels.

    After replication / roll optimization, operands are often inlined under e.g. ``CONV2D``
    instead of ``CS`` placeholders. In that case ``LayoutAssignment.update_kernels`` does not
    add the operand term's cached ``kernel_costs`` row, so input-only biases must be applied
    by walking the kernel tree.
    """
    from ir.kernel import KernelOp

    total = 0.0
    for k in kernel.post_order():
        if k.op != KernelOp.TENSOR:
            continue
        term = k.layout.term
        if not secret_map.get(term, False):
            continue
        sh = padded_shapes.get(term)
        if sh is None or len([x for x in sh if x > 1]) != 3:
            continue
        total += channel_dim_leading_gap_alignment_penalty(k.layout)
    return total


def layout_simplicity_penalty(layout: Layout) -> float:
    """Return a non-negative score; larger means more structural fragmentation.

    Components:

    - One point per ``DimType.EMPTY`` **slot** dimension (the ``[G:*]`` groups).
    - A quarter point per ciphertext axis beyond the first (penalizes splitting
      the logical tensor across many CT dimensions).
    """
    g_groups = sum(1 for d in layout.slot_dims if d.dim_type == DimType.EMPTY)
    ct_axes = len(layout.ct_dims)
    extra_ct = max(0, ct_axes - 1)
    return float(g_groups) + 0.25 * float(extra_ct)
