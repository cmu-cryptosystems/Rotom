"""Direct diagonal lowering for matmul and conv2d.

This file is intentionally small and explicit.  It lowers each supported
operator by:

1. enumerating its logical scalar products,
2. finding where each source and target index lives in ciphertext slots,
3. rotating sources into the target slot, and
4. masking/adding products into each target ciphertext.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from ir.he import HEOp, HETerm
from ir.kernel import KernelOp
from lower.layout_cts import LayoutCiphertexts
from util.layout_embedding import Index, PhysicalLane, layout_embedding
from util.layout_embedding import within_shape as _within_shape
from util.shape_util import get_term_shape


DEFAULT_DIRECT_WORK_LIMIT = 1_024


@dataclass(frozen=True)
class DiagonalContributionGroup:
    """One physical HE product shared by a set of target slots."""

    target_ct: int
    a_ct: int
    b_ct: int
    a_rot: int
    b_rot: int
    target_slots: Tuple[int, ...]

    @property
    def lane_count(self) -> int:
        return len(self.target_slots)


@dataclass(frozen=True)
class DiagonalContributionPlan:
    """Exact physical schedule used by both lowering and exact costing."""

    groups: Tuple[DiagonalContributionGroup, ...]
    output_ct_count: int
    slot_count: int
    work: int


def lower_diagonal_matmul(env, kernel):
    """Lower ``DIAGONAL_MATMUL`` to explicit rotations, masks, muls, and adds."""

    return _lower_diagonal(env, kernel, _matmul_contributions)


def lower_diagonal_conv2d(env, kernel):
    """Lower ``DIAGONAL_CONV2D`` to explicit rotations, masks, muls, and adds."""

    return _lower_diagonal(env, kernel, _conv2d_contributions)


def estimate_diagonal_matmul_ops(
    kernel, exact_work_limit: int = DEFAULT_DIRECT_WORK_LIMIT
) -> Dict[str, int]:
    """Estimate the HE ops emitted by ``lower_diagonal_matmul``.

    Small kernels use the same contribution grouping as lowering.  Large
    kernels return a conservative count so layout search does not spend
    ``M*N*K`` time per candidate or accidentally select an enormous direct
    circuit.
    """

    return _estimate_diagonal_ops(kernel, _matmul_contributions, exact_work_limit)


def estimate_diagonal_conv2d_ops(
    kernel, exact_work_limit: int = DEFAULT_DIRECT_WORK_LIMIT
) -> Dict[str, int]:
    """Estimate the HE ops emitted by ``lower_diagonal_conv2d``."""

    return _estimate_diagonal_ops(kernel, _conv2d_contributions, exact_work_limit)


def contribution_plan(kernel) -> DiagonalContributionPlan:
    """Expose exact grouping for tests and research diagnostics."""

    match kernel.op:
        case KernelOp.DIAGONAL_MATMUL | KernelOp.MATMUL:
            return _build_contribution_plan(kernel, _matmul_contributions)
        case KernelOp.DIAGONAL_CONV2D | KernelOp.CONV2D:
            return _build_contribution_plan(kernel, _conv2d_contributions)
        case _:
            raise NotImplementedError(kernel.op)


def direct_diagonal_work(kernel) -> int:
    """Return the logical contribution count without expanding lane groups."""

    match kernel.op:
        case KernelOp.MATMUL | KernelOp.DIAGONAL_MATMUL:
            a_shape, b_shape = _binary_shapes(kernel)
            _require_2d_matmul(a_shape, b_shape)
            return int(a_shape[0] * b_shape[1] * a_shape[1])
        case KernelOp.CONV2D | KernelOp.DIAGONAL_CONV2D:
            input_shape, filter_shape = _binary_shapes(kernel)
            _require_conv2d(input_shape, filter_shape)
            _, out_h, out_w = _conv2d_output_shape(kernel, input_shape, filter_shape)
            return int(
                filter_shape[0]
                * out_h
                * out_w
                * input_shape[0]
                * filter_shape[2]
                * filter_shape[3]
            )
        case _:
            raise NotImplementedError(kernel.op)


def _lower_diagonal(env, kernel, contribution_fn):
    plan = _build_contribution_plan(kernel, contribution_fn)
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]

    out_terms = _terms_from_plan(plan, a_cts, b_cts)
    return LayoutCiphertexts(layout=kernel.layout, cts=out_terms)


def _estimate_diagonal_ops(kernel, contribution_fn, exact_work_limit):
    work = direct_diagonal_work(kernel)
    if work > exact_work_limit:
        # Large direct diagonal circuits can have M*N*K or
        # Cout*H*W*Cin*FH*FW products.  The bounded fallback deliberately
        # over-prices them so layout search never wins by under-estimation.
        return _coarse_expensive_ops(work)

    plan = _build_contribution_plan(kernel, contribution_fn, work=work)
    a_cts = _fake_layout_cts(kernel.cs[0].layout, "a")
    b_cts = _fake_layout_cts(kernel.cs[1].layout, "b")
    out_terms = _terms_from_plan(plan, a_cts, b_cts)
    return _count_secret_ops(out_terms.values())


def _build_contribution_plan(
    kernel,
    contribution_fn: Callable,
    work: Optional[int] = None,
) -> DiagonalContributionPlan:
    a_shape, b_shape = _binary_shapes(kernel)
    out_shape = tuple(int(dim) for dim in get_term_shape(kernel.layout.term))
    a_embedding = layout_embedding(kernel.cs[0].layout, a_shape)
    b_embedding = layout_embedding(kernel.cs[1].layout, b_shape)
    out_embedding = layout_embedding(kernel.layout, out_shape)

    n = kernel.layout.n
    groups: Dict[Tuple[int, int, int, int, int], set[int]] = {}
    contribution_count = 0
    for target_lane in out_embedding.lanes:
        if target_lane.index is None:
            continue
        if not _within_shape(target_lane.index, out_shape):
            continue

        for a_index, b_index in contribution_fn(kernel, target_lane.index):
            a_lane = _choose_source_lane(
                target_lane, a_embedding.by_index.get(a_index, ())
            )
            b_lane = _choose_source_lane(
                target_lane, b_embedding.by_index.get(b_index, ())
            )
            if a_lane is None or b_lane is None:
                continue

            # Toy/OpenFHE use left rotation: rot(v, r)[t] = v[t + r].
            # To move source slot s into target slot t, choose r = s - t mod n.
            a_rot = (a_lane.slot - target_lane.slot) % n
            b_rot = (b_lane.slot - target_lane.slot) % n

            # A group is one reusable HE product.  It is reusable exactly when
            # target ct, both source cts, and both rotations are identical.
            key = (target_lane.ct, a_lane.ct, b_lane.ct, a_rot, b_rot)
            groups.setdefault(key, set()).add(target_lane.slot)
            contribution_count += 1

    physical_groups = tuple(
        DiagonalContributionGroup(
            target_ct=target_ct,
            a_ct=a_ct,
            b_ct=b_ct,
            a_rot=a_rot,
            b_rot=b_rot,
            target_slots=tuple(sorted(target_slots)),
        )
        for (target_ct, a_ct, b_ct, a_rot, b_rot), target_slots in sorted(
            groups.items()
        )
    )
    return DiagonalContributionPlan(
        groups=physical_groups,
        output_ct_count=kernel.layout.num_ct(),
        slot_count=kernel.layout.n,
        work=contribution_count if work is None else work,
    )


def _terms_from_plan(plan, a_cts, b_cts):
    source_cache = {}
    rot_cache = {}
    product_cache = {}
    mask_cache = {}
    target_terms: Dict[int, List[HETerm]] = {
        ct_index: [] for ct_index in range(plan.output_ct_count)
    }

    def source(label, cts, ct_index):
        key = (label, ct_index)
        if key not in source_cache:
            base = cts[ct_index]
            source_cache[key] = HETerm(HEOp.CS, [base], base.secret)
        return source_cache[key]

    def rotate(label, cts, ct_index, rot):
        base = source(label, cts, ct_index)
        if rot == 0:
            return base
        key = (label, ct_index, rot)
        if key not in rot_cache:
            rot_cache[key] = base << rot
        return rot_cache[key]

    def mask_term(slots):
        key = tuple(slots)
        if key not in mask_cache:
            mask = [0] * plan.slot_count
            for slot in slots:
                mask[slot] = 1
            mask_cache[key] = HETerm(HEOp.MASK, [mask], False, "diagonal mask")
        return mask_cache[key]

    for group in plan.groups:
        a_term = rotate("a", a_cts, group.a_ct, group.a_rot)
        b_term = rotate("b", b_cts, group.b_ct, group.b_rot)
        product_key = (a_term, b_term)
        if product_key not in product_cache:
            product_cache[product_key] = a_term * b_term
        term = product_cache[product_key]

        # The mask is the characteristic vector of this group's target slots.
        # Full-ct groups need no mask: multiplying by all ones is a no-op.
        if group.lane_count < plan.slot_count:
            term = term * mask_term(group.target_slots)
        target_terms[group.target_ct].append(term)

    out_terms = {}
    for ct_index in range(plan.output_ct_count):
        terms = target_terms[ct_index]
        if not terms:
            out_terms[ct_index] = HETerm(HEOp.ZERO_MASK, [], False)
            continue
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        out_terms[ct_index] = total
    return out_terms


def _matmul_contributions(_kernel, out_index: Index) -> Iterable[Tuple[Index, Index]]:
    """Yield the scalar products C[m,n] += A[m,k] * B[k,n]."""

    a_shape, b_shape = _binary_shapes(_kernel)
    _require_2d_matmul(a_shape, b_shape)
    m, n = out_index
    for k in range(a_shape[1]):
        yield (m, k), (k, n)


def _conv2d_contributions(kernel, out_index: Index) -> Iterable[Tuple[Index, Index]]:
    """Yield Y[co,h,w] += X[ci,h*stride+kh-pad,w*stride+kw-pad] * W[...]."""

    input_shape, filter_shape = _binary_shapes(kernel)
    _require_conv2d(input_shape, filter_shape)
    _c_out, out_h, out_w = _conv2d_output_shape(kernel, input_shape, filter_shape)
    co, h_out, w_out = out_index
    stride = int(kernel.layout.term.cs[2])
    pad_top, pad_left = _conv2d_top_left_padding(
        input_shape, filter_shape, stride, kernel.layout.term.cs[3], out_h, out_w
    )

    for ci in range(input_shape[0]):
        for kh in range(filter_shape[2]):
            input_h = h_out * stride + kh - pad_top
            if input_h < 0 or input_h >= input_shape[1]:
                continue
            for kw in range(filter_shape[3]):
                input_w = w_out * stride + kw - pad_left
                if input_w < 0 or input_w >= input_shape[2]:
                    continue
                yield (ci, input_h, input_w), (co, ci, kh, kw)


def _binary_shapes(kernel) -> Tuple[Index, Index]:
    return (
        tuple(int(dim) for dim in get_term_shape(kernel.cs[0].layout.term)),
        tuple(int(dim) for dim in get_term_shape(kernel.cs[1].layout.term)),
    )


def _require_2d_matmul(a_shape: Index, b_shape: Index):
    if len(a_shape) != 2 or len(b_shape) != 2 or a_shape[1] != b_shape[0]:
        raise NotImplementedError(
            f"diagonal matmul currently supports 2D x 2D, got {a_shape} @ {b_shape}"
        )


def _require_conv2d(input_shape: Index, filter_shape: Index):
    if len(input_shape) != 3 or len(filter_shape) != 4:
        raise NotImplementedError(
            f"diagonal conv2d supports [Cin,H,W] x [Cout,Cin,FH,FW], got {input_shape} and {filter_shape}"
        )
    if input_shape[0] != filter_shape[1]:
        raise ValueError(
            f"conv2d channel mismatch: {input_shape[0]} input vs {filter_shape[1]} filter"
        )


def _conv2d_output_shape(kernel, input_shape: Index, filter_shape: Index) -> Index:
    padding = kernel.layout.term.cs[3]
    stride = int(kernel.layout.term.cs[2])
    if padding == "valid":
        out_h = (input_shape[1] - filter_shape[2]) // stride + 1
        out_w = (input_shape[2] - filter_shape[3]) // stride + 1
    elif padding == "same":
        out_h = (input_shape[1] + stride - 1) // stride
        out_w = (input_shape[2] + stride - 1) // stride
    else:
        raise ValueError(f"unsupported conv2d padding: {padding}")
    return filter_shape[0], out_h, out_w


def _conv2d_top_left_padding(
    input_shape: Index,
    filter_shape: Index,
    stride: int,
    padding: str,
    out_h: int,
    out_w: int,
) -> Tuple[int, int]:
    if padding == "valid":
        return 0, 0
    if padding != "same":
        raise ValueError(f"unsupported conv2d padding: {padding}")

    # Match TensorTerm.eval_conv2d exactly: for even total padding the halves
    # are equal; for odd total padding, the extra cell is on bottom/right.
    pad_top = math.floor((stride * (out_h - 1) - input_shape[1] + filter_shape[2]) / 2)
    pad_left = math.floor((stride * (out_w - 1) - input_shape[2] + filter_shape[3]) / 2)
    return pad_top, pad_left


def _choose_source_lane(
    target_lane: PhysicalLane, source_lanes: Tuple[PhysicalLane, ...]
) -> Optional[PhysicalLane]:
    if not source_lanes:
        return None
    for source_lane in source_lanes:
        if source_lane.ct == target_lane.ct and source_lane.slot == target_lane.slot:
            return source_lane
    for source_lane in source_lanes:
        if source_lane.ct == target_lane.ct:
            return source_lane
    return source_lanes[0]


def _fake_layout_cts(layout, label):
    cts = {}
    for ct_index in range(layout.num_ct()):
        cts[ct_index] = HETerm(
            HEOp.PACK,
            [layout],
            layout.secret,
            f"{ct_index} diagonal-{label}",
        )
    return LayoutCiphertexts(layout=layout, cts=cts)


def _count_secret_ops(terms) -> Dict[str, int]:
    ops = {"add": 0, "mul": 0, "rot": 0}
    seen = set()
    for term in terms:
        for node in term.post_order():
            if node in seen or not node.secret:
                continue
            seen.add(node)
            match node.op:
                case HEOp.ADD:
                    ops["add"] += 1
                case HEOp.MUL:
                    ops["mul"] += 1
                case HEOp.ROT:
                    ops["rot"] += 1
    return ops


def _coarse_expensive_ops(work: int) -> Dict[str, int]:
    # This is deliberately conservative for layout search.  It is still
    # bounded and monotone in the logical contribution count, while avoiding
    # exact expansion for large M*N*K or Cout*H*W*Cin*FH*FW kernels.
    return {"add": int(work), "mul": int(work), "rot": int(work)}
