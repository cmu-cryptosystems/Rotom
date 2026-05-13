"""Diagonal action analysis for layout assignment.

The key object is a canonical logical action plan: a tensor operator is written
as a small set of index shifts before any particular ciphertext packing is
chosen.  A layout embedding then maps each logical shift into concrete
``(source_ct, target_ct, rotation)`` pieces, which lets layout assignment favor
packings where diagonals become native HE rotations instead of many masks,
adds, or cross-ciphertext moves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

from frontends.tensor import TensorOp, TensorTerm
from ir.dim import DimType
from ir.layout import Layout
from util.layout_embedding import (
    Index,
    LayoutEmbedding,
    PhysicalLane,
    layout_embedding,
    within_shape as _within_shape,
)
from util.util import prod


DEFAULT_EXACT_WORK_LIMIT = 262_144


@dataclass(frozen=True)
class LogicalShift:
    """A logical shift in tensor-index space.

    ``offset`` is interpreted as ``source_index = target_index + offset``.
    This matches stencil-style diagonalization, where each output lane pulls
    from a shifted source lane before applying a diagonal/plaintext mask.
    """

    offset: Index
    label: str = ""


@dataclass(frozen=True)
class DiagonalActionPlan:
    """Canonical logical movement for a tensor operator."""

    op: TensorOp
    output_shape: Index
    source_shape: Index
    target_shape: Index
    lane_shape: Index
    shifts: Tuple[LogicalShift, ...]
    reduction_axes: Tuple[int, ...] = ()
    description: str = ""


@dataclass(frozen=True)
class PhysicalMoveGroup:
    """A group of lanes sharing one physical displacement."""

    source_ct: int
    target_ct: int
    rotation: int
    lane_count: int

    @property
    def crosses_ciphertexts(self) -> bool:
        return self.source_ct != self.target_ct


@dataclass(frozen=True)
class ShiftSchedule:
    """Physical realization of one logical shift under a layout."""

    shift: LogicalShift
    groups: Tuple[PhysicalMoveGroup, ...]
    active_lanes: int
    skipped_lanes: int
    missing_source_lanes: int

    @property
    def is_single_native_rotation(self) -> bool:
        if not self.groups:
            return False
        if any(group.crosses_ciphertexts for group in self.groups):
            return False
        rotations = {group.rotation for group in self.groups}
        return len(rotations) == 1


def _groups_by_target_ct(
    groups: Tuple[PhysicalMoveGroup, ...],
) -> Dict[int, Tuple[PhysicalMoveGroup, ...]]:
    grouped: Dict[int, List[PhysicalMoveGroup]] = {}
    for group in groups:
        grouped.setdefault(group.target_ct, []).append(group)
    return {
        target_ct: tuple(target_groups) for target_ct, target_groups in grouped.items()
    }


def _group_needs_mask(
    group: PhysicalMoveGroup,
    target_groups: Tuple[PhysicalMoveGroup, ...],
    boundary_mask: bool,
) -> bool:
    return boundary_mask or group.crosses_ciphertexts or len(target_groups) > 1


@dataclass(frozen=True)
class EmbeddingScore:
    """Aggregate physical-cost signals for a diagonal action plan."""

    schedules: Tuple[ShiftSchedule, ...]
    native_rotation_groups: int
    zero_rotation_groups: int
    masked_rotation_groups: int
    cross_ct_groups: int
    mask_groups: int
    active_lanes: int
    skipped_lanes: int
    missing_source_lanes: int

    def estimated_ops(self) -> Dict[str, int]:
        """Estimate only the extra movement needed by the chosen embedding.

        The operator cost model already counts the plaintext products and the
        reductions.  This method counts the layout-dependent part of a diagonal
        action.  For a logical diagonal ``d`` the mathematical action is
        ``target[t] <- source[t + d]``.  A layout embedding ``phi`` maps this to
        physical groups keyed by ``(source_ct, target_ct, displacement)``.  The
        score uses ``displacement = target_slot - source_slot (mod n)`` as a
        stable grouping label; exact lowering converts to the Toy/OpenFHE
        rotation amount ``source_slot - target_slot (mod n)``.

        Arithmetic used here:
        - same-ct, ``rotation != 0``: one native HE rotation ``rho(rotation)``;
        - boundary diagonal or same-target multi-piece diagonal: one plaintext
          mask multiply per piece to zero lanes outside this diagonal;
        - cross-ct piece: one optional rotation, one mask, one add into the
          target ciphertext accumulator because HE rotations do not change cts;
        - several pieces with the same target ciphertext need ``pieces - 1``
          additions after masking.  Pieces in different target ciphertexts are
          independent and are not added together.
        """

        ops = {"add": 0, "mul": 0, "rot": 0}
        for schedule in self.schedules:
            groups_by_target = _groups_by_target_ct(schedule.groups)
            boundary_mask = (
                schedule.skipped_lanes > 0 or schedule.missing_source_lanes > 0
            )
            for target_groups in groups_by_target.values():
                cross_ct_pieces = 0
                for group in target_groups:
                    if group.crosses_ciphertexts:
                        ops["rot"] += int(group.rotation != 0)
                        ops["mul"] += 1
                        cross_ct_pieces += 1
                        continue

                    ops["rot"] += int(group.rotation != 0)
                    if _group_needs_mask(group, target_groups, boundary_mask):
                        ops["mul"] += 1

                ops["add"] += max(0, len(target_groups) - 1) + cross_ct_pieces

        return ops

    def weighted_cost(self, cost_model: Mapping[str, float]) -> float:
        """Convert ``estimated_ops`` into the same units as ``KernelCost``."""

        return sum(cost_model[op] * count for op, count in self.estimated_ops().items())


def diagonalize_matmul(
    a_shape: Iterable[int], b_shape: Iterable[int]
) -> DiagonalActionPlan:
    """Build a canonical contraction-shift plan for matmul.

    The lane space is the elementwise multiply space used by Rotom after
    replication/alignment.  For a 2D matmul ``A[m, k] @ B[k, n]`` it is
    ``(m, k, n)`` and has one logical diagonal per contraction index.
    """

    a_shape = tuple(int(dim) for dim in a_shape)
    b_shape = tuple(int(dim) for dim in b_shape)
    if not a_shape or not b_shape:
        raise ValueError("matmul diagonalization requires non-scalar operands")
    if a_shape[-1] != b_shape[-2 if len(b_shape) > 1 else -1]:
        raise ValueError(f"incompatible matmul shapes: {a_shape} and {b_shape}")

    contraction_extent = a_shape[-1]
    if len(a_shape) == 1 and len(b_shape) == 1:
        output_shape = (1,)
        lane_shape = (contraction_extent,)
        reduction_axes = (0,)
    elif len(b_shape) == 1:
        output_shape = tuple(a_shape[:-1]) + (1,)
        lane_shape = tuple(a_shape[:-1]) + (contraction_extent,)
        reduction_axes = (len(lane_shape) - 1,)
    elif len(a_shape) == 1:
        output_shape = (1,) + tuple(b_shape[:-2]) + (b_shape[-1],)
        lane_shape = (contraction_extent,) + tuple(b_shape[:-2]) + (b_shape[-1],)
        reduction_axes = (0,)
    else:
        output_shape = tuple(a_shape[:-1]) + tuple(b_shape[:-2]) + (b_shape[-1],)
        lane_shape = (
            tuple(a_shape[:-1])
            + (contraction_extent,)
            + tuple(b_shape[:-2])
            + (b_shape[-1],)
        )
        reduction_axes = (len(a_shape) - 1,)

    shifts = []
    for k in range(contraction_extent):
        offset = [0] * len(lane_shape)
        offset[reduction_axes[0]] = k
        shifts.append(LogicalShift(tuple(offset), f"k={k}"))

    return DiagonalActionPlan(
        op=TensorOp.MATMUL,
        output_shape=output_shape,
        source_shape=lane_shape,
        target_shape=lane_shape,
        lane_shape=lane_shape,
        shifts=tuple(shifts),
        reduction_axes=reduction_axes,
        description="matmul contraction diagonals",
    )


def diagonalize_conv2d(
    input_shape: Iterable[int],
    filter_shape: Iterable[int],
    stride: int,
    padding: str,
) -> DiagonalActionPlan:
    """Build the canonical stencil shifts for 2D convolution.

    The first slice intentionally models spatial movement over the input
    tensor layout.  Input-channel summation and output-channel batching are
    recorded as reduction/output metadata for later integration.
    """

    input_shape = tuple(int(dim) for dim in input_shape)
    filter_shape = tuple(int(dim) for dim in filter_shape)
    if len(input_shape) != 3:
        raise ValueError(f"conv2d input shape must be [C, H, W], got {input_shape}")
    if len(filter_shape) == 3:
        c_out, filter_h, filter_w = filter_shape
        c_in = input_shape[0]
    elif len(filter_shape) == 4:
        c_out, c_in, filter_h, filter_w = filter_shape
    else:
        raise ValueError(
            f"conv2d filter shape must be [Cout, H, W] or [Cout, Cin, H, W], got {filter_shape}"
        )
    if c_in != input_shape[0]:
        raise ValueError(
            f"filter input channels {c_in} do not match input {input_shape[0]}"
        )
    if padding not in {"same", "valid"}:
        raise ValueError(f"unsupported conv2d padding: {padding}")
    if stride <= 0:
        raise ValueError("conv2d stride must be positive")

    _, input_h, input_w = input_shape
    if padding == "valid":
        output_h = (input_h - filter_h) // stride + 1
        output_w = (input_w - filter_w) // stride + 1
        pad_top = 0
        pad_left = 0
    else:
        output_h = (input_h + stride - 1) // stride
        output_w = (input_w + stride - 1) // stride
        total_padding_h = max(0, (output_h - 1) * stride + filter_h - input_h)
        total_padding_w = max(0, (output_w - 1) * stride + filter_w - input_w)
        pad_top = total_padding_h // 2
        pad_left = total_padding_w // 2

    shifts = []
    for kh in range(filter_h):
        for kw in range(filter_w):
            offset = (0, kh - pad_top, kw - pad_left)
            shifts.append(LogicalShift(offset, f"kh={kh},kw={kw}"))

    # The target lane shape keeps input channels because each channel sees the
    # same spatial stencil before channel reduction.
    target_shape = (input_shape[0], output_h, output_w)
    return DiagonalActionPlan(
        op=TensorOp.CONV2D,
        output_shape=(c_out, output_h, output_w),
        source_shape=input_shape,
        target_shape=target_shape,
        lane_shape=input_shape,
        shifts=tuple(shifts),
        reduction_axes=(0,),
        description=f"conv2d spatial stencil, output_channels={c_out}",
    )


def diagonalize_term(
    term: TensorTerm, child_shapes: Iterable[Iterable[int]]
) -> DiagonalActionPlan:
    """Build a diagonal action plan for a supported tensor term."""

    child_shapes = [tuple(shape) for shape in child_shapes]
    if term.op == TensorOp.MATMUL:
        return diagonalize_matmul(child_shapes[0], child_shapes[1])
    if term.op == TensorOp.CONV2D:
        return diagonalize_conv2d(
            child_shapes[0], child_shapes[1], term.cs[2], term.cs[3]
        )
    raise NotImplementedError(f"diagonal action plan not implemented for {term.op}")


def schedule_shift(
    shift: LogicalShift,
    embedding: LayoutEmbedding,
    source_shape: Iterable[int],
    target_shape: Iterable[int],
) -> ShiftSchedule:
    """Schedule one logical shift under a layout embedding."""

    source_shape = tuple(int(dim) for dim in source_shape)
    target_shape = tuple(int(dim) for dim in target_shape)
    if len(shift.offset) != len(source_shape) or len(target_shape) != len(source_shape):
        raise ValueError(
            f"shift rank {len(shift.offset)}, source rank {len(source_shape)}, "
            f"target rank {len(target_shape)} must agree"
        )

    groups: Dict[Tuple[int, int, int], int] = {}
    active_lanes = 0
    skipped_lanes = 0
    missing_source_lanes = 0
    for target_lane in embedding.lanes:
        if target_lane.index is None:
            continue
        if not _within_shape(target_lane.index, target_shape):
            continue

        source_index = tuple(
            target_lane.index[dim] + shift.offset[dim]
            for dim in range(len(source_shape))
        )
        if not _within_shape(source_index, source_shape):
            skipped_lanes += 1
            continue

        source_lane = _choose_source_lane(
            target_lane, embedding.by_index.get(source_index, ())
        )
        if source_lane is None:
            missing_source_lanes += 1
            continue

        # This score only needs a stable physical displacement.  We keep the
        # historical target-minus-source label here; exact lowering uses the
        # Toy/OpenFHE rotation amount source-minus-target.
        rotation = (target_lane.slot - source_lane.slot) % embedding.layout.n
        key = (source_lane.ct, target_lane.ct, rotation)
        groups[key] = groups.get(key, 0) + 1
        active_lanes += 1

    move_groups = tuple(
        PhysicalMoveGroup(source_ct, target_ct, rotation, lane_count)
        for (source_ct, target_ct, rotation), lane_count in sorted(groups.items())
    )
    return ShiftSchedule(
        shift=shift,
        groups=move_groups,
        active_lanes=active_lanes,
        skipped_lanes=skipped_lanes,
        missing_source_lanes=missing_source_lanes,
    )


def score_layout_embedding(plan: DiagonalActionPlan, layout: Layout) -> EmbeddingScore:
    """Score how a layout realizes every logical shift in a plan."""

    embedding = layout_embedding(layout, plan.source_shape)
    schedules = tuple(
        schedule_shift(shift, embedding, plan.source_shape, plan.target_shape)
        for shift in plan.shifts
    )

    native_rotation_groups = 0
    zero_rotation_groups = 0
    masked_rotation_groups = 0
    cross_ct_groups = 0
    mask_groups = 0
    active_lanes = 0
    skipped_lanes = 0
    missing_source_lanes = 0
    for schedule in schedules:
        active_lanes += schedule.active_lanes
        skipped_lanes += schedule.skipped_lanes
        missing_source_lanes += schedule.missing_source_lanes
        boundary_mask = schedule.skipped_lanes > 0 or schedule.missing_source_lanes > 0
        for target_groups in _groups_by_target_ct(schedule.groups).values():
            for group in target_groups:
                needs_mask = _group_needs_mask(group, target_groups, boundary_mask)
                if needs_mask:
                    mask_groups += 1

                if group.crosses_ciphertexts:
                    cross_ct_groups += 1
                elif group.rotation == 0:
                    zero_rotation_groups += 1
                elif schedule.is_single_native_rotation and not needs_mask:
                    native_rotation_groups += 1
                else:
                    masked_rotation_groups += 1

    return EmbeddingScore(
        schedules=schedules,
        native_rotation_groups=native_rotation_groups,
        zero_rotation_groups=zero_rotation_groups,
        masked_rotation_groups=masked_rotation_groups,
        cross_ct_groups=cross_ct_groups,
        mask_groups=mask_groups,
        active_lanes=active_lanes,
        skipped_lanes=skipped_lanes,
        missing_source_lanes=missing_source_lanes,
    )


def estimate_layout_embedding_ops(
    plan: DiagonalActionPlan,
    layout: Layout,
    exact_work_limit: int = DEFAULT_EXACT_WORK_LIMIT,
) -> Dict[str, int]:
    """Estimate diagonal movement ops, using exact scheduling when affordable.

    Exact scoring expands every physical lane for every logical shift.  That is
    the right model for small tensors and stencils, but replicated matmul lanes
    can have ``M*K*N`` positions and ``K`` shifts.  Above ``exact_work_limit`` we
    keep the same arithmetic categories and estimate from layout dimensions.
    The coarse path is intentionally conservative: it counts one rotation per
    independent target ciphertext, masks boundary/carry pieces, and charges
    ciphertext-axis shifts for adds because native rotations cannot change ct id.
    """

    work = lane_count(plan.source_shape) * max(1, len(plan.shifts))
    if work <= exact_work_limit:
        return score_layout_embedding(plan, layout).estimated_ops()
    return _coarse_layout_embedding_ops(plan, layout)


def _coarse_layout_embedding_ops(
    plan: DiagonalActionPlan, layout: Layout
) -> Dict[str, int]:
    ops = {"add": 0, "mul": 0, "rot": 0}
    for shift in plan.shifts:
        axes = tuple(axis for axis, offset in enumerate(shift.offset) if offset)
        if not axes:
            continue

        ct_groups = _ct_group_count(layout)
        slot_pieces = 1
        crosses_ct = False
        rotates_slots = False
        needs_mask = _shift_hits_boundary(shift, plan.source_shape, plan.target_shape)
        for axis in axes:
            axis_slot_pieces = _layout_dim_pieces(layout.slot_dims, axis)
            if axis_slot_pieces:
                rotates_slots = True
                slot_pieces *= len(axis_slot_pieces)
            if _layout_dim_pieces(layout.ct_dims, axis):
                crosses_ct = True
                needs_mask = True

            if not axis_slot_pieces and not _layout_dim_pieces(layout.ct_dims, axis):
                # A missing axis is not a single homomorphism rho(r); treat it
                # as two masked pieces so the optimizer does not prefer it.
                needs_mask = True
                slot_pieces *= 2

        groups = ct_groups * slot_pieces
        if rotates_slots:
            ops["rot"] += groups
        if needs_mask:
            ops["mul"] += groups
        if crosses_ct:
            ops["add"] += groups
        elif slot_pieces > 1:
            ops["add"] += ct_groups * (slot_pieces - 1)

    return ops


def _layout_dim_pieces(dims, axis: int):
    return [
        dim
        for dim in dims
        if dim.dim == axis and dim.dim_type == DimType.FILL and dim.extent > 1
    ]


def _ct_group_count(layout: Layout) -> int:
    # Every independent target ciphertext needs its own rotation/mask.  A shift
    # along a ciphertext dimension is still counted per target ciphertext; the
    # added cross-ct penalty below accounts for the fact that source_ct != target_ct.
    count = 1
    for dim in layout.ct_dims:
        if dim.dim_type == DimType.FILL and dim.extent > 1:
            count *= dim.extent
    return count


def _shift_hits_boundary(
    shift: LogicalShift, source_shape: Index, target_shape: Index
) -> bool:
    for axis, offset in enumerate(shift.offset):
        if offset < 0 or target_shape[axis] - 1 + offset >= source_shape[axis]:
            return True
    return False


def _choose_source_lane(
    target_lane: PhysicalLane, source_lanes: Tuple[PhysicalLane, ...]
) -> Optional[PhysicalLane]:
    if not source_lanes:
        return None
    for source_lane in source_lanes:
        if source_lane.ct == target_lane.ct:
            return source_lane
    return source_lanes[0]


def lane_count(shape: Iterable[int]) -> int:
    """Return the number of logical lanes in a shape."""

    return int(prod(tuple(shape)))
