from functools import reduce
from operator import mul

from hypothesis import given, settings, strategies as st

from assignment.diagonal import (
    DiagonalActionPlan,
    LogicalShift,
    estimate_layout_embedding_ops,
    layout_embedding,
    schedule_shift,
    score_layout_embedding,
)
from frontends.tensor import TensorOp, TensorTerm
from ir.dim import Dim
from ir.layout import Layout
from lower.lower import Lower
from lower.lower_diagonal import contribution_plan, estimate_diagonal_matmul_ops
from tests.diagonal_helpers import direct_matmul_kernel, secret_arith_ops


def _product(values):
    return reduce(mul, values, 1)


def _within_shape(index, shape):
    return len(index) == len(shape) and all(
        0 <= value < shape[axis] for axis, value in enumerate(index)
    )


@st.composite
def layout_shift_cases(draw):
    rank = draw(st.integers(min_value=1, max_value=3))
    shape = tuple(
        draw(st.lists(st.sampled_from([2, 4, 8]), min_size=rank, max_size=rank))
    )
    slot_suffix_len = draw(st.integers(min_value=1, max_value=rank))
    n = _product(shape[rank - slot_suffix_len :])
    layout = Layout(
        None,
        [],
        [Dim(axis, extent) for axis, extent in enumerate(shape)],
        n,
    )
    offset = tuple(
        draw(st.integers(min_value=1 - extent, max_value=extent - 1))
        for extent in shape
    )
    plan = DiagonalActionPlan(
        op=TensorOp.CONV2D,
        output_shape=shape,
        source_shape=shape,
        target_shape=shape,
        lane_shape=shape,
        shifts=(LogicalShift(offset, "generated"),),
    )
    return layout, plan


@given(layout_shift_cases())
@settings(max_examples=150, deadline=None)
def test_shift_schedule_partition_and_rotation_invariants(case):
    layout, plan = case
    embedding = layout_embedding(layout, plan.source_shape)
    schedule = schedule_shift(
        plan.shifts[0], embedding, plan.source_shape, plan.target_shape
    )
    valid_target_lanes = sum(
        1
        for lane in embedding.lanes
        if lane.index is not None and _within_shape(lane.index, plan.target_shape)
    )

    assert sum(group.lane_count for group in schedule.groups) == schedule.active_lanes
    assert (
        schedule.active_lanes + schedule.skipped_lanes + schedule.missing_source_lanes
        == valid_target_lanes
    )
    assert all(0 <= group.rotation < layout.n for group in schedule.groups)


@given(layout_shift_cases())
@settings(max_examples=150, deadline=None)
def test_embedding_score_ops_are_nonnegative_and_stable(case):
    layout, plan = case

    first_score = score_layout_embedding(plan, layout)
    second_score = score_layout_embedding(plan, layout)
    first_ops = first_score.estimated_ops()
    second_ops = second_score.estimated_ops()

    assert first_ops == second_ops
    assert set(first_ops) == {"add", "mul", "rot"}
    assert all(count >= 0 for count in first_ops.values())
    assert estimate_layout_embedding_ops(plan, layout) == first_ops


@given(st.sampled_from([2, 4]), st.booleans())
@settings(max_examples=20, deadline=None)
def test_direct_matmul_grouping_covers_every_contribution(size, b_secret):
    kernel = _direct_matmul_kernel(size, b_secret)
    plan = contribution_plan(kernel)

    assert plan.work == size * size * size
    assert sum(group.lane_count for group in plan.groups) == plan.work
    assert all(0 <= group.a_rot < kernel.layout.n for group in plan.groups)
    assert all(0 <= group.b_rot < kernel.layout.n for group in plan.groups)
    assert all(
        0 <= slot < kernel.layout.n
        for group in plan.groups
        for slot in group.target_slots
    )


@given(st.sampled_from([2, 4]), st.booleans())
@settings(max_examples=20, deadline=None)
def test_direct_matmul_estimator_is_stable_and_matches_exact_lowering(size, b_secret):
    kernel = _direct_matmul_kernel(size, b_secret)
    first = estimate_diagonal_matmul_ops(kernel)
    second = estimate_diagonal_matmul_ops(kernel)

    lower = Lower(kernel)
    lower.lower()

    assert first == second
    assert set(first) == {"add", "mul", "rot"}
    assert all(count >= 0 for count in first.values())
    assert first == secret_arith_ops(lower.env[kernel].cts.values())


def _direct_matmul_kernel(size, b_secret):
    a = TensorTerm.Tensor("a", [size, size], True)
    b = TensorTerm.Tensor("b", [size, size], b_secret)
    return direct_matmul_kernel(a, b, a @ b)
