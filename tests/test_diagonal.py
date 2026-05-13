from types import SimpleNamespace

import numpy as np

import assignment.diagonal as diagonal
from assignment.assignment import LayoutAssignment
from assignment.diagonal import (
    DiagonalActionPlan,
    LogicalShift,
    diagonalize_conv2d,
    diagonalize_matmul,
    diagonalize_term,
    estimate_layout_embedding_ops,
    layout_embedding,
    score_layout_embedding,
)
from backends.toy import Toy
from frontends.tensor import TensorOp, TensorTerm
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from lower.lower import Lower
from lower.lower_diagonal import (
    estimate_diagonal_conv2d_ops,
    estimate_diagonal_matmul_ops,
)
from tests.diagonal_helpers import (
    assert_direct_kernel_toy_execution,
    direct_conv2d_kernel,
    direct_matmul_kernel,
    secret_arith_ops,
)
from tests.test_util import get_default_args
from util.layout_util import apply_layout


def test_conv2d_same_padding_diagonal_shifts():
    plan = diagonalize_conv2d((1, 8, 8), (1, 1, 3, 3), stride=1, padding="same")

    assert plan.op == TensorOp.CONV2D
    assert plan.source_shape == (1, 8, 8)
    assert plan.target_shape == (1, 8, 8)
    assert plan.reduction_axes == (0,)
    assert [shift.offset for shift in plan.shifts] == [
        (0, -1, -1),
        (0, -1, 0),
        (0, -1, 1),
        (0, 0, -1),
        (0, 0, 0),
        (0, 0, 1),
        (0, 1, -1),
        (0, 1, 0),
        (0, 1, 1),
    ]


def test_matmul_diagonal_plan_uses_contraction_lane_space():
    plan = diagonalize_matmul((4, 4), (4, 4))

    assert plan.op == TensorOp.MATMUL
    assert plan.output_shape == (4, 4)
    assert plan.lane_shape == (4, 4, 4)
    assert plan.reduction_axes == (1,)
    assert [shift.offset for shift in plan.shifts] == [
        (0, 0, 0),
        (0, 1, 0),
        (0, 2, 0),
        (0, 3, 0),
    ]


def test_diagonalize_term_dispatches_supported_ops():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    matmul_plan = diagonalize_term(a @ b, [(4, 4), (4, 4)])

    x = TensorTerm.Tensor("x", [1, 8, 8], True)
    w = TensorTerm.Tensor("w", [1, 1, 3, 3], False)
    conv_plan = diagonalize_term(
        TensorTerm.conv2d(x, w, 1, "same"), [(1, 8, 8), (1, 1, 3, 3)]
    )

    assert matmul_plan.op == TensorOp.MATMUL
    assert conv_plan.op == TensorOp.CONV2D


def test_layout_embedding_maps_row_major_slots_to_indices():
    layout = Layout(None, [], [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")], 16)
    embedding = layout_embedding(layout, (4, 4))

    assert embedding.lanes[0].index == (0, 0)
    assert embedding.lanes[1].index == (0, 1)
    assert embedding.lanes[4].index == (1, 0)
    assert embedding.by_index[(3, 3)][0].slot == 15


def test_boundary_slot_shift_scores_as_masked_rotation():
    layout = Layout(None, [], [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")], 16)
    plan = DiagonalActionPlan(
        op=TensorOp.CONV2D,
        output_shape=(4, 4),
        source_shape=(4, 4),
        target_shape=(4, 4),
        lane_shape=(4, 4),
        shifts=(LogicalShift((0, 1), "next-column"),),
    )

    score = score_layout_embedding(plan, layout)
    schedule = score.schedules[0]

    assert schedule.is_single_native_rotation
    assert schedule.groups[0].rotation == 15
    assert schedule.groups[0].lane_count == 12
    assert schedule.skipped_lanes == 4
    assert score.native_rotation_groups == 0
    assert score.masked_rotation_groups == 1
    assert score.mask_groups == 1
    assert score.cross_ct_groups == 0
    assert score.estimated_ops() == {"add": 0, "mul": 1, "rot": 1}


def test_interior_slot_shift_needs_only_one_native_rotation():
    layout = Layout(None, [], [Dim.parse("[0:4:1]"), Dim.parse("[1:8:1]")], 32)
    plan = DiagonalActionPlan(
        op=TensorOp.CONV2D,
        output_shape=(4, 4),
        source_shape=(4, 8),
        target_shape=(4, 4),
        lane_shape=(4, 8),
        shifts=(LogicalShift((0, 1), "interior-next-column"),),
    )

    score = score_layout_embedding(plan, layout)

    assert score.native_rotation_groups == 1
    assert score.masked_rotation_groups == 0
    assert score.mask_groups == 0
    assert score.estimated_ops() == {"add": 0, "mul": 0, "rot": 1}
    assert estimate_layout_embedding_ops(plan, layout, exact_work_limit=0) == {
        "add": 0,
        "mul": 0,
        "rot": 1,
    }


def test_ct_split_shift_scores_as_cross_ciphertext_movement():
    layout = Layout(None, [], [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]")], 4)
    plan = DiagonalActionPlan(
        op=TensorOp.MATMUL,
        output_shape=(4, 4),
        source_shape=(4, 4),
        target_shape=(4, 4),
        lane_shape=(4, 4),
        shifts=(LogicalShift((1, 0), "next-row"),),
    )

    score = score_layout_embedding(plan, layout)

    assert score.cross_ct_groups == 3
    assert score.native_rotation_groups == 0
    assert score.active_lanes == 12
    assert score.skipped_lanes == 4
    assert score.estimated_ops() == {"add": 3, "mul": 3, "rot": 0}


def test_conv2d_spatial_layout_scores_one_group_per_stencil_offset():
    layout = Layout(None, [], [Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")], 16)
    plan = diagonalize_conv2d((1, 4, 4), (1, 1, 3, 3), stride=1, padding="same")

    score = score_layout_embedding(plan, layout)

    assert len(score.schedules) == 9
    assert all(len(schedule.groups) == 1 for schedule in score.schedules)
    assert score.zero_rotation_groups == 1
    assert score.native_rotation_groups == 0
    assert score.masked_rotation_groups == 8
    assert score.mask_groups == 8
    assert score.cross_ct_groups == 0
    assert score.estimated_ops() == {"add": 0, "mul": 8, "rot": 8}
    assert estimate_layout_embedding_ops(plan, layout, exact_work_limit=0) == {
        "add": 0,
        "mul": 8,
        "rot": 8,
    }


def test_large_matmul_coarse_estimate_matches_simple_exact_schedule():
    layout = Layout(
        None,
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        64,
    )
    plan = diagonalize_matmul((4, 4), (4, 4))

    assert score_layout_embedding(plan, layout).estimated_ops() == {
        "add": 0,
        "mul": 3,
        "rot": 3,
    }
    assert estimate_layout_embedding_ops(plan, layout, exact_work_limit=0) == {
        "add": 0,
        "mul": 3,
        "rot": 3,
    }


def test_matmul_estimate_counts_one_rotation_per_target_ciphertext():
    layout = Layout(
        None,
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        16,
    )
    plan = diagonalize_matmul((4, 4), (4, 4))

    assert score_layout_embedding(plan, layout).estimated_ops() == {
        "add": 0,
        "mul": 12,
        "rot": 12,
    }
    assert estimate_layout_embedding_ops(plan, layout, exact_work_limit=0) == {
        "add": 0,
        "mul": 12,
        "rot": 12,
    }


def test_coarse_estimate_is_conservative_for_ciphertext_axis_shift():
    layout = Layout(
        None,
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        4,
    )
    plan = diagonalize_matmul((4, 4), (4, 4))
    exact_ops = score_layout_embedding(plan, layout).estimated_ops()
    coarse_ops = estimate_layout_embedding_ops(plan, layout, exact_work_limit=0)

    assert all(coarse_ops[op] >= exact_ops[op] for op in exact_ops)
    assert coarse_ops["add"] > exact_ops["add"]
    assert coarse_ops["mul"] > exact_ops["mul"]


def test_large_matmul_uses_coarse_estimator_without_exact_expansion(monkeypatch):
    plan = diagonalize_matmul((64, 64), (64, 64))
    layout = Layout(
        None,
        [],
        [Dim.parse("[0:64:1]"), Dim.parse("[1:64:1]"), Dim.parse("[2:64:1]")],
        4096,
    )

    def fail_exact_scoring(*_args, **_kwargs):
        raise AssertionError("large plans must not expand exact lane schedules")

    monkeypatch.setattr(diagonal, "score_layout_embedding", fail_exact_scoring)

    assert diagonal.estimate_layout_embedding_ops(plan, layout) == {
        "add": 0,
        "mul": 4032,
        "rot": 4032,
    }


def test_layout_assignment_uses_diagonal_embedding_cost_hook():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    term = a @ b
    args = SimpleNamespace(n=64, diagonal_first=True)
    assignment = LayoutAssignment(term, args)
    assignment.shape.padded_shapes = {a: (4, 4), b: (4, 4), term: (4, 4)}
    assignment.shape.shapes = dict(assignment.shape.padded_shapes)

    source_layout = Layout(
        a,
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        64,
        True,
    )
    plaintext_layout = Layout(
        b,
        [],
        [Dim.parse("[0:4:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        64,
        False,
    )
    output_layout = Layout(
        term, [], [Dim.parse("[0:4:1]"), Dim.parse("[2:4:1]")], 16, True
    )
    kernel = Kernel(
        KernelOp.MATMUL,
        [
            Kernel(KernelOp.CS, [0], source_layout),
            Kernel(KernelOp.CS, [1], plaintext_layout),
        ],
        output_layout,
    )

    assert assignment.diagonal_embedding_cost(term, kernel) > 0

    args.diagonal_first = False
    disabled_assignment = LayoutAssignment(term, args)
    disabled_assignment.shape.padded_shapes = assignment.shape.padded_shapes
    disabled_assignment.shape.shapes = assignment.shape.shapes

    assert disabled_assignment.diagonal_embedding_cost(term, kernel) == 0


def test_layout_assignment_diagonal_path_is_silent(capsys):
    args = get_default_args()
    args.n = 16
    args.diagonal_first = True
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)

    LayoutAssignment(a @ b, args).run()

    assert capsys.readouterr().out == ""


def test_diagonal_matmul_toy_execution():
    args = get_default_args()
    args.n = 16
    args.diagonal_first = True
    args.benchmark = "diagonal_matmul_research"
    inputs = {
        "a": np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]),
        "b": np.array([[1, 2, 0, 1], [0, 1, 1, 0], [2, 0, 1, 1], [1, 1, 0, 2]]),
    }
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    term = a @ b

    _assert_toy_execution(term, inputs, args)


def test_direct_diagonal_matmul_toy_execution_ct_pt_and_ct_ct():
    inputs = {
        "a": np.array([[1, 0, 1, 1], [0, 1, 1, 0], [1, 1, 0, 0], [0, 0, 1, 1]]),
        "b": np.array([[1, 2, 0, 1], [0, 1, 1, 0], [2, 0, 1, 1], [1, 1, 0, 2]]),
    }

    for b_secret in (False, True):
        a = TensorTerm.Tensor("a", [4, 4], True)
        b = TensorTerm.Tensor("b", [4, 4], b_secret)
        term = a @ b
        kernel = direct_matmul_kernel(a, b, term)

        assert_direct_kernel_toy_execution(kernel, term, inputs)


def test_direct_diagonal_conv2d_toy_execution_padding_and_stride():
    inputs = {
        "a": np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]),
        "b": np.array([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]]),
    }

    for padding, stride in (("same", 1), ("valid", 1), ("same", 2)):
        a = TensorTerm.Tensor("a", [1, 4, 4], True)
        b = TensorTerm.Tensor("b", [1, 1, 3, 3], False)
        term = TensorTerm.conv2d(a, b, stride, padding)
        kernel = direct_conv2d_kernel(a, b, term)

        assert_direct_kernel_toy_execution(kernel, term, inputs)


def test_direct_diagonal_estimate_matches_unoptimized_lowering_ops():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], True)
    matmul = a @ b
    matmul_kernel = direct_matmul_kernel(a, b, matmul)

    lower = Lower(matmul_kernel)
    lower.lower()
    assert secret_arith_ops(lower.env[matmul_kernel].cts.values()) == (
        estimate_diagonal_matmul_ops(matmul_kernel)
    )

    x = TensorTerm.Tensor("x", [1, 4, 4], True)
    w = TensorTerm.Tensor("w", [1, 1, 3, 3], False)
    conv = TensorTerm.conv2d(x, w, 1, "same")
    conv_kernel = direct_conv2d_kernel(x, w, conv)

    lower = Lower(conv_kernel)
    lower.lower()
    assert secret_arith_ops(lower.env[conv_kernel].cts.values()) == (
        estimate_diagonal_conv2d_ops(conv_kernel)
    )


def test_diagonal_candidate_guard_prevents_rolled_matmul_regression():
    for size in (4, 8, 16):
        a = TensorTerm.Tensor("a", [size, size], True)
        b = TensorTerm.Tensor("b", [size, size], True)
        term = a @ b

        args = get_default_args()
        args.n = size * size
        args.rolls = True
        args.diagonal_first = False
        baseline = LayoutAssignment(term, args).run()
        baseline_cost = KernelCost(baseline, args.net).total_cost()

        args.diagonal_first = True
        guarded = LayoutAssignment(term, args).run()
        guarded_cost = KernelCost(guarded, args.net).total_cost()

        assert guarded_cost <= baseline_cost


def test_diagonal_candidate_guard_skips_worse_matmul_clone():
    a = TensorTerm.Tensor("a", [4, 4], True)
    b = TensorTerm.Tensor("b", [4, 4], False)
    term = a @ b
    args = get_default_args()
    assignment = LayoutAssignment(term, args)
    kernel = direct_matmul_kernel(a, b, term, op=KernelOp.MATMUL)

    assert assignment.diagonal_kernel_clone(kernel) is None


def test_diagonal_candidate_guard_emits_cheaper_replicated_weight_conv_clone():
    x = TensorTerm.Tensor("x", [1, 4, 4], True)
    w = TensorTerm.Tensor("w", [1, 1, 1, 1], False)
    term = TensorTerm.conv2d(x, w, 1, "same")
    args = get_default_args()
    assignment = LayoutAssignment(term, args)

    x_layout = Layout(
        x,
        [],
        [Dim.parse("[0:1:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        16,
        True,
    )
    # A scalar public filter replicated across all slots makes the diagonal
    # circuit one ct-pt product over the whole output vector.  The standard
    # coarse conv estimator charges one multiply per output lane.
    w_layout = Layout(w, [], [Dim(None, 16)], 16, False)
    out_layout = Layout(
        term,
        [],
        [Dim.parse("[0:1:1]"), Dim.parse("[1:4:1]"), Dim.parse("[2:4:1]")],
        16,
        True,
    )
    base = Kernel(
        KernelOp.CONV2D,
        [Kernel(KernelOp.TENSOR, [], x_layout), Kernel(KernelOp.TENSOR, [], w_layout)],
        out_layout,
    )

    clone = assignment.diagonal_kernel_clone(base)

    assert clone is not None
    assert clone.op == KernelOp.DIAGONAL_CONV2D
    assert KernelCost(clone, args.net).op_cost() < KernelCost(base, args.net).op_cost()


def test_diagonal_conv2d_toy_execution():
    args = get_default_args()
    args.n = 16
    args.rolls = True
    args.diagonal_first = True
    args.benchmark = "diagonal_conv2d_research"
    inputs = {
        "a": np.array([[[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]]),
        "b": np.array([[[[1, 0, 1], [0, 1, 0], [1, 0, 1]]]]),
    }
    a = TensorTerm.Tensor("a", [1, 4, 4], True)
    b = TensorTerm.Tensor("b", [1, 1, 3, 3], False)
    term = TensorTerm.conv2d(a, b, 1, "same")

    _assert_toy_execution(term, inputs, args)


def _assert_toy_execution(term, inputs, args):
    expected = term.eval(inputs)
    kernel = LayoutAssignment(term, args).run()
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    expected_cts = apply_layout(expected, kernel.layout)
    assert expected_cts == results
