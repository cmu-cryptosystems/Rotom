"""Small fixtures for direct diagonal lowering tests."""

from __future__ import annotations

from ir.dim import Dim
from ir.he import HEOp
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from lower.lower import Lower
from tests.test_util import get_default_args
from util.layout_util import apply_layout
from util.util import prod


def direct_matmul_kernel(a, b, term, op=KernelOp.DIAGONAL_MATMUL):
    """Build a compact square 2D direct-matmul kernel for tests."""

    a_shape = tuple(a.cs[1])
    b_shape = tuple(b.cs[1])
    assert len(a_shape) == 2 and len(b_shape) == 2
    assert a_shape[1] == b_shape[0]
    assert a_shape[0] == b_shape[1]

    n = prod(a_shape)
    a_dims = [Dim.parse(f"[0:{a_shape[0]}:1]"), Dim.parse(f"[1:{a_shape[1]}:1]")]
    b_dims = [Dim.parse(f"[0:{b_shape[0]}:1]"), Dim.parse(f"[1:{b_shape[1]}:1]")]
    out_dims = [Dim.parse(f"[0:{a_shape[0]}:1]"), Dim.parse(f"[1:{b_shape[1]}:1]")]

    return Kernel(
        op,
        [
            Kernel(KernelOp.TENSOR, [], Layout(a, [], a_dims, n, True)),
            Kernel(KernelOp.TENSOR, [], Layout(b, [], b_dims, n, b.cs[2])),
        ],
        Layout(term, [], out_dims, n, True),
    )


def direct_conv2d_kernel(x, w, term, op=KernelOp.DIAGONAL_CONV2D):
    """Build a 4x4-image direct-conv kernel with padded filter lanes."""

    x_shape = tuple(x.cs[1])
    w_shape = tuple(w.cs[1])
    assert len(x_shape) == 3 and len(w_shape) == 4

    n = prod(x_shape)
    x_dims = [
        Dim.parse(f"[0:{x_shape[0]}:1]"),
        Dim.parse(f"[1:{x_shape[1]}:1]"),
        Dim.parse(f"[2:{x_shape[2]}:1]"),
    ]
    w_dims = [
        Dim.parse(f"[0:{w_shape[0]}:1]"),
        Dim.parse(f"[1:{w_shape[1]}:1]"),
        Dim.parse(f"[2:{_next_power_of_two(w_shape[2])}:1]"),
        Dim.parse(f"[3:{_next_power_of_two(w_shape[3])}:1]"),
    ]
    out_dims = [
        Dim.parse(f"[0:{w_shape[0]}:1]"),
        Dim.parse(f"[1:{x_shape[1]}:1]"),
        Dim.parse(f"[2:{x_shape[2]}:1]"),
    ]

    return Kernel(
        op,
        [
            Kernel(KernelOp.TENSOR, [], Layout(x, [], x_dims, n, True)),
            Kernel(KernelOp.TENSOR, [], Layout(w, [], w_dims, n, w.cs[2])),
        ],
        Layout(term, [], out_dims, n, True),
    )


def assert_direct_kernel_toy_execution(kernel, term, inputs):
    from backends.toy import Toy

    args = get_default_args()
    args.n = kernel.layout.n
    circuit_ir = Lower(kernel).run()
    results = Toy(circuit_ir, inputs, args).run()

    expected_cts = apply_layout(term.eval(inputs), kernel.layout)
    assert expected_cts == results


def secret_arith_ops(terms):
    counts = {"add": 0, "mul": 0, "rot": 0}
    seen = set()
    for term in terms:
        for node in term.post_order():
            if node in seen or not node.secret:
                continue
            seen.add(node)
            match node.op:
                case HEOp.ADD:
                    counts["add"] += 1
                case HEOp.MUL:
                    counts["mul"] += 1
                case HEOp.ROT:
                    counts["rot"] += 1
    return counts


def _next_power_of_two(value):
    return 1 << (int(value) - 1).bit_length()
