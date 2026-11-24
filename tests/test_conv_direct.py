"""The direct conv kernel: groups of masked rotations between two layouts."""

import numpy as np
import pytest

from assignment.alignment import conv_spec
from assignment.assignment import LayoutAssignment
from assignment.gen.gen_window import FILLS, direct_dims
from backends.toy import Toy
from frontends.tensor import TensorTerm
from ir.he import HEOp, HETerm
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from lower.lower import Lower
from tests.conftest import assert_results_equal, run_compiler_and_backend
from tests.test_util import get_default_args
from util.layout_util import apply_layout, parse_layout
from util.window import (
    GroupWeights,
    Window,
    output_ct,
    position_indices,
    summed_pieces,
    window_keys,
    window_ops,
    relayout_ops,
)


def _conv_window(c_in, size, c_out, k, stride, pad, out):
    return Window(
        conv_spec((stride, stride)),
        (c_in, size, size),
        (c_out, c_in, k, k),
        (c_out, out, out),
        (pad, pad),
    )


def _layouts(window, in_layout, n, fill):
    a = parse_layout(in_layout, n, True)
    z_dims, out_dims = direct_dims(window, a, fill)
    return a, Layout(None, [], z_dims, n, True), Layout(None, [], out_dims, n, True)


@pytest.mark.parametrize(
    "layout,shape",
    [
        ("[1:8:1][2:8:1][0:4:1]", (4, 8, 8)),
        ("[0:4:1];[G:4][1:8:1][2:8:1]", (4, 8, 8)),
        ("[0:2:2];[1:4:2][0:2:1][2:8:1][1:2:1]", (4, 8, 8)),
        ("[0:3:1];[G:4][1:8:1][2:8:1]", (3, 6, 7)),
    ],
)
def test_position_indices_match_apply_layout(layout, shape):
    """Packing a tensor of its own flat indices reads back the index table."""
    n = 64 if ";" in layout else 256
    parsed = parse_layout(layout, n, True)
    padded = [1 << (s - 1).bit_length() for s in shape]
    tensor = np.full(padded, -1.0)
    tensor[tuple(slice(0, s) for s in shape)] = np.arange(np.prod(shape)).reshape(shape)
    packed = np.concatenate(
        [np.asarray(ct, dtype=np.float64) for ct in apply_layout(tensor + 1, parsed)]
    )
    index = position_indices(parsed, shape)
    held = index[:, 0] >= 0
    flat = np.ravel_multi_index(tuple(index[held].T), shape)
    assert np.array_equal(packed[held], flat + 1)
    assert not packed[~held].any()


CASES = [
    # window, input layout, n
    (_conv_window(4, 8, 4, 3, 1, 1, 8), "[1:8:1][2:8:1][0:4:1]", 256),
    (_conv_window(4, 8, 8, 3, 2, 1, 4), "[1:8:1][2:8:1][0:4:1]", 256),
    (_conv_window(4, 8, 8, 3, 2, 1, 4), "[0:4:1];[G:4][1:8:1][2:8:1]", 256),
    (_conv_window(4, 8, 4, 3, 1, 0, 6), "[G:2][0:4:1][1:8:1][2:8:1]", 512),
    (_conv_window(4, 8, 4, 3, 2, 0, 3), "[G:2][0:4:1][1:8:1][2:8:1]", 512),
    (_conv_window(4, 8, 4, 5, 1, 0, 4), "[G:2][0:4:1][1:8:1][2:8:1]", 512),
    (_conv_window(3, 6, 5, 3, 1, 1, 6), "[G:4][1:8:1][2:8:1][0:4:1]", 1024),
    # split spatial dims
    (_conv_window(4, 8, 8, 3, 2, 1, 4), "[1:4:2][0:4:1][2:8:1][1:2:1]", 256),
    (_conv_window(4, 8, 4, 3, 1, 1, 8), "[1:2:4][2:8:1][0:4:1][1:4:1]", 256),
]


def _simulate(window, a, z, out, x, w):
    """Runs the kernel on packed numpy vectors: groups, then the sums."""
    n = z.n
    in_cts = [np.asarray(ct, dtype=np.float64) for ct in apply_layout(x, a)]
    z_cts = {}
    for key in window_keys(window, a, z):
        z_ct, in_ct, shift = (int(k) for k in key)
        slots, index = GroupWeights(window, a, z, key).entries()
        weights = np.zeros(n)
        weights[slots] = w[index]
        z_cts[z_ct] = z_cts.get(z_ct, 0) + np.roll(in_cts[in_ct], -shift) * weights
    cts = [np.zeros(n) for _ in range(out.num_ct())]
    for z_ct, ct in z_cts.items():
        cts[int(output_ct(window, z, z_ct))] += ct
    for i, ct in enumerate(cts):
        for _, extent, stride in summed_pieces(window, z)[1]:
            ct = sum(np.roll(ct, -j * stride) for j in range(extent))
        cts[i] = ct
    return cts


@pytest.mark.parametrize("case", range(len(CASES)))
@pytest.mark.parametrize("fill", FILLS)
def test_groups_compute_the_convolution(case, fill):
    window, in_layout, n = CASES[case]
    a, z, out = _layouts(window, in_layout, n, fill)
    rng = np.random.default_rng(case)
    x = rng.normal(size=window.in_shape)
    w = rng.normal(size=window.filter_shape)

    stride = window.spec.relations[0].stride
    pad = window.pads[0]
    k = window.filter_shape[2]
    padded = np.pad(x, ((0, 0), (pad, k), (pad, k)))
    expected = np.zeros(window.out_shape)
    for i in range(window.out_shape[1]):
        for j in range(window.out_shape[2]):
            patch = padded[:, stride * i : stride * i + k, stride * j : stride * j + k]
            expected[:, i, j] = np.tensordot(w, patch, axes=3)

    cts = _simulate(window, a, z, out, x, w)
    index = position_indices(out, window.out_shape)
    packed = np.concatenate(cts)
    held = index[:, 0] >= 0
    assert np.allclose(packed[held], expected[tuple(index[held].T)])


@pytest.mark.parametrize("case", range(len(CASES)))
def test_keys_match_pair_enumeration(case):
    """The sumset of per-label terms is the set of groups the pairs fall in."""
    window, in_layout, n = CASES[case]
    a, z, _ = _layouts(window, in_layout, n, "gaps")
    x_at = {
        tuple(row): p
        for p, row in enumerate(position_indices(a, window.in_shape))
        if row[0] >= 0
    }
    stride = window.spec.relations[0].stride
    pad = window.pads[0]
    keys = set()
    for p, (c_out, h_out, w_out, c_in) in enumerate(
        position_indices(z, window.z_shape)
    ):
        if c_out < 0:
            continue
        for kh in range(window.filter_shape[2]):
            for kw in range(window.filter_shape[3]):
                source = (c_in, stride * h_out + kh - pad, stride * w_out + kw - pad)
                if source in x_at:
                    q = x_at[source]
                    keys.add((p // n, q // n, (q % n - p % n) % n))
    assert {tuple(int(k) for k in key) for key in window_keys(window, a, z)} == keys


def test_stride_two_fills_the_freed_positions_with_channels():
    """Four output channels share the 16 shifts of a 3x3 stride-2 window."""
    window = _conv_window(16, 32, 32, 3, 2, 1, 16)
    in_layout = "[G:2][1:32:1][2:32:1][0:16:1]"
    a, z, out = _layouts(window, in_layout, 32768, "none")
    assert out.layout_str() == "[0:32:1];[G:2][1:16:1][G:2][2:16:1][G:32]"
    assert window_ops(window, a, z) == {"mul": 320, "rot": 136, "add": 384}

    a, z, out = _layouts(window, in_layout, 32768, "strides")
    assert out.layout_str() == "[0:8:4];[G:2][1:16:1][0:2:2][2:16:1][0:2:1][G:16]"
    assert window_ops(window, a, z) == {"mul": 136, "rot": 47, "add": 152}


def _he_ops(circuit):
    """Distinct ciphertext operations of a lowered circuit."""
    ops = {"mul": 0, "rot": 0, "add": 0}
    names = {HEOp.MUL: "mul", HEOp.ROT: "rot", HEOp.ADD: "add"}
    seen = set()
    stack = [ct for cts in circuit.values() for ct in cts.cts.values()]
    while stack:
        term = stack.pop()
        if not isinstance(term, HETerm) or term in seen:
            continue
        seen.add(term)
        if term.op in names and term.secret:
            ops[names[term.op]] += 1
        stack.extend(term.cs)
    return ops


def _compile(term, n):
    args = get_default_args()
    args.n = n
    args.backend = "toy"
    args.benchmark = "conv_direct"
    return LayoutAssignment(term, args).run()


@pytest.mark.parametrize("stride,padding", [(1, "same"), (2, "same"), (2, "valid")])
def test_cost_counts_the_lowered_circuit(stride, padding):
    a = TensorTerm.Tensor("a", [4, 8, 8], True)
    b = TensorTerm.Tensor("b", [8, 4, 3, 3], False)
    kernel = _compile(TensorTerm.conv2d(a, b, stride, padding), 256)
    conv = [k for k in kernel.post_order() if k.op == KernelOp.CONV_DIRECT]
    assert len(conv) == 1
    lowered = Lower(conv[0])
    lowered.lower()
    assert _he_ops({0: lowered.env[conv[0]]}) == KernelCost(conv[0], "lan").ops()


def _run(term, inputs, n):
    args = get_default_args()
    args.n = n
    args.backend = "toy"
    args.benchmark = "conv_direct"
    results, kernel = run_compiler_and_backend(term, inputs, args, "toy")
    assert_results_equal(apply_layout(term.eval(inputs), kernel.layout), results, "toy")
    return kernel


@pytest.mark.parametrize(
    "in_shape,filter_shape,stride,padding,layout",
    [
        ([2, 8, 8], [4, 2, 3, 3], 1, "valid", None),
        ([2, 8, 8], [4, 2, 3, 3], 2, "valid", None),
        ([4, 8, 8], [4, 4, 5, 5], 1, "valid", None),
        ([3, 6, 6], [5, 3, 3, 3], 1, "same", None),
        ([4, 8, 8], [8, 4, 1, 1], 2, "same", None),
        # the input channel on ciphertexts
        ([4, 8, 8], [8, 4, 3, 3], 2, "same", "[0:4:1];[1:8:1][2:8:1]"),
        # split spatial dims
        ([4, 8, 8], [8, 4, 3, 3], 2, "same", "[1:4:2][0:4:1][2:8:1][1:2:1]"),
        ([4, 8, 8], [4, 4, 3, 3], 1, "same", "[1:2:4][2:8:1][0:4:1][1:4:1]"),
    ],
)
def test_conv_matches_eval(in_shape, filter_shape, stride, padding, layout):
    rng = np.random.default_rng(0)
    a = TensorTerm.Tensor("a", in_shape, True, layout)
    b = TensorTerm.Tensor("b", filter_shape, False)
    inputs = {"a": rng.normal(size=in_shape), "b": rng.normal(size=filter_shape)}
    n = 64 if layout and ";" in layout else 256
    kernel = _run(TensorTerm.conv2d(a, b, stride, padding), inputs, n)
    assert any(k.op == KernelOp.CONV_DIRECT for k in kernel.post_order())


def test_two_convs_match_eval_and_count_the_first_conv():
    """A conv's cost includes the cost of computing its input."""
    rng = np.random.default_rng(0)
    a = TensorTerm.Tensor("a", [4, 8, 8], True)
    b = TensorTerm.Tensor("b", [8, 4, 3, 3], False)
    c = TensorTerm.Tensor("c", [8, 8, 3, 3], False)
    first = TensorTerm.conv2d(a, b, 2, "same").poly_call("relu", -8.0, 8.0)
    term = TensorTerm.conv2d(first, c, 1, "same")
    inputs = {
        "a": rng.normal(size=(4, 8, 8)),
        "b": rng.normal(size=(8, 4, 3, 3)),
        "c": rng.normal(size=(8, 8, 3, 3)),
    }
    _run(term, inputs, 256)

    args = get_default_args()
    args.n = 256
    args.backend = "toy"
    args.benchmark = "conv_direct"
    search = LayoutAssignment(term, args)
    search.run()
    assert (
        min(search.kernel_costs[term].values())
        > min(search.kernel_costs[first].values())
        > 0
    )


RELAYOUTS = [
    (
        "[0:2:8][1:8:1][0:2:4][2:8:1][G:2][0:4:1][G:2]",
        "[0:2:8][1:8:1][0:2:4][2:8:1][0:2:2][G:4][0:2:1]",
        4096,
    ),
    ("[0:4:1];[G:4][1:8:1][2:8:1]", "[1:8:1][0:4:1][2:8:1][G:4]", 256),
    ("[1:8:1][2:8:1][0:4:1]", "[0:2:2];[G:2][0:2:1][2:8:1][1:8:1]", 256),
    # copies in slots, on ciphertexts, and in the source
    ("[0:4:1][1:8:1][2:8:1]", "[R:2:1];[0:4:1][2:8:1][1:8:1]", 256),
    ("[G:4][0:4:1][1:8:1]", "[1:8:1][R:4:1][0:4:1]", 128),
    ("[R:4:1];[0:4:1][1:8:1]", "[1:8:1];[R:4:1][0:4:1]", 32),
    ("[R:2:1][0:4:1];[G:2][1:8:1][2:2:1]", "[1:8:1];[R:2:1][2:2:1][R:2:1][0:4:1]", 32),
]


def _conversion(source, target, n):
    """A CONVERSION kernel between two layouts of one placeholder tensor."""
    from util.window import layout_shape

    shape = layout_shape(parse_layout(target, n, True))
    term = TensorTerm.Tensor("x", list(shape), True)
    a = parse_layout(source, n, True)
    b = parse_layout(target, n, True)
    a = Layout(term, [], a.get_dims(), n, True)
    b = Layout(term, [], b.get_dims(), n, True)
    child = Kernel(KernelOp.TENSOR, [], a)
    kernel = Kernel(
        KernelOp.CONVERSION, [tuple(a.get_dims()), tuple(b.get_dims()), child], b
    )
    return kernel, {"x": np.random.default_rng(0).normal(size=shape)}


@pytest.mark.parametrize("source,target,n", RELAYOUTS)
def test_relayout_moves_every_element(source, target, n):
    kernel, inputs = _conversion(source, target, n)
    args = get_default_args()
    args.n = n
    args.backend = "toy"
    # the toy backend checks each kernel's ciphertexts against its layout
    Toy(Lower(kernel).run(), inputs, args).run()


@pytest.mark.parametrize("source,target,n", RELAYOUTS)
def test_relayout_cost_counts_the_lowered_circuit(source, target, n):
    kernel, _ = _conversion(source, target, n)
    circuit = Lower(kernel).run()
    counts = {"mul": 0, "rot": 0, "add": 0}
    seen = set()
    for root in circuit[kernel].values():
        for term in root.post_order():
            if term in seen:
                continue
            seen.add(term)
            name = {HEOp.MUL: "mul", HEOp.ROT: "rot", HEOp.ADD: "add"}.get(term.op)
            if name:
                counts[name] += 1
    assert counts == relayout_ops(kernel.cs[2].layout, kernel.layout)


def test_downsampling_block_with_a_conversion_matches_eval():
    """The two branches of a residual add arrive in different layouts."""
    rng = np.random.default_rng(0)
    inputs = {"x": rng.normal(size=(8, 16, 16))}

    def weights(name, shape):
        inputs[name] = rng.normal(size=shape) * 0.3
        return TensorTerm.Tensor(name, list(shape), False)

    def conv(x, name, c_out, c_in, k, stride, size):
        y = TensorTerm.conv2d(x, weights(name, (c_out, c_in, k, k)), stride, "same")
        return y * weights(name + "_s", (c_out, size, size)) + weights(
            name + "_b", (c_out, size, size)
        )

    x = TensorTerm.Tensor("x", [8, 16, 16], True)
    h = conv(x, "w0", 8, 8, 3, 1, 16).poly_call("relu", -8.0, 8.0)
    f = conv(h, "w1", 16, 8, 3, 2, 8).poly_call("relu", -8.0, 8.0)
    y = (conv(f, "w2", 16, 16, 3, 1, 8) + conv(h, "w3", 16, 8, 1, 2, 8)).poly_call(
        "relu", -8.0, 8.0
    )
    term = conv(y, "w4", 16, 16, 3, 1, 8) + y
    kernel = _run(term, inputs, 4096)
    assert any(k.op == KernelOp.CONVERSION for k in kernel.post_order())


def test_layouts_with_a_dims_digits_in_another_order_are_not_aligned():
    from assignment.gen.gen_align import _same_digits

    def pieces(layout):
        return [p for p in parse_layout(layout, 64, True).get_dims() if p.dim == 0]

    assert _same_digits(pieces("[0:2:2][0:2:1][1:16:1]"), pieces("[0:4:1][1:16:1]"))
    assert not _same_digits(pieces("[0:2:1][0:2:2][1:16:1]"), pieces("[0:4:1][1:16:1]"))


@pytest.mark.parametrize("channels,n", [(4, 512), (8, 1024)])
def test_residual_add_of_branches_with_reordered_channel_digits(channels, n):
    """A conv can place the channel's low digit ahead of its high digit; the
    shortcut it is added to holds the channel in order."""
    rng = np.random.default_rng(0)
    size = 8
    inputs = {"x": rng.normal(size=(2, size, size))}

    def weights(name, shape):
        inputs[name] = rng.normal(size=shape) * 0.3
        return TensorTerm.Tensor(name, list(shape), False)

    def conv(x, name, c_in):
        y = TensorTerm.conv2d(x, weights(name, (channels, c_in, 3, 3)), 1, "same")
        shape = (channels, size, size)
        return y * weights(name + "_s", shape) + weights(name + "_b", shape)

    x = TensorTerm.Tensor("x", [2, size, size], True)
    h = conv(x, "w0", 2).poly_call("relu", -8.0, 8.0)
    out = conv(h, "w1", channels).poly_call("relu", -8.0, 8.0)
    _run(conv(out, "w2", channels) + h, inputs, n)
