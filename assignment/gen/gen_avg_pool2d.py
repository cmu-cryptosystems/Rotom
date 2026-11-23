"""Layout generation for 2D average pooling (CHW tensors)."""

from copy import deepcopy as copy

from assignment.alignment import conv_spec
from assignment.gen.gen_conv2d import apply_replication
from assignment.gen.gen_window import (
    filter_dims,
    materialized_shape,
    output_dims,
    replication_dims,
)
from assignment.gen.gen_compaction import find_compaction
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from util.util import round_to_ceiling_power_of_2


def _avg_pool_out_hw(
    h_i: int, w_i: int, k: int, s: int, padding: str
) -> tuple[int, int]:
    if padding == "valid":
        h_o = (h_i - k) // s + 1
        w_o = (w_i - k) // s + 1
    elif padding == "same":
        if s == 1:
            h_o, w_o = h_i, w_i
        else:
            p = k // 2
            h_o = (h_i + 2 * p - k) // s + 1
            w_o = (w_i + 2 * p - k) // s + 1
    else:
        raise NotImplementedError(f"unknown padding: {padding!r}")
    return h_o, w_o


def gen_avg_pool2d(term, cs_kernels, shapes):
    """Generate AVG_POOL2D kernels with the output dims of a conv with a 1×k×k filter."""
    a_shape = shapes[0]
    k_sz = int(term.cs[1])
    stride = int(term.cs[2])
    padding = term.cs[3]
    h_i, w_i = int(a_shape[1]), int(a_shape[2])
    h_o, w_o = _avg_pool_out_hw(h_i, w_i, k_sz, stride, padding)
    h_o_p2 = round_to_ceiling_power_of_2(h_o)
    w_o_p2 = round_to_ceiling_power_of_2(w_o)

    # Pooling is laid out as a conv with a [C, 1, k, k] filter.
    b_shape = [int(a_shape[0]), 1, k_sz, k_sz]
    spec = conv_spec([stride] * 2)
    output_kernels = set()
    for a_kernel in cs_kernels:
        if a_kernel.layout.rolls:
            continue
        a_work = copy(a_kernel)
        replicated_dims, b_dims = replication_dims(spec, a_shape, b_shape)
        for dim in replicated_dims[::-1]:
            a_work = apply_replication(term.cs[0], a_work, dim)

        b_dims = filter_dims(spec, a_work.layout.get_dims(), b_dims)
        if b_dims is None:
            continue
        dims = output_dims(
            spec, a_work.layout.get_dims(), b_dims, {"h_out": h_o_p2, "w_out": w_o_p2}
        )
        if dims is None:
            continue
        output_layout = Layout(term, [], dims, a_work.layout.n, a_work.layout.secret)
        expected = [int(a_shape[0]), h_o_p2, w_o_p2]
        if materialized_shape(output_layout) != expected:
            continue
        # Child is the replicated input kernel DAG (like CONV2D), not a CS placeholder,
        # so ``update_kernels`` can resolve ``self.kernels[term.cs[0]][merged_layout]``.
        kavg = Kernel(KernelOp.AVG_POOL2D, [a_work], output_layout)
        output_kernels.add(kavg)
        if not kavg.layout.rolls:
            output_kernels.add(find_compaction(kavg))
    return output_kernels
