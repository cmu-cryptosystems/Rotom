from ir.analysis.shape import Shape
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import _get_apply_layout_plan


def lower_cumsum(env, kernel):
    """Lower CUMSUM by summing prefix (or suffix) contributions per output slot."""
    term = kernel.layout.term
    input_cts = env[kernel.cs[0]]
    in_layout = kernel.cs[0].layout
    out_layout = kernel.layout

    axis = int(term.cs[1])
    exclusive = bool(term.cs[2]) if len(term.cs) > 2 else False
    reverse = bool(term.cs[3]) if len(term.cs) > 3 else False

    shape_an = Shape(term)
    shape_an.run()
    in_shape = [int(x) for x in shape_an.padded_shapes[term.cs[0]]]
    out_shape = [int(x) for x in shape_an.padded_shapes[term]]
    rank = len(in_shape)
    if len(out_shape) != rank:
        raise ValueError("CUMSUM expects output rank to match input")

    in_plan = _get_apply_layout_plan(
        in_layout, pt_tensor_ndim=rank, layout_len=max(len(in_layout), in_layout.n)
    )
    out_plan = _get_apply_layout_plan(
        out_layout, pt_tensor_ndim=rank, layout_len=max(len(out_layout), out_layout.n)
    )
    in_rows = in_plan["base_indices_by_cts"]
    out_rows = out_plan["base_indices_by_cts"]

    in_lookup = {}
    for in_ct_idx in range(in_rows.shape[0]):
        for in_slot_idx in range(in_rows.shape[1]):
            row = in_rows[in_ct_idx, in_slot_idx]
            if (row < 0).any():
                continue
            in_lookup[tuple(int(v) for v in row.tolist())] = (in_ct_idx, in_slot_idx)

    def axis_values_for_out(out_ax: int):
        if not reverse:
            if exclusive:
                return range(0, out_ax)
            return range(0, out_ax + 1)
        if exclusive:
            return range(out_ax + 1, in_shape[axis])
        return range(out_ax, in_shape[axis])

    n = int(out_layout.n)
    cts = {}
    for out_ct_idx in range(out_rows.shape[0]):
        acc = None
        for out_slot_idx in range(out_rows.shape[1]):
            out_row = out_rows[out_ct_idx, out_slot_idx]
            if (out_row < 0).any():
                continue
            out_idx = [int(v) for v in out_row.tolist()]
            if any(out_idx[d] < 0 or out_idx[d] >= out_shape[d] for d in range(rank)):
                continue
            slot_acc = None
            for av in axis_values_for_out(out_idx[axis]):
                in_idx = list(out_idx)
                in_idx[axis] = av
                if any(in_idx[d] < 0 or in_idx[d] >= in_shape[d] for d in range(rank)):
                    continue
                t = tuple(in_idx)
                if t not in in_lookup:
                    continue
                src_ct_idx, src_slot_idx = in_lookup[t]
                rot_amt = (src_slot_idx - out_slot_idx) % n
                src = input_cts[src_ct_idx]
                moved = (
                    src
                    if rot_amt == 0
                    else HETerm(HEOp.ROT, [src, int(rot_amt)], src.secret)
                )
                mask_vec = [0] * n
                mask_vec[out_slot_idx] = 1
                term_h = moved * HETerm.mask([mask_vec])
                slot_acc = term_h if slot_acc is None else (slot_acc + term_h)
            if slot_acc is not None:
                acc = slot_acc if acc is None else (acc + slot_acc)
        if acc is None:
            acc = HETerm(HEOp.ZERO_MASK, [], False)
        cts[out_ct_idx] = acc
    return LayoutCiphertexts(layout=out_layout, cts=cts)
