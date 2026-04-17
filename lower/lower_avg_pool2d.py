from ir.analysis.shape import Shape
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import _get_apply_layout_plan


def lower_avg_pool2d(env, kernel):
    """Lower AVG_POOL2D as sum of receptive-field cells, then scale by 1/(k*k)."""
    term = kernel.layout.term
    input_cts = env[kernel.cs[0]]
    in_layout = kernel.cs[0].layout
    out_layout = kernel.layout

    k_sz = int(term.cs[1])
    stride = int(term.cs[2])
    padding = term.cs[3]

    shape_an = Shape(term)
    shape_an.run()
    in_shape = [int(x) for x in shape_an.padded_shapes[term.cs[0]]]
    out_shape = [int(x) for x in shape_an.padded_shapes[term]]
    rank = len(in_shape)
    if rank != 3:
        raise ValueError("AVG_POOL2D lowering expects CHW rank-3 tensors")

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

    n = int(out_layout.n)
    den = k_sz * k_sz
    if den <= 0:
        raise ValueError("AVG_POOL2D kernel size must be positive")
    if den & (den - 1) != 0:
        raise NotImplementedError(
            "AVG_POOL2D lowering only supports kernel areas that are powers of two "
            f"(got k={k_sz}, area={den})"
        )
    scale_exp = den.bit_length() - 1
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
            for dh in range(k_sz):
                for dw in range(k_sz):
                    if padding == "valid":
                        ih = out_idx[1] * stride + dh
                        iw = out_idx[2] * stride + dw
                    elif padding == "same":
                        if stride == 1:
                            total_ph = max(
                                0, (out_shape[1] - 1) * stride + k_sz - in_shape[1]
                            )
                            total_pw = max(
                                0, (out_shape[2] - 1) * stride + k_sz - in_shape[2]
                            )
                            pad_top = total_ph // 2
                            pad_left = total_pw // 2
                        else:
                            pad_top = pad_left = k_sz // 2
                        ih = out_idx[1] * stride + dh - pad_top
                        iw = out_idx[2] * stride + dw - pad_left
                    else:
                        raise NotImplementedError(f"unknown padding: {padding!r}")
                    in_idx = (out_idx[0], ih, iw)
                    if any(
                        in_idx[d] < 0 or in_idx[d] >= in_shape[d] for d in range(rank)
                    ):
                        continue
                    if in_idx not in in_lookup:
                        continue
                    src_ct_idx, src_slot_idx = in_lookup[in_idx]
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
        else:
            acc = HETerm(HEOp.RESCALE, [acc, int(scale_exp)], acc.secret)
        cts[out_ct_idx] = acc
    return LayoutCiphertexts(layout=out_layout, cts=cts)
