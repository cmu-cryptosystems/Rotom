from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import _get_apply_layout_plan


def lower_tile(env, kernel):
    """Lower TILE by periodic slot remapping from output logical indices to input."""
    input_cts = env[kernel.cs[0]]
    in_layout = kernel.cs[0].layout
    out_layout = kernel.layout
    in_shape = kernel.layout.term.cs[0].cs[1]
    reps = [int(x) for x in kernel.layout.term.cs[1]]
    rank = len(in_shape)
    out_shape = [int(in_shape[i]) * int(reps[i]) for i in range(rank)]
    if len(reps) != rank:
        raise ValueError("TILE reps rank mismatch")

    in_plan = _get_apply_layout_plan(
        in_layout, pt_tensor_ndim=rank, layout_len=max(len(in_layout), in_layout.n)
    )
    out_plan = _get_apply_layout_plan(
        out_layout, pt_tensor_ndim=rank, layout_len=max(len(out_layout), out_layout.n)
    )
    in_rows = in_plan["base_indices_by_cts"]
    out_rows = out_plan["base_indices_by_cts"]

    # lookup input logical index -> (ct_idx, slot_idx)
    in_lookup = {}
    for in_ct_idx in range(in_rows.shape[0]):
        for in_slot_idx in range(in_rows.shape[1]):
            row = in_rows[in_ct_idx, in_slot_idx]
            if (row < 0).any():
                continue
            in_lookup[tuple(int(v) for v in row.tolist())] = (in_ct_idx, in_slot_idx)

    cts = {}
    n = int(out_layout.n)
    for out_ct_idx in range(out_rows.shape[0]):
        groups = {}
        for out_slot_idx in range(out_rows.shape[1]):
            out_row = out_rows[out_ct_idx, out_slot_idx]
            if (out_row < 0).any():
                continue
            out_idx = [int(v) for v in out_row.tolist()]
            valid = True
            for d in range(rank):
                if out_idx[d] < 0 or out_idx[d] >= out_shape[d]:
                    valid = False
                    break
            if not valid:
                continue
            in_idx = tuple(out_idx[d] % int(in_shape[d]) for d in range(rank))
            if in_idx not in in_lookup:
                continue
            src_ct_idx, src_slot_idx = in_lookup[in_idx]
            rot_amt = (src_slot_idx - out_slot_idx) % n
            groups.setdefault((src_ct_idx, rot_amt), [0] * n)[out_slot_idx] = 1

        acc = None
        for (src_ct_idx, rot_amt), mask_vec in groups.items():
            src = input_cts[src_ct_idx]
            moved = (
                src
                if rot_amt == 0
                else HETerm(HEOp.ROT, [src, int(rot_amt)], src.secret)
            )
            term = moved * HETerm.mask([mask_vec])
            acc = term if acc is None else (acc + term)
        if acc is None:
            acc = HETerm(HEOp.ZERO_MASK, [], False)
        cts[out_ct_idx] = acc
    return LayoutCiphertexts(layout=out_layout, cts=cts)
