from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import _get_apply_layout_plan


def lower_concat(env, kernel):
    """Lower CONCAT by slot-remapping from selected input tensors."""
    term = kernel.layout.term
    input_terms = term.cs[0]
    axis = int(term.cs[1])
    rank = len(input_terms[0].cs[1])
    input_shapes = [t.cs[1] for t in input_terms]
    out_shape = list(input_shapes[0])
    out_shape[axis] = sum(int(s[axis]) for s in input_shapes)

    out_layout = kernel.layout
    out_plan = _get_apply_layout_plan(
        out_layout, pt_tensor_ndim=rank, layout_len=max(len(out_layout), out_layout.n)
    )
    out_rows = out_plan["base_indices_by_cts"]

    input_cts = [env[cs_kernel] for cs_kernel in kernel.cs]
    input_lookups = []
    for i, in_cts in enumerate(input_cts):
        in_layout = kernel.cs[i].layout
        in_plan = _get_apply_layout_plan(
            in_layout, pt_tensor_ndim=rank, layout_len=max(len(in_layout), in_layout.n)
        )
        in_rows = in_plan["base_indices_by_cts"]
        in_lookup = {}
        for in_ct_idx in range(in_rows.shape[0]):
            for in_slot_idx in range(in_rows.shape[1]):
                row = in_rows[in_ct_idx, in_slot_idx]
                if (row < 0).any():
                    continue
                in_lookup[tuple(int(v) for v in row.tolist())] = (
                    in_ct_idx,
                    in_slot_idx,
                )
        input_lookups.append(in_lookup)

    # Prefix offsets along concat axis.
    axis_offsets = [0]
    for shp in input_shapes[:-1]:
        axis_offsets.append(axis_offsets[-1] + int(shp[axis]))

    n = int(out_layout.n)
    cts = {}
    for out_ct_idx in range(out_rows.shape[0]):
        groups = {}
        for out_slot_idx in range(out_rows.shape[1]):
            out_row = out_rows[out_ct_idx, out_slot_idx]
            if (out_row < 0).any():
                continue
            out_idx = [int(v) for v in out_row.tolist()]
            if any(
                out_idx[d] < 0 or out_idx[d] >= int(out_shape[d]) for d in range(rank)
            ):
                continue

            src_input = None
            for i, offset in enumerate(axis_offsets):
                lo = offset
                hi = offset + int(input_shapes[i][axis])
                if lo <= out_idx[axis] < hi:
                    src_input = i
                    break
            if src_input is None:
                continue

            in_idx = list(out_idx)
            in_idx[axis] -= axis_offsets[src_input]
            in_idx = tuple(in_idx)

            in_lookup = input_lookups[src_input]
            if in_idx not in in_lookup:
                continue
            src_ct_idx, src_slot_idx = in_lookup[in_idx]
            rot_amt = (src_slot_idx - out_slot_idx) % n
            groups.setdefault((src_input, src_ct_idx, rot_amt), [0] * n)[
                out_slot_idx
            ] = 1

        acc = None
        for (src_input, src_ct_idx, rot_amt), mask_vec in groups.items():
            src = input_cts[src_input][src_ct_idx]
            moved = (
                src
                if rot_amt == 0
                else HETerm(HEOp.ROT, [src, int(rot_amt)], src.secret)
            )
            term_ct = moved * HETerm.mask([mask_vec])
            acc = term_ct if acc is None else (acc + term_ct)
        if acc is None:
            acc = HETerm(HEOp.ZERO_MASK, [], False)
        cts[out_ct_idx] = acc
    return LayoutCiphertexts(layout=out_layout, cts=cts)
