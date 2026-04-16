from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import _get_apply_layout_plan


def lower_index(env, kernel):
    """Lower INDEX for int/slice/tuple index specs by slot remapping.

    Builds output ciphertext slots from input ciphertext slots using
    rotation + mask accumulation. This supports multi-axis slicing where
    source slots may appear at different positions than destination slots.
    """
    input_cts = env[kernel.cs[0]]
    in_layout = kernel.cs[0].layout
    out_layout = kernel.layout
    in_term = in_layout.term
    out_term = out_layout.term
    in_rank = len(in_term.cs[1])

    # Normalize index spec to rank.
    index_spec = out_term.cs[1]
    if isinstance(index_spec, tuple):
        idx_items = list(index_spec)
    else:
        idx_items = [index_spec]
    if len(idx_items) < in_rank:
        idx_items += [slice(None, None, None)] * (in_rank - len(idx_items))
    idx_items = idx_items[:in_rank]

    # Build mapping output logical index -> input logical index according to index spec.
    in_dim_to_out_dim = {}
    out_dim = 0
    for d in range(in_rank):
        if isinstance(idx_items[d], int):
            in_dim_to_out_dim[d] = None
        else:
            in_dim_to_out_dim[d] = out_dim
            out_dim += 1

    def _map_out_to_in(out_idx_row):
        in_idx = [0] * in_rank
        for d in range(in_rank):
            spec = idx_items[d]
            if isinstance(spec, int):
                in_idx[d] = int(spec)
            else:
                od = in_dim_to_out_dim[d]
                start = 0 if spec.start is None else int(spec.start)
                step = 1 if spec.step is None else int(spec.step)
                in_idx[d] = start + int(out_idx_row[od]) * step
        return tuple(in_idx)

    # Get compact plan indices for slot-to-logical-index mapping.
    in_plan = _get_apply_layout_plan(
        in_layout, pt_tensor_ndim=in_rank, layout_len=max(len(in_layout), in_layout.n)
    )
    out_rank = out_dim
    out_plan = _get_apply_layout_plan(
        out_layout,
        pt_tensor_ndim=out_rank,
        layout_len=max(len(out_layout), out_layout.n),
    )
    in_rows = in_plan["base_indices_by_cts"]
    out_rows = out_plan["base_indices_by_cts"]

    # Build lookup: input logical index -> (ct_idx, slot_idx)
    in_lookup = {}
    for in_ct_idx in range(in_rows.shape[0]):
        for in_slot_idx in range(in_rows.shape[1]):
            row = in_rows[in_ct_idx, in_slot_idx]
            if (row < 0).any():
                continue
            in_lookup[tuple(int(v) for v in row.tolist())] = (in_ct_idx, in_slot_idx)

    # Build each output ciphertext by accumulating rotated/masked input ciphertexts.
    cts = {}
    n = int(out_layout.n)
    for out_ct_idx in range(out_rows.shape[0]):
        # Group destination slots by (source_ct, rot_amt) so each group uses one masked rotate.
        groups = {}
        for out_slot_idx in range(out_rows.shape[1]):
            out_row = out_rows[out_ct_idx, out_slot_idx]
            if (out_row < 0).any():
                continue
            in_logical_idx = _map_out_to_in(tuple(int(v) for v in out_row.tolist()))
            if in_logical_idx not in in_lookup:
                continue
            src_ct_idx, src_slot_idx = in_lookup[in_logical_idx]
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
