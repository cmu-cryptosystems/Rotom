from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from lower.lower_util import bsgs, get_rots_per_ct
from util.layout_util import get_segments
from util.util import split_lists


def lower_roll(env, kernel):
    n = kernel.layout.n
    roll = kernel.cs[0]
    ct_dims = kernel.layout.ct_dims

    def _build_global_base_map(base_indices_by_cts):
        """
        Map a full-layout index tuple -> list of (ct_index, slot_index).
        We store a list since different slots can share the same projected index tuple.
        """
        global_map = {}
        for ct_i, indices in enumerate(base_indices_by_cts):
            for slot_i, index in enumerate(indices):
                key = tuple(index)
                if key not in global_map:
                    global_map[key] = []
                global_map[key].append((ct_i, slot_i))
        return global_map

    def _pop_from_global_map(global_map, key):
        if key not in global_map or len(global_map[key]) == 0:
            raise KeyError(f"missing base slot for index {key}")
        ct_slot = global_map[key][0]
        global_map[key] = global_map[key][1:]
        return ct_slot

    if roll.dim_to_roll in ct_dims and roll.dim_to_roll_by in ct_dims:
        input_cts = env[kernel.cs[1]]

        base_indices_by_cts, rolled_indices_by_cts = get_rots_per_ct(kernel)

        base_ct_map = {}
        for i, indices in enumerate(base_indices_by_cts):
            if str(indices) not in base_ct_map:
                base_ct_map[str(indices)] = i

        new_cts = {}
        for i, indices in enumerate(rolled_indices_by_cts):
            if str(indices) not in base_ct_map:
                raise KeyError("missing ct")
            new_cts[i] = input_cts[base_ct_map[str(indices)]]
        return LayoutCiphertexts(layout=kernel.layout, cts=new_cts)
    else:
        base_indices_by_cts, rolled_indices_by_cts = get_rots_per_ct(kernel)
        # Build a global mapping so rolled indices can source from different ciphertexts.
        # This avoids KeyError when a roll crosses ciphertext boundaries.
        global_base_map = _build_global_base_map(base_indices_by_cts)

        input_cts = env[kernel.cs[1]]
        # Normalize input_cts to sequential keys starting from 0, since
        # _build_global_base_map uses 0-based indices
        input_cts_list = list(input_cts.values())
        if len(input_cts_list) == 0:
            # Empty input - check if layout expects ciphertexts
            if len(rolled_indices_by_cts) > 0:
                raise ValueError(
                    f"lower_roll: input_cts is empty but {len(rolled_indices_by_cts)} output ciphertexts expected. "
                    f"This suggests a bug in the previous lowering step."
                )
            # Empty input and no output expected - return empty ciphertexts
            return LayoutCiphertexts(layout=kernel.layout, cts={})
        
        out_cts = {}
        for out_ct_index in range(len(rolled_indices_by_cts)):
            # Group contributions by (source_ct, rot_amt) -> output slot indices.
            contribs = {}  # (src_ct, rot_amt) -> [out_slot_i, ...]
            for out_slot_i, index in enumerate(rolled_indices_by_cts[out_ct_index]):
                key = tuple(index)
                src_ct_index, src_slot_i = _pop_from_global_map(global_base_map, key)
                rot_amt = src_slot_i - out_slot_i
                contrib_key = (src_ct_index, rot_amt)
                if contrib_key not in contribs:
                    contribs[contrib_key] = []
                contribs[contrib_key].append(out_slot_i)

            # Fast path: one source ciphertext and one rotation => pure rotation.
            if len(contribs) == 1:
                (src_ct_index, rot_amt), _indices = next(iter(contribs.items()))
                if src_ct_index >= len(input_cts_list):
                    raise IndexError(
                        f"src_ct_index {src_ct_index} out of range for input_cts (len={len(input_cts_list)})"
                    )
                out_cts[out_ct_index] = input_cts_list[src_ct_index] << rot_amt
                continue

            # General (correct) path: masked rotated contributions summed together.
            terms_to_sum = []
            for (src_ct_index, rot_amt), out_indices in contribs.items():
                if src_ct_index >= len(input_cts_list):
                    raise IndexError(
                        f"src_ct_index {src_ct_index} out of range for input_cts (len={len(input_cts_list)})"
                    )
                mask = [0] * n
                for idx in out_indices:
                    mask[idx] = 1
                mask_term = HETerm(HEOp.MASK, [mask], False, "roll mask")
                rot_term = input_cts_list[src_ct_index] << rot_amt
                terms_to_sum.append(rot_term * mask_term)

            sum_term = terms_to_sum[0]
            for t in terms_to_sum[1:]:
                sum_term = sum_term + t
            out_cts[out_ct_index] = sum_term

        return LayoutCiphertexts(layout=kernel.layout, cts=out_cts)


def lower_rot_roll(env, kernel):
    roll = kernel.cs[0]
    rot_stride = 1
    for dim in kernel.layout.slot_dims[::-1]:
        if dim == roll.dim_to_roll_by:
            break
        rot_stride *= dim.extent
    input_cts = env[kernel.cs[1]]
    split_cts = split_lists(list(input_cts.values()), roll.dim_to_roll.extent)
    cts = {}
    ct_index = 0
    for i, split_ct in enumerate(split_cts):
        rot_amt = rot_stride * i
        for ct in split_ct:
            cts[ct_index] = HETerm(HEOp.ROT, [ct, rot_amt], ct.secret)
            ct_index += 1
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)


def lower_bsgs_rot_roll(env, kernel):
    return lower_rot_roll(env, kernel)


def lower_split_roll(env, kernel):
    n = kernel.layout.n
    cts = {}

    # env ct term
    cache = {}
    for ct_index in range(kernel.layout.num_ct()):
        roll = kernel.cs[0]
        roll_index = roll.roll_index(kernel.layout.get_dims())

        rots = {}
        segments = get_segments(kernel.layout.get_dims())
        segment = segments[roll_index[0]]
        rot_amt = segment[2] * (ct_index // segment[2])
        rot_amt_2 = -((segment[1] * segment[2] - rot_amt) % (segment[1] * segment[2]))

        if (rot_amt, rot_amt_2) not in cache:
            rots[rot_amt] = [
                1 if i % (segment[1] * segment[2]) < abs(rot_amt_2) else 0
                for i in range(segment[0])
            ]
            rots[rot_amt_2] = [
                1 if i % (segment[1] * segment[2]) >= abs(rot_amt_2) else 0
                for i in range(segment[0])
            ]
            cache[(rot_amt, rot_amt_2)] = rots
        else:
            rots = cache[(rot_amt, rot_amt_2)]

        input_cts = env[kernel.cs[1]]
        base_term = input_cts[ct_index]
        if len(rots) == 1:
            # this roll can be done with a single rotation
            for rot_amt, _ in rots.items():
                if isinstance(base_term, list):
                    cts[ct_index] = [ct << rot_amt for ct in base_term]
                else:
                    cts[ct_index] = [base_term << rot_amt]
        elif len(rots) == 2:
            # keep only the left rotations
            rot_amts = list(rots.keys())
            mask = rots[rot_amts[0]]

            # rotate mask
            mask = [mask[(i - rot_amts[0]) % n] for i in range(len(mask))]
            mask2 = [i ^ 1 for i in mask]
            mask_term = HETerm(HEOp.MASK, [mask], False, "roll mask")
            mask_term_2 = HETerm(HEOp.MASK, [mask2], False, "roll mask 2")

            # create masked_a_term
            if isinstance(base_term, list):
                # should be [left right] -> [ll lr rl rr]
                res = []
                for ct in base_term:
                    masked_left_term = ct * mask_term
                    # create masked_b_term through subtraction
                    masked_right_term = ct * mask_term_2

                    # rotate both
                    rot_left_term = masked_left_term << rot_amts[0]
                    rot_right_term = masked_right_term << rot_amts[1]
                    res += [rot_left_term, rot_right_term]
                cts[ct_index] = res
            else:
                masked_left_term = base_term * mask_term
                # create masked_b_term through subtraction
                masked_right_term = base_term * mask_term_2

                # rotate both
                rot_left_term = masked_left_term << rot_amts[0]
                rot_right_term = masked_right_term << rot_amts[1]
                cts[ct_index] = [rot_left_term, rot_right_term]
    # Note: lower_split_roll returns a dict with list values, which is a special case
    # We'll keep it as-is for now, but it may need special handling
    return cts


def lower_bsgs_roll(env, kernel):
    n = kernel.layout.n
    # find bsgs interval
    roll = kernel.cs[0]
    dim_size = roll.dim_to_roll.extent

    # find rotation amounts
    base_indices_by_cts, rolled_indices_by_cts = get_rots_per_ct(kernel)
    # If this roll crosses ciphertext boundaries, fall back to the general (masked) lowering.
    # The BSGS logic assumes each output ciphertext is sourced from exactly one input ciphertext.
    global_base_map = {}
    for ct_i, indices in enumerate(base_indices_by_cts):
        for slot_i, index in enumerate(indices):
            key = tuple(index)
            if key not in global_base_map:
                global_base_map[key] = []
            global_base_map[key].append((ct_i, slot_i))

    def _pop_global(key):
        if key not in global_base_map or len(global_base_map[key]) == 0:
            raise KeyError(f"missing base slot for index {key}")
        v = global_base_map[key][0]
        global_base_map[key] = global_base_map[key][1:]
        return v

    crosses_ct = False
    for out_ct_index in range(len(rolled_indices_by_cts)):
        for index in rolled_indices_by_cts[out_ct_index]:
            src_ct_i, _src_slot_i = _pop_global(tuple(index))
            if src_ct_i != out_ct_index:
                crosses_ct = True
                break
        if crosses_ct:
            break

    if crosses_ct:
        return lower_roll(env, kernel)

    # Rebuild per-ct mapping now that we've consumed the global map during detection.
    base_indices_by_cts, rolled_indices_by_cts = get_rots_per_ct(kernel)
    cts = {}
    for ct_index in range(len(base_indices_by_cts)):
        base_map = {}
        for i, index in enumerate(base_indices_by_cts[ct_index]):
            key = tuple(index)
            if key not in base_map:
                base_map[key] = []
            base_map[key].append(i)

        rots = {}
        for i, index in enumerate(rolled_indices_by_cts[ct_index]):
            key = tuple(index)
            if key not in base_map or len(base_map[key]) == 0:
                # Safety fallback: shouldn't happen after crosses_ct detection,
                # but falling back keeps us correct.
                return lower_roll(env, kernel)
            rot_amt = base_map[key][0] - i
            base_map[key] = base_map[key][1:]
            if rot_amt not in rots:
                rots[rot_amt] = []
            rots[rot_amt].append(i)

        # create left and right masks
        masks = {}
        for rot_amt, indices in rots.items():
            mask = [0] * n
            for index in indices:
                mask[index] = 1
            mask_term = HETerm(HEOp.MASK, [mask], False, "roll mask")
            masks[rot_amt] = mask_term

        left_masks = []
        right_masks = []
        for rot_amt, mask_term in masks.items():
            if rot_amt == 0:
                left_masks.append((rot_amt, mask_term))
                right_masks.append(
                    (
                        rot_amt,
                        HETerm(HEOp.MASK, [[0] * n], False, "roll mask"),
                    )
                )
            elif rot_amt > 0:
                left_masks.append((rot_amt, mask_term))
            else:
                right_masks.append((rot_amt, mask_term))

        left_masks = [y[1] for y in sorted(left_masks, key=lambda x: x[0])]
        right_masks = [
            y[1] for y in sorted(right_masks, key=lambda x: x[0], reverse=True)
        ]

        # find bsgs stride
        stride = 1
        for i in range(1, len(left_masks[0].cs[0])):
            if left_masks[0].cs[0][i]:
                break
            stride += 1

        input_cts = env[kernel.cs[1]]
        base_term = input_cts[ct_index]
        left = bsgs(base_term, left_masks, dim_size, stride, True)
        right = bsgs(base_term, right_masks, dim_size, stride, False)
        cts[ct_index] = left + right
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
