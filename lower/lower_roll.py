from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from lower.lower_util import bsgs, get_rots_per_ct
from util.layout_util import get_segments
from util.util import split_lists


def lower_roll(env, kernel):
    print("kernel:", kernel)
    print("roll:", kernel.cs[0])
    for cs in kernel.cs:
        print("cs:", cs)

    print("layout:", kernel.layout)
    print("kernel.layout.rolls:", kernel.layout.rolls)
    print("kernel.layout.ct_dims:", kernel.layout.ct_dims)
    print("kernel.layout.slot_dims:", kernel.layout.slot_dims)
    # offset removed
    print("kernel.layout.n:", kernel.layout.n)
    print("kernel.layout.secret:", kernel.layout.secret)
    print("kernel.layout.term:", kernel.layout.term)
    print("kernel.layout.term.cs:", kernel.layout.term.cs)

    print("kernel", kernel)
    for k in kernel.post_order():
        print("k:", k)

    n = kernel.layout.n
    roll = kernel.cs[0]
    ct_dims = kernel.layout.ct_dims

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

        print("base_indices_by_cts:", base_indices_by_cts)
        print("rolled_indices_by_cts:", rolled_indices_by_cts)
        cts = {}
        for ct_index in range(len(base_indices_by_cts)):
            # create base mapping
            base_map = {}
            for i, index in enumerate(base_indices_by_cts[ct_index]):
                if tuple(index) not in base_map:
                    base_map[tuple(index)] = []
                base_map[tuple(index)].append(i)

            # map rotations to new ct
            rots = {}
            for i, index in enumerate(rolled_indices_by_cts[ct_index]):
                index = tuple(index)
                rot_amt = base_map[index][0] - i
                base_map[index] = base_map[index][1:]
                if rot_amt not in rots:
                    rots[rot_amt] = []
                rots[rot_amt].append(i)

            # self.env ct term
            input_cts = env[kernel.cs[1]]
            base_term = input_cts[ct_index]
            if len(rots) == 1:
                # this roll can be done with a single rotation
                for rot_amt, _ in rots.items():
                    cts[ct_index] = base_term << rot_amt
            elif len(rots) == 2:
                # rolls can be optimized with subtraction
                rot_amts = list(rots.keys())
                indices = rots[rot_amts[0]]
                mask = [0] * n
                for index in indices:
                    mask[index] = 1

                # rotate mask
                mask = [mask[(i - rot_amts[0]) % n] for i in range(len(mask))]
                mask_term = HETerm(HEOp.MASK, [mask], False, "roll mask")

                # create masked_a_term
                masked_a_term = base_term * mask_term
                # create masked_b_term through subtraction
                masked_b_term = base_term - masked_a_term

                # rotate both
                rot_a_term = masked_a_term << rot_amts[0]
                rot_b_term = masked_b_term << rot_amts[1]
                cts[ct_index] = rot_a_term + rot_b_term
            else:
                # TODO: THERE SHOULD BE SOME BSGS OPTIMIZATION THAT CAN BE APPLIED HERE
                terms_to_sum = []
                for rot_amt, indices in rots.items():
                    mask = [0] * n
                    for index in indices:
                        mask[index] = 1
                    mask_term = HETerm(HEOp.MASK, [mask], False, "roll mask")
                    rot_base = base_term << rot_amt
                    mask_base = rot_base * mask_term
                    terms_to_sum.append(mask_base)
                sum_base = terms_to_sum[0]
                for sum_term in terms_to_sum[1:]:
                    sum_base = sum_base + sum_term
                cts[ct_index] = sum_base
        return LayoutCiphertexts(layout=kernel.layout, cts=cts)


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
    cts = {}
    for ct_index in range(len(base_indices_by_cts)):
        # create base mapping
        base_map = {}
        for i, index in enumerate(base_indices_by_cts[ct_index]):
            if tuple(index) not in base_map:
                base_map[tuple(index)] = []
            base_map[tuple(index)].append(i)

        # map rotations to new ct
        rots = {}
        for i, index in enumerate(rolled_indices_by_cts[ct_index]):
            index = tuple(index)
            rot_amt = base_map[index][0] - i
            base_map[index] = base_map[index][1:]
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
