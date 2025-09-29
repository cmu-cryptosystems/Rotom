import math

from ir.he import HEOp, HETerm
from util.layout_util import add_vec, get_dim_indices, mul


def rotate_and_sum(base_term, extent, mul_offset, replicate=False):
    num_rots = int(math.ceil(math.log2(extent)))
    for i in range(num_rots):
        rot_amt = int(math.pow(2, i) * mul_offset)
        if replicate:
            rot_amt = -rot_amt
        rotated_ct = HETerm(
            HEOp.ROT,
            [base_term, rot_amt],
            base_term.secret,
            f"rot-and-sum ({extent},{mul_offset})",
        )
        summed_ct = rotated_ct + base_term
        base_term = summed_ct
    return base_term


def helper_find_sum_dim(dims, sum_dim):
    sum_dims = []
    for dim in dims:
        if dim.dim == sum_dim:
            sum_dims.append(dim)
    return sum_dims


def find_sum_dim(layout, sum_dim):
    ct_dims = helper_find_sum_dim(layout.ct_dims, sum_dim)
    slot_dims = helper_find_sum_dim(layout.slot_dims, sum_dim)
    return (ct_dims, slot_dims)


def find_bsgs_interval(n):
    k = int(math.log2(n))
    giant_step = 2 ** (k // 2)
    baby_step = 2 ** (k - (k // 2))
    return baby_step, giant_step


def bsgs(ct, pts, dim_size, stride, left):
    # 3. find bsgs interval
    baby_step, giant_step = find_bsgs_interval(dim_size)

    # 4. apply bsgs
    # 4.1 apply baby_steps to `a`
    ct_baby_steps = {}
    for bs in range(baby_step):
        rot = bs * stride if left else -bs * stride
        ct_baby_steps[bs] = HETerm(
            HEOp.ROT, [ct, rot], ct.secret, f"rot baby steps << {rot}"
        )

    # replicate terms
    for gs in range(giant_step):
        for bs in range(baby_step):
            ct_baby_steps[bs + baby_step * gs] = ct_baby_steps[bs]

    # 4.2 apply reverse giant_steps to `b`
    pt_giant_steps = {}
    for gs in range(giant_step):
        for bs in range(baby_step):
            rot = -baby_step * gs * stride if left else baby_step * gs * stride
            pt_giant_steps[bs + baby_step * gs] = HETerm(
                HEOp.ROT,
                [pts[bs + baby_step * gs], rot],
                pts[bs + baby_step * gs].secret,
                f"rot rev giant steps << {rot}",
            )

    # 4.3 multiplication
    mul_terms = {}
    for gs in range(giant_step):
        for bs in range(baby_step):
            mul_terms[bs + baby_step * gs] = HETerm(
                HEOp.MUL,
                [
                    ct_baby_steps[bs + baby_step * gs],
                    pt_giant_steps[bs + baby_step * gs],
                ],
                ct_baby_steps[bs + baby_step * gs].secret
                or pt_giant_steps[bs + baby_step * gs].secret,
                f"bsgs mul",
            )

    # 4.4 summation
    sum_terms = {}
    for gs in range(giant_step):
        sum_terms[gs] = mul_terms[baby_step * gs]
        for bs in range(1, baby_step):
            sum_terms[gs] = HETerm(
                HEOp.ADD,
                [sum_terms[gs], mul_terms[bs + baby_step * gs]],
                sum_terms[gs].secret or mul_terms[bs + baby_step * gs].secret,
                f"bsgs sum",
            )

    # 4.5 apply giant steps
    giant_cts = {}
    for gs in range(giant_step):
        rot = baby_step * gs * stride if left else -baby_step * gs * stride
        giant_cts[gs] = HETerm(
            HEOp.ROT,
            [sum_terms[gs], rot],
            sum_terms[gs].secret,
            f"rot giant steps << {rot}",
        )

    # 4.6 sum giant_steps
    base_term = giant_cts[0]
    for gs in range(1, giant_step):
        base_term = HETerm(
            HEOp.ADD,
            [base_term, giant_cts[gs]],
            base_term.secret or giant_cts[gs].secret,
            f"sum giant steps",
        )
    return base_term


def get_rots_per_ct(kernel):
    n = kernel.layout.n
    layout_len = len(kernel.layout)

    # get base_term indices
    base_dims = kernel.cs[1].layout.get_dims()
    base_dim_indices = get_dim_indices(base_dims)

    # apply any base term rolls
    base_rolled = {}
    for roll in kernel.cs[1].layout.rolls:
        roll_index = roll.roll_index(base_dims)
        base_rolled[roll_index[0]] = [
            (base_dim_indices[roll_index[0]][i] + base_dim_indices[roll_index[1]][i])
            % roll.dim_to_roll.extent
            for i in range(layout_len)
        ]
    for k, v in base_rolled.items():
        base_dim_indices[k] = v

    # get base_term indices
    rolled_dims = kernel.layout.get_dims()
    rolled_dim_indices = get_dim_indices(rolled_dims)

    # apply any base term rolls
    rolled = {}
    for roll in kernel.layout.rolls:
        roll_index = roll.roll_index(rolled_dims)
        rolled[roll_index[0]] = [
            (
                rolled_dim_indices[roll_index[0]][i]
                + rolled_dim_indices[roll_index[1]][i]
            )
            % roll.dim_to_roll.extent
            for i in range(layout_len)
        ]
    for k, v in rolled.items():
        rolled_dim_indices[k] = v

    # split indices by dimensions:
    base_indices_map = {}
    for i, dim in enumerate(base_dims):
        if dim.dim is not None:
            if dim.dim not in base_indices_map:
                base_indices_map[dim.dim] = mul(base_dim_indices[i], dim.stride)
            else:
                base_indices_map[dim.dim] = add_vec(
                    base_indices_map[dim.dim], mul(base_dim_indices[i], dim.stride)
                )
    rolled_indices_map = {}
    for i, dim in enumerate(rolled_dims):
        if dim.dim is not None:
            if dim.dim not in rolled_indices_map:
                rolled_indices_map[dim.dim] = mul(rolled_dim_indices[i], dim.stride)
            else:
                rolled_indices_map[dim.dim] = add_vec(
                    rolled_indices_map[dim.dim],
                    mul(rolled_dim_indices[i], dim.stride),
                )

    # map to pertinent dimensions
    base_indices = [[0] * (max(base_indices_map) + 1) for _ in range(layout_len)]
    for dim, indices in base_indices_map.items():
        if dim in base_indices_map:
            for i, index in enumerate(indices):
                base_indices[i][dim] = index

    rolled_indices = [[0] * (max(rolled_indices_map) + 1) for _ in range(layout_len)]
    for dim, indices in rolled_indices_map.items():
        if dim in rolled_indices_map:
            for i, index in enumerate(indices):
                rolled_indices[i][dim] = index

    # split by cts
    base_indices_by_cts = [
        base_indices[i * n : (i + 1) * n] for i in range((layout_len // n))
    ]
    rolled_indices_by_cts = [
        rolled_indices[i * n : (i + 1) * n] for i in range((layout_len // n))
    ]

    return base_indices_by_cts, rolled_indices_by_cts


def ops(env, term):
    ops = {}
    seen = set()
    for i, ct in env[term].items():
        for t in ct.post_order():
            if t in seen:
                continue
            if not t.secret:
                continue
            seen.add(t)
            if t.op not in ops:
                ops[t.op] = 0
            ops[t.op] += 1
    return ops


def total_ops(terms):
    ops = {}
    seen = set()
    for ct in terms:
        for t in ct.post_order():
            if t in seen:
                continue
            if not t.secret:
                continue
            seen.add(t)
            op_str = str(t.op)
            if t.op == HEOp.MUL and any(not cs.secret for cs in t.cs):
                op_str = f"{op_str}_plain"
            if op_str not in ops:
                ops[op_str] = 0
            ops[op_str] += 1
    return ops
