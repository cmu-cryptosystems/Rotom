from ir.dim import DimType
from ir.kernel import Kernel, KernelOp
from ir.he import HETerm, HEOp
from lower.lower_util import find_sum_dim, rotate_and_sum, bsgs
from util.util import prod
from util.layout_util import (
    get_ct_idxs_by_dim,
    get_cts_by_dim,
    convert_layout_to_mask,
    get_segment,
)

from copy import deepcopy as copy


def lower_matmul(env, kernel):
    # BUG: alignment sometimes fails
    # KernelOp.ROT_ROLL: roll(0,2) [1:2:1][1:2:2];[2:1][0:4:1]
    # KernelOp.ROLL: roll(0,2) [0:4:1];[1:2:1][4:1]
    assert env[kernel.cs[0]].keys() == env[kernel.cs[1]].keys()

    # create cs pointers
    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in env[kernel.cs[0]].values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in env[kernel.cs[1]].values()]

    # calculate the multiplications between ct
    cts = {}
    for i, (a, b) in enumerate(zip(a_cs, b_cs)):
        mul_term = a * b
        cts[i] = mul_term

    # get layout
    layout = kernel.cs[0].layout
    ct_dims = copy(layout.ct_dims)
    slot_dims = copy(layout.slot_dims)

    # find summing dimension
    sum_dim = max(dim.dim for dim in kernel.cs[0].layout.get_dims() if dim.dim is not None)
    (ct_sum_dims, slot_sum_dims) = find_sum_dim(kernel.cs[0].layout, sum_dim)

    # sum together ciphertexts
    if ct_sum_dims:
        for ct_sum_dim in ct_sum_dims:
            ct_groups = get_cts_by_dim(cts, ct_dims, ct_sum_dim)

            # sum within group
            sum_cts = {}
            for i, group in enumerate(ct_groups):
                base = group[0]
                for j in range(1, len(group)):
                    base = base + group[j]
                sum_cts[i] = base

            # update layout and cts
            ct_dims.remove(ct_sum_dim)
            cts = sum_cts

    # sum together slot dimensions per ciphertext
    if slot_sum_dims:
        summed_cts = {}
        for index, term in cts.items():
            for slot_sum_dim in slot_sum_dims:
                segment = get_segment(slot_sum_dim, slot_dims)
                extent = segment[1]
                mul_offset = segment[2]
                if index not in summed_cts:
                    summed_cts[index] = rotate_and_sum(term, extent, mul_offset)
                else:
                    summed_cts[index] = rotate_and_sum(
                        summed_cts[index], extent, mul_offset
                    )
        cts = summed_cts
    # mask out gap dimensions
    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(kernel.layout)])
        masked_cts = {}
        for index, term in cts.items():
            mask_term = term * mask
            masked_cts[index] = mask_term
        cts = masked_cts
    return cts


def lower_bsgs_matmul(env, kernel):
    if kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL:
        replicated_kernel = kernel.cs[1].cs[1]
        rot_rolled_kernel = kernel.cs[1]
        rolled_kernel = kernel.cs[2]
    else:
        replicated_kernel = kernel.cs[2].cs[1]
        rot_rolled_kernel = kernel.cs[2]
        rolled_kernel = kernel.cs[1]

    assert env[replicated_kernel].keys() == env[rolled_kernel].keys()

    repeated_ct_dims = []
    for dim in replicated_kernel.layout.ct_dims:
        if dim.dim is None:
            repeated_ct_dims.append(dim)

    assert len(repeated_ct_dims) == 1
    ct_groups = get_ct_idxs_by_dim(
        rot_rolled_kernel.layout.ct_dims,
        rot_rolled_kernel.layout.rolls[0].dim_to_roll,
    )

    # group terms
    bsgs_terms = {}
    other_terms = {}
    for i, ct_group in enumerate(ct_groups):
        bsgs_terms[i] = env[replicated_kernel][ct_group[0]]
        other_cts = {}
        for j, ct in enumerate(ct_group):
            other_cts[j] = env[rolled_kernel][ct]
        other_terms[i] = other_cts

    dims = rot_rolled_kernel.layout.get_dims()

    # find dim size
    roll = kernel.cs[0]
    dim_size = dims[roll[0]].extent

    # find bsgs stride
    stride = prod(dim.extent for dim in rot_rolled_kernel.layout.slot_dims[1:])

    # run bsgs
    cts = {}
    for ct_idx, bsgs_term in bsgs_terms.items():
        cts[ct_idx] = bsgs(bsgs_term, other_terms[ct_idx], dim_size, stride, True)

    
    # find summation dimension:
    kernel_cs = [cs for cs in kernel.cs if isinstance(cs, Kernel)]
    sum_dim = max(dim.dim for dim in kernel_cs[0].layout.get_dims() if dim.dim is not None)

    if any(dim.dim == sum_dim for dim in replicated_kernel.layout.ct_dims):
        new_cts = {}
        for dim in replicated_kernel.layout.ct_dims:
            if dim.dim == sum_dim:
                segment = get_segment(dim, replicated_kernel.layout.ct_dims)
                split_cts = [[] for _ in range(len(cts) // segment[1])]
                for i in range(len(cts)):
                    split_cts[i // segment[1]].append(cts[i])

                for i, split in enumerate(split_cts):
                    base = split[0]
                    for j in range(1, len(split)):
                        base = base + split[j]
                    new_cts[i] = base 
        return new_cts
    return cts

