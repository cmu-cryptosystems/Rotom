from copy import deepcopy as copy

from ir.dim import DimType
from ir.he import HEOp, HETerm
from ir.kernel import Kernel, KernelOp
from ir.layout import Layout
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import bsgs, find_sum_dim, rotate_and_sum
from util.layout_util import (
    convert_layout_to_mask,
    get_ct_idxs_by_dim,
    get_cts_by_dim,
    get_segment,
)
from util.util import prod


def lower_matmul(env, kernel):
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert a_cts.keys() == b_cts.keys()

    # create cs pointers
    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in a_cts.values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    # calculate the multiplications between ct
    cts = {}
    for i, (a, b) in enumerate(zip(a_cs, b_cs)):
        mul_term = a * b
        cts[i] = mul_term

    # Create initial layout_cts with input layout
    layout_cts = LayoutCiphertexts(layout=kernel.cs[0].layout, cts=cts)

    # find summing dimension
    sum_dim = max(
        dim.dim for dim in kernel.cs[0].layout.get_dims() if dim.dim is not None
    )
    ct_sum_dims, slot_sum_dims = find_sum_dim(kernel.cs[0].layout, sum_dim)

    # sum together ciphertexts
    for ct_sum_dim in ct_sum_dims:
        ct_groups = get_cts_by_dim(layout_cts, ct_sum_dim)

        # sum within group
        sum_cts = {}
        for i, group in enumerate(ct_groups):
            base = group[0]
            for j in range(1, len(group)):
                base = base + group[j]
            sum_cts[i] = base

        # Create new layout without the summed dimension
        new_layout = create_layout_without_dims(layout_cts.layout, [ct_sum_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=sum_cts)

    # sum together slot dimensions per ciphertext
    for slot_sum_dim in slot_sum_dims:
        segment = get_segment(slot_sum_dim, layout_cts.layout.slot_dims)
        extent = segment[1]
        mul_offset = segment[2]
        summed_cts = {}
        for index, term in layout_cts.cts.items():
            summed_cts[index] = rotate_and_sum(term, extent, mul_offset)

        # Create new layout without the summed dimension
        new_layout = create_layout_without_dims(layout_cts.layout, [slot_sum_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=summed_cts)

    # mask out gap dimensions
    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(kernel.layout)])
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            mask_term = term * mask
            masked_cts[index] = mask_term
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=masked_cts)
    else:
        # Update layout to output layout if no mask needed
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)

    return layout_cts


def lower_bsgs_matmul(env, kernel):
    if kernel.cs[1].op == KernelOp.BSGS_ROT_ROLL:
        replicated_kernel = kernel.cs[1].cs[1]
        rot_rolled_kernel = kernel.cs[1]
        rolled_kernel = kernel.cs[2]
    else:
        replicated_kernel = kernel.cs[2].cs[1]
        rot_rolled_kernel = kernel.cs[2]
        rolled_kernel = kernel.cs[1]

    replicated_cts = env[replicated_kernel]
    rolled_cts = env[rolled_kernel]
    assert replicated_cts.keys() == rolled_cts.keys()

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
        bsgs_terms[i] = replicated_cts[ct_group[0]]
        other_cts = {}
        for j, ct in enumerate(ct_group):
            other_cts[j] = rolled_cts[ct]
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
    sum_dim = max(
        dim.dim for dim in kernel_cs[0].layout.get_dims() if dim.dim is not None
    )

    slot_sum_dims = []
    for a_dim, b_dim in zip(kernel.cs[1].layout.slot_dims, kernel.cs[2].layout.slot_dims):
        if a_dim.dim is not None and b_dim.dim is not None:
            slot_sum_dims.append(a_dim)

    # recreate layout ct
    output_dims = []
    for ct_dim in kernel.cs[1].layout.ct_dims:
        if ct_dim.dim != sum_dim:
            output_dims.append(ct_dim)
    output_dims.extend(kernel.cs[1].layout.slot_dims)

    output_layout = Layout(
        kernel.cs[1].layout.term,
        [],
        output_dims,
        kernel.cs[1].layout.offset,
        kernel.cs[1].layout.n,
        rolled_kernel.layout.secret,
    )

    layout_cts = LayoutCiphertexts(layout=output_layout, cts=cts)
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

        # Create layout without summed dimension and return LayoutCiphertexts
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=new_cts)

    for slot_sum_dim in slot_sum_dims:
        segment = get_segment(slot_sum_dim, layout_cts.layout.slot_dims)
        extent = segment[1]
        mul_offset = segment[2]
        summed_cts = {}
        for index, term in layout_cts.cts.items():
            summed_cts[index] = rotate_and_sum(term, extent, mul_offset)

        # Create new layout without the summed dimension
        new_layout = create_layout_without_dims(layout_cts.layout, [slot_sum_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=summed_cts)


    # mask out gap dimensions
    needs_mask = False
    for dim in layout_cts.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm.mask([convert_layout_to_mask(layout_cts.layout)])
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            mask_term = term * mask
            masked_cts[index] = mask_term
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=masked_cts)
    else:
        # Update layout to output layout if no mask needed
        layout_cts = LayoutCiphertexts(layout=layout_cts.layout, cts=layout_cts.cts)
    
    return layout_cts