from ir.dim import DimType
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts, create_layout_without_dims
from lower.lower_util import rotate_and_product
from util.layout_util import convert_layout_to_mask, get_cts_by_dim, get_segment


def lower_product(env, kernel):
    """Lower PRODUCT by multiplying over ciphertext groups and slot trees."""
    input_cts = env[kernel.cs[0]]
    layout_cts = input_cts
    prod_dims = kernel.cs[1]

    ct_prod_dims = []
    slot_prod_dims = []
    for d in prod_dims:
        if d in layout_cts.layout.ct_dims:
            ct_prod_dims.append(d)
        elif d in layout_cts.layout.slot_dims:
            slot_prod_dims.append(d)

    for ct_dim in ct_prod_dims:
        groups = get_cts_by_dim(layout_cts, ct_dim)
        prod_cts = {}
        for i, group in enumerate(groups):
            base = group[0]
            for j in range(1, len(group)):
                base = base * group[j]
            prod_cts[i] = base
        new_layout = create_layout_without_dims(layout_cts.layout, [ct_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=prod_cts)

    for slot_dim in slot_prod_dims:
        segment = get_segment(slot_dim, layout_cts.layout.slot_dims)
        extent = segment[1]
        mul_offset = segment[2]
        out_cts = {}
        for index, term in layout_cts.cts.items():
            out_cts[index] = rotate_and_product(term, extent, mul_offset)
        new_layout = create_layout_without_dims(layout_cts.layout, [slot_dim])
        layout_cts = LayoutCiphertexts(layout=new_layout, cts=out_cts)

    needs_mask = False
    for dim in kernel.layout.slot_dims:
        if dim.dim_type == DimType.EMPTY:
            needs_mask = True
            break

    if needs_mask:
        mask = HETerm(HEOp.MASK, [convert_layout_to_mask(kernel.layout)], False, "mask")
        masked_cts = {}
        for index, term in layout_cts.cts.items():
            masked_cts[index] = mask * term
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=masked_cts)
    else:
        layout_cts = LayoutCiphertexts(layout=kernel.layout, cts=layout_cts.cts)

    return layout_cts
