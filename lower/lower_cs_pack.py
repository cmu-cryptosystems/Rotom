from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from util.layout_util import layout_to_ct_indices
from util.shape_util import get_term_shape


def lower_cs_pack(kernel):
    layout = kernel.layout
    shape = get_term_shape(layout.term)

    # evaluate ct dims
    if layout.ct_dims:
        cts = {}
        ct_indices = layout_to_ct_indices(layout)
        for i, offset in enumerate(ct_indices):
            if not all(a < b for a, b in zip(offset, shape)):
                cts[i] = HETerm(HEOp.ZERO_MASK, [], False)
            else:
                cts[i] = HETerm(
                    HEOp.CS_PACK, [kernel.cs[0], layout], layout.secret, f"{i} {kernel}"
                )
        return LayoutCiphertexts(layout=layout, cts=cts)

    else:
        term = HETerm(
            HEOp.CS_PACK, [kernel.cs[0], layout], layout.secret, f"0 {kernel}"
        )
        return LayoutCiphertexts(layout=layout, cts={0: term})
