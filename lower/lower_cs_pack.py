from ir.analysis.shape import Shape
from ir.he import HEOp, HETerm
from util.layout_util import layout_to_ct_indices


def lower_cs_pack(kernel):
    layout = kernel.layout
    shape = Shape(layout.term).get_shape()

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
        return cts

    else:
        term = HETerm(
            HEOp.CS_PACK, [kernel.cs[0], layout], layout.secret, f"0 {kernel}"
        )
        return {0: term}
