from ir.he import HEOp, HETerm
from util.shape_util import get_term_shape, layout_to_shape_indices


def lower_tensor(kernel):
    layout = kernel.layout

    # get shape of term
    layout_shape = get_term_shape(layout.term)

    # evaluate ct dims
    if layout.ct_dims:
        cts = {}
        ct_indices = layout_to_shape_indices(layout)
        for i, offset in enumerate(ct_indices):
            if not all(a < b for a, b in zip(offset, layout_shape)):
                cts[i] = HETerm(HEOp.ZERO_MASK, [], False)
            else:
                cts[i] = HETerm(HEOp.PACK, [layout], layout.secret, f"{i} {kernel}")
        return cts

    else:
        term = HETerm(HEOp.PACK, [layout], layout.secret, f"0 {kernel}")
        return {0: term}
