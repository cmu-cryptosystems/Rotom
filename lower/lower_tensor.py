from ir.analysis.shape import Shape
from ir.he import HEOp, HETerm
from util.shape_util import layout_to_shape_indices


def lower_tensor(kernel):
    layout = kernel.layout

    # get shape of term
    shape = Shape(layout.term)
    shape.run()
    layout_shape = shape.shapes[layout.term]

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
