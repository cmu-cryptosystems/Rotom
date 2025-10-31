from frontends.tensor import TensorOp
from ir.he import HEOp, HETerm
from util.layout_util import layout_to_ct_indices


def get_term_shape(term):
    """Get the shape of a tensor term directly.
    
    Args:
        term: TensorTerm to get shape from
        
    Returns:
        list: Shape of the tensor
    """
    if term.op == TensorOp.TENSOR:
        return term.cs[1]
    else:
        # For other ops, this would need proper shape inference
        # but CS_PACK is only called on TENSOR terms
        raise NotImplementedError(f"get_term_shape not implemented for {term.op}")


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
        return cts

    else:
        term = HETerm(
            HEOp.CS_PACK, [kernel.cs[0], layout], layout.secret, f"0 {kernel}"
        )
        return {0: term}
