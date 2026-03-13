from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts

def lower_poly_call(env, kernel):
    input_cts = env[kernel.cs[0]]
    # Pass poly descriptor so backends (e.g. toy) can apply the actual function
    tensor_term = kernel.layout.term
    poly_func = tensor_term.cs[1] 
    upper_bound = tensor_term.cs[2]
    lower_bound = tensor_term.cs[3]
    cts = {}
    metadata = {}
    metadata["poly_func"] = poly_func
    metadata["upper_bound"] = upper_bound
    metadata["lower_bound"] = lower_bound
    for k, v in input_cts.items():
        cts[k] = HETerm(
            HEOp.POLY_CALL, [v, metadata], v.secret
        )
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)