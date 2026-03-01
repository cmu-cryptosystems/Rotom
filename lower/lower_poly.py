from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts


def lower_poly(env, kernel):
    input_cts = env[kernel.cs[0]]
    # Pass poly descriptor so backends (e.g. toy) can apply the actual function
    tensor_term = kernel.layout.term
    poly_func = tensor_term.cs[1] if len(tensor_term.cs) > 1 else "identity"
    # Callables are not serializable; backends that need them use identity
    if callable(poly_func):
        poly_func = None
    cts = {}
    for k, v in input_cts.items():
        cts[k] = HETerm(HEOp.POLY, [v], v.secret, poly_func=poly_func)
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
