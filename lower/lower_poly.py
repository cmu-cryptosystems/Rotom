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
        # Pass channel index for batchnorm so toy backend uses correct per-channel params
        poly_channel = (
            k
            if isinstance(poly_func, tuple)
            and poly_func
            and poly_func[0] == "batchnorm"
            else None
        )
        cts[k] = HETerm(
            HEOp.POLY, [v], v.secret, poly_func=poly_func, poly_channel=poly_channel
        )
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
