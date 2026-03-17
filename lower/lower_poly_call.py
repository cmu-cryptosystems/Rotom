from frontends.tensor_args import PolyCallArgs
from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts


def lower_poly_call(env, kernel):
    input_cts = env[kernel.cs[0]]
    tensor_term = kernel.layout.term
    args = PolyCallArgs.from_term(tensor_term)
    cts = {}
    metadata = {}
    metadata["poly_func"] = args.name
    metadata["lower_bound"] = args.lower_bound
    metadata["upper_bound"] = args.upper_bound
    for k, v in input_cts.items():
        cts[k] = HETerm(HEOp.POLY_CALL, [v, metadata], v.secret)
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
