from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from lower.metadata import kernel_metadata


def lower_poly(env, kernel):
    input_cts = env[kernel.cs[0]]
    cts = {}
    metadata = kernel_metadata(kernel, "poly")
    for k, v in input_cts.items():
        cts[k] = HETerm(HEOp.POLY, [v], v.secret, metadata)
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
