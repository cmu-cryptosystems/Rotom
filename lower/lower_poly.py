from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts


def lower_poly(env, kernel):
    input_cts = env[kernel.cs[0]]
    cts = {}
    for k, v in input_cts.items():
        cts[k] = HETerm(HEOp.POLY, [v])
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
