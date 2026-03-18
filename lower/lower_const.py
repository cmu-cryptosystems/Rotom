from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts


def lower_const(kernel):
    """Lower a CONST kernel into layout-aligned constant ciphertexts."""
    layout = kernel.layout
    cts = {}
    for i in range(layout.num_ct()):
        cts[i] = HETerm(HEOp.CONST, [layout, kernel.layout.term.cs[0]], layout.secret)
    return LayoutCiphertexts(layout=layout, cts=cts)
