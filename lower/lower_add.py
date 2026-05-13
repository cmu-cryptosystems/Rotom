from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from lower.metadata import kernel_metadata


def lower_add(env, kernel):
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert len(a_cts.keys()) == len(b_cts.keys())

    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in a_cts.values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    cts = {}
    metadata = kernel_metadata(kernel, "add")
    for i, (a, b) in enumerate(zip(a_cs, b_cs)):
        cts[i] = HETerm(HEOp.ADD, [a, b], a.secret or b.secret, metadata)
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
