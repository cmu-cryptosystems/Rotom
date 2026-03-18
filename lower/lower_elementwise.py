from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts


def lower_elementwise_binop(env, kernel, op):
    """
    Lower an elementwise binary kernel to ciphertext operations.

    `op` should be a callable taking two `HETerm` operands and returning a
    new `HETerm`, for example: `lambda a, b: a + b`.
    """
    a_cts = env[kernel.cs[0]]
    b_cts = env[kernel.cs[1]]
    assert len(a_cts.keys()) == len(b_cts.keys())

    a_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in a_cts.values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret) for ct in b_cts.values()]

    cts = {}
    for i, (a, b) in enumerate(zip(a_cs, b_cs)):
        cts[i] = op(a, b)
    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
