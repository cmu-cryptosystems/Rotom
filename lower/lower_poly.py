from ir.he import HEOp, HETerm


def lower_poly(env, kernel):
    cts = {}
    for k, v in env[kernel.cs[0]].items():
        cts[k] = HETerm(HEOp.POLY, [v])
    return cts
