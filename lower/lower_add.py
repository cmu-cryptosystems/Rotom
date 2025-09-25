from ir.he import HETerm, HEOp


def lower_add(env, kernel):
    assert len(env[kernel.cs[0]].keys()) == len(env[kernel.cs[1]].keys())

    a_cs = [HETerm(HEOp.CS, [ct], ct.secret)
            for ct in env[kernel.cs[0]].values()]
    b_cs = [HETerm(HEOp.CS, [ct], ct.secret)
            for ct in env[kernel.cs[1]].values()]

    cts = {}
    for i, (a, b) in enumerate(zip(a_cs, b_cs)):
        cts[i] = a + b
    return cts
