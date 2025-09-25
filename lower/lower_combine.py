def lower_combine(env, kernel):
    cts = {}
    ct_idx = 0
    for cs_kernel in kernel.cs:
        for ct in env[cs_kernel].values():
            cts[ct_idx] = ct
            ct_idx += 1
    return cts
