from lower.layout_cts import LayoutCiphertexts


def lower_combine(env, kernel):
    cts = {}
    ct_idx = 0
    for cs_kernel in kernel.cs:
        input_cts = env[cs_kernel]
        for ct in input_cts.values():
            cts[ct_idx] = ct
            ct_idx += 1

    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
