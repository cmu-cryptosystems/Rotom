from lower.layout_cts import LayoutCiphertexts


def lower_transpose(env, kernel):
    input_cts = env[kernel.cs[0]]
    return LayoutCiphertexts(layout=kernel.layout, cts=input_cts.cts)
