from lower.layout_cts import LayoutCiphertexts


def lower_select(env, kernel):
    input_cts = env[kernel.cs[0]]
    selected_cts = {}
    selected_cts[0] = input_cts[kernel.cs[1]]
    return LayoutCiphertexts(layout=kernel.layout, cts=selected_cts)
