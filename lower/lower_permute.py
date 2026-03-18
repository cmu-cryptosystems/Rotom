from lower.layout_cts import LayoutCiphertexts


def lower_permute(env, kernel):
    """Lower a PERMUTE kernel by reinterpreting ciphertexts with permuted dimensions."""
    input_cts = env[kernel.cs[0]]
    return LayoutCiphertexts(layout=kernel.layout, cts=input_cts.cts)
