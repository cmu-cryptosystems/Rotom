from lower.layout_cts import LayoutCiphertexts


def lower_reshape(env, kernel):
    """Lower a RESHAPE kernel by reusing ciphertexts under a reshaped layout."""
    input_cts = env[kernel.cs[0]]
    return LayoutCiphertexts(layout=kernel.layout, cts=input_cts.cts)
