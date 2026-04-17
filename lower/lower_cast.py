from lower.layout_cts import LayoutCiphertexts


def lower_cast(env, kernel):
    """CAST is a layout-preserving no-op on ciphertext slots (dtype only affects NumPy eval)."""
    inc = env[kernel.cs[0]]
    return LayoutCiphertexts(layout=kernel.layout, cts=dict(inc.cts))
