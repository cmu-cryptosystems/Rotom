from ir.he import HEOp, HETerm
from lower.layout_cts import LayoutCiphertexts
from lower.metadata import kernel_metadata


def lower_rescale(env, kernel):
    """Lower a rescale operation to HE terms.

    Args:
        env: Environment mapping kernels to their computed values
        kernel: The rescale kernel operation

    Returns:
        LayoutCiphertexts: Layout-aware ciphertexts with rescaled terms
    """
    # Get the input tensor's ciphertexts
    input_cts = env[kernel.cs[0]]
    scale_exp = kernel.cs[1]  # The exponent for the scale (e.g., 14 for 2^14)

    # For each ciphertext in the input, apply rescale
    cts = {}
    metadata = kernel_metadata(kernel, "rescale")
    for i, ct in input_cts.items():
        # Create a RESCALE HETerm, passing the scale exponent
        rescale_term = HETerm(HEOp.RESCALE, [ct, scale_exp], ct.secret, metadata)
        cts[i] = rescale_term

    return LayoutCiphertexts(layout=kernel.layout, cts=cts)
