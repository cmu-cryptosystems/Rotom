from ir.he import HEOp, HETerm
from util.shape_util import get_term_shape, layout_to_shape_indices


def lower_punctured_tensor(kernel):
    """Lower PUNCTURED_TENSOR kernel to circuit IR.

    This function handles packing for punctured tensors differently from regular TENSOR.
    For punctured convolution, weights are public and need to be packed into a punctured
    matrix structure for efficient convolution computation.

    Args:
        kernel: Kernel with op PUNCTURED_TENSOR

    Returns:
        dict: Dictionary mapping ciphertext indices to HETerm nodes
    """
    layout = kernel.layout

    # evaluate ct dims
    if layout.ct_dims:
        cts = {}
        ct_indices = layout_to_shape_indices(layout)
        for i, _ in enumerate(ct_indices):
            cts[i] = HETerm(
                HEOp.PUNCTURED_PACK,
                [layout],
                layout.secret,
                f"{i} {kernel} (punctured)",
            )
        return cts
    else:
        # For punctured tensors, use PUNCTURED_PACK operation (or PACK with special handling)
        # This can be extended to use a different packing strategy for punctured matrices
        term = HETerm(
            HEOp.PUNCTURED_PACK, [layout], layout.secret, f"0 {kernel} (punctured)"
        )
        return {0: term}
