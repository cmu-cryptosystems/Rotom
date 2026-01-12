from ir.he import HEOp, HETerm
from util.shape_util import get_term_shape, layout_to_shape_indices


def lower_toeplitz_tensor(kernel):
    """Lower TOEPLITZ_TENSOR kernel to circuit IR.

    This function handles packing for Toeplitz tensors differently from regular TENSOR.
    For Toeplitz convolution, weights are public and need to be packed into a Toeplitz
    matrix structure for efficient convolution computation.

    Args:
        kernel: Kernel with op TOEPLITZ_TENSOR

    Returns:
        dict: Dictionary mapping ciphertext indices to HETerm nodes
    """
    layout = kernel.layout
    
    # evaluate ct dims
    if layout.ct_dims:
        cts = {}
        ct_indices = layout_to_shape_indices(layout)
        for i, _ in enumerate(ct_indices):
            cts[i] = HETerm(HEOp.TOEPLITZ_PACK, [layout], layout.secret, f"{i} {kernel} (toeplitz)")
        return cts
    else:
        # For Toeplitz tensors, use TOEPLITZ_PACK operation (or PACK with special handling)
        # This can be extended to use a different packing strategy for Toeplitz matrices
        term = HETerm(HEOp.TOEPLITZ_PACK, [layout], layout.secret, f"0 {kernel} (toeplitz)")
        return {0: term}
