from ir.he import HEOp, HETerm
from ir.kernel import Kernel, KernelOp
from lower.layout_cts import LayoutCiphertexts
from lower.lower_sum import lower_sum


def lower_mean(env, kernel):
    """Reduce-sum over ``mean_dims`` then multiply by ``inv_scale`` (public)."""
    sum_kernel = Kernel(
        KernelOp.SUM,
        [kernel.cs[0], kernel.cs[1]],
        kernel.layout,
    )
    summed = lower_sum(env, sum_kernel)
    inv_scale = float(kernel.cs[2])
    scale_term = HETerm(HEOp.CONST, [kernel.layout, inv_scale], False)
    scaled = {}
    for i, term in summed.cts.items():
        scaled[i] = term * scale_term
    return LayoutCiphertexts(layout=kernel.layout, cts=scaled)
