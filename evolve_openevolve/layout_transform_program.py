"""
OpenEvolve mutates this file to search layout transformations.

Implement::

    def propose_transformed_kernel(seed_kernel: Kernel) -> Kernel

The seed is a TENSOR kernel from ``layout_transform_search.build_seed_tensor_kernel``.
Use ``layout_transform_lib`` (apply_permute_traversal_dims, apply_append_roll) to explore.
Invalid rolls (extent mismatch) or broken layouts will fail lowering or Toy checks.

Do not change the function name. Preserve imports the evaluator relies on.
"""

from __future__ import annotations

from ir.kernel import Kernel

# Example: from evolve_openevolve.layout_transform_lib import apply_permute_traversal_dims


def propose_transformed_kernel(seed_kernel: Kernel) -> Kernel:
    """Baseline: identity (no layout change)."""
    return seed_kernel
