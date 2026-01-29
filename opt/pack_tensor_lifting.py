"""
Pack tensor lifting optimization pass.

This optimization analyzes kernels to find cases where input tensors are
wrapped in REPLICATE operations (and potentially ROT_ROLL operations).
Instead of performing replication and rotation (compute), we can pack the
tensor directly in its rotated/replicated form and send more ciphertexts
(communication). This trades compute for communication.

Key Concepts:
- Pattern Detection: Find REPLICATE (+ ROT_ROLL) patterns on input tensors
- Alternative Packing: Create TENSOR with rolls/replication in layout instead
- Cost Comparison: Evaluate compute vs communication trade-off
- Kernel Transformation: Replace REPLICATE(+ROT_ROLL) with packed TENSOR
"""

from copy import deepcopy as copy

from frontends.tensor import TensorOp
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from ir.layout_utils import dimension_merging


def find_replicate_patterns(kernel, parent_map=None):
    """
    Find REPLICATE patterns on input tensors that can be lifted to packed TENSOR.

    Returns a list of (replicate_kernel, rot_roll_kernel_or_none, tensor_term) tuples
    where the pattern can be optimized.
    """
    if parent_map is None:
        parent_map = {}

        def build_parent_map(k, parent=None):
            if parent is not None:
                parent_map[k] = parent
            if hasattr(k, "cs") and k.cs:
                for child in k.cs:
                    if not isinstance(child, (int, tuple)):
                        build_parent_map(child, k)

        build_parent_map(kernel)

    patterns = []

    # Find REPLICATE kernels that wrap input tensors
    for k in kernel.post_order():
        if k.op == KernelOp.REPLICATE and k.cs and len(k.cs) > 0:
            replicate_child = k.cs[0]

            # Check if REPLICATE wraps a TENSOR or CS (input tensor)
            if replicate_child.op in [KernelOp.TENSOR, KernelOp.CS]:
                # Get the original tensor term
                if (
                    hasattr(replicate_child.layout, "term")
                    and replicate_child.layout.term
                    and hasattr(replicate_child.layout.term, "op")
                    and replicate_child.layout.term.op == TensorOp.TENSOR
                ):

                    # Check if this REPLICATE has a ROT_ROLL parent
                    rot_roll_parent = None
                    if k in parent_map:
                        parent = parent_map[k]
                        if parent.op == KernelOp.ROT_ROLL:
                            rot_roll_parent = parent

                    patterns.append((k, rot_roll_parent, replicate_child.layout.term))

    return patterns


def create_packed_tensor(replicate_kernel, rot_roll_kernel, tensor_term):
    """
    Create a packed TENSOR kernel with the replication/rolls in the layout.

    Instead of REPLICATE (+ ROT_ROLL), we pack the tensor directly with
    the rolls/replication in its layout, trading compute for communication.

    Args:
        replicate_kernel: The REPLICATE kernel
        rot_roll_kernel: The ROT_ROLL kernel that wraps REPLICATE (or None)
        tensor_term: The original tensor term

    Returns:
        New TENSOR kernel with rolls/replication in layout, or None if not possible
    """
    # Use the layout from ROT_ROLL if present, otherwise from REPLICATE
    if rot_roll_kernel is not None:
        source_layout = rot_roll_kernel.layout
    else:
        source_layout = replicate_kernel.layout

    # Create a TENSOR kernel with the source layout but the original tensor term
    # The layout already has the rolls and dimensions we need
    try:
        packed_layout = Layout(
            tensor_term,
            source_layout.rolls,  # Rolls from ROT_ROLL or REPLICATE
            source_layout.dims,  # Dimensions from ROT_ROLL or REPLICATE
            source_layout.n,
            source_layout.secret,
        )
        # Use dimension_merging to ensure rolls reference correct dimension objects
        packed_layout = dimension_merging(packed_layout)

        packed_kernel = Kernel(
            KernelOp.TENSOR,
            [],
            packed_layout,
        )
        return packed_kernel
    except (AssertionError, ValueError) as e:
        # Layout creation failed (e.g., roll dimension mismatch)
        return None


def evaluate_packing_vs_replication(
    replicate_kernel, rot_roll_kernel, packed_kernel, network="lan"
):
    """
    Evaluate whether packing is cheaper than replication (+ rotation).

    Args:
        replicate_kernel: Original REPLICATE kernel
        rot_roll_kernel: Original ROT_ROLL kernel (or None)
        packed_kernel: Alternative packed TENSOR kernel
        network: Network type for cost calculation

    Returns:
        Tuple of (should_pack: bool, compute_cost, comm_cost)
    """
    # Cost of current approach: REPLICATE (+ ROT_ROLL)
    replicate_cost = KernelCost(replicate_kernel, network)
    current_compute_cost = replicate_cost.op_cost()

    if rot_roll_kernel is not None:
        rot_roll_cost = KernelCost(rot_roll_kernel, network)
        current_compute_cost += rot_roll_cost.op_cost()

    # Cost of packed approach: Communication for extra ciphertexts
    packed_cost = KernelCost(packed_kernel, network)
    packed_num_cts = packed_kernel.layout.num_ct()

    # Original tensor would have fewer ciphertexts
    original_num_cts = (
        replicate_kernel.cs[0].layout.num_ct()
        if hasattr(replicate_kernel.cs[0].layout, "num_ct")
        else 1
    )
    extra_cts = packed_num_cts - original_num_cts

    cost_model = packed_cost.cost_model()
    comm_cost = cost_model["comm"] * extra_cts

    # Pack if communication cost is less than compute cost
    # This trades compute (replication + rotation) for communication (more ciphertexts)
    should_pack = comm_cost < current_compute_cost

    return should_pack, current_compute_cost, comm_cost


def apply_pack_tensor_lifting(kernel, network="lan"):
    """
    Apply pack tensor lifting optimization to a kernel.

    This finds REPLICATE (+ ROT_ROLL) patterns on input tensors and replaces
    them with packed TENSOR kernels when beneficial.

    Args:
        kernel: Kernel to optimize
        network: Network type for cost calculation

    Returns:
        Optimized kernel with packing optimizations applied
    """
    optimized = copy(kernel)
    update_map = {}

    # Build parent map
    parent_map = {}

    def build_parent_map(k, parent=None):
        if parent is not None:
            parent_map[k] = parent
        if hasattr(k, "cs") and k.cs:
            for child in k.cs:
                if not isinstance(child, (int, tuple)):
                    build_parent_map(child, k)

    build_parent_map(optimized)

    # Find patterns
    patterns = find_replicate_patterns(optimized, parent_map)

    if not patterns:
        # No patterns found, return original kernel
        return optimized

    # Process patterns in post-order to handle nested cases
    for replicate_kernel, rot_roll_kernel, tensor_term in patterns:
        # Skip if already processed
        if replicate_kernel in update_map or (
            rot_roll_kernel and rot_roll_kernel in update_map
        ):
            continue

        # Create packed alternative
        packed_kernel = create_packed_tensor(
            replicate_kernel, rot_roll_kernel, tensor_term
        )
        if packed_kernel is None:
            continue

        # Evaluate cost
        should_pack, compute_cost, comm_cost = evaluate_packing_vs_replication(
            replicate_kernel, rot_roll_kernel, packed_kernel, network
        )

        if should_pack:
            # Replace REPLICATE with packed TENSOR
            update_map[replicate_kernel] = packed_kernel
            # Replace ROT_ROLL with packed TENSOR if present (rolls are now in TENSOR layout)
            if rot_roll_kernel is not None:
                update_map[rot_roll_kernel] = packed_kernel

    # Apply updates by rebuilding the kernel tree
    def rebuild_kernel(k):
        # Check if this kernel should be replaced
        if k in update_map:
            return update_map[k]

        # Rebuild children first
        if hasattr(k, "cs") and k.cs:
            new_cs = []
            for cs in k.cs:
                if isinstance(cs, (int, tuple)):
                    new_cs.append(cs)
                else:
                    new_cs.append(rebuild_kernel(cs))
            # Create new kernel with updated children
            return Kernel(k.op, new_cs, k.layout)

        return k

    optimized = rebuild_kernel(optimized)

    return optimized


def run_pack_tensor_lifting(kernel, network="lan"):
    """
    Run pack tensor lifting optimization pass.

    This is the main entry point for the optimization. It analyzes the kernel
    for REPLICATE (+ ROT_ROLL) patterns and replaces them with packed TENSOR
    kernels when it's cheaper to trade compute for communication.

    Args:
        kernel: Kernel to optimize (typically the final selected kernel)
        network: Network type for cost calculation

    Returns:
        Optimized kernel with packing optimizations applied
    """
    return apply_pack_tensor_lifting(kernel, network)
