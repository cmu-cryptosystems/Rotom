"""
Strassen's algorithm layout generation utilities.

This module provides functions for generating optimal layouts for matrix
multiplication using Strassen's algorithm in FHE computations. Strassen's
algorithm reduces the number of multiplications required for matrix
multiplication from O(n³) to approximately O(n^2.81) by using a divide-and-conquer
approach with clever submatrix operations.

Key functions:
- get_shape: Extracts shape information from kernels
- traverse_ct_dims_helper: Helper for traversing ciphertext dimensions
- gen_strassens: Main function for generating Strassen's algorithm layouts
"""

from copy import deepcopy as copy

from assignment.gen.gen_binop import gen_binop
from ir.dim import Dim
from ir.kernel import Kernel, KernelOp
from ir.kernel_cost import KernelCost
from ir.layout import Layout
from ir.roll import Roll
from opt.opt import Optimizer
from util.layout_util import dimension_merging


def layout_to_str(layout):
    return str(layout.rolls) + " " + str(layout.get_dims())


def get_shape(kernel):
    """Extracts shape information from a kernel layout.

    This function analyzes a kernel's layout dimensions and returns
    the corresponding tensor shape. It aggregates dimensions with the
    same index to compute the total extent for each dimension.

    Args:
        kernel: Kernel to extract shape from

    Returns:
        List of integers representing the tensor shape
    """
    shapes = {}
    for dim in kernel.layout.get_dims():
        if dim.dim is None:
            continue
        if dim.dim not in shapes:
            shapes[dim.dim] = dim.extent
        else:
            shapes[dim.dim] *= dim.extent
    shape = []
    for i in range(len(shapes)):
        shape.append(shapes[i])
    return shape


def traverse_ct_dims_helper(remaining_ct_dims, offsets):
    if not remaining_ct_dims:
        return offsets
    ct_dim = remaining_ct_dims[0]
    dim = ct_dim.dim
    extent = ct_dim.extent
    stride = ct_dim.stride

    dim_offsets = [i * stride for i in range(extent)]

    if dim not in offsets:
        offsets[dim] = dim_offsets
    else:
        new_offsets = []
        for a in offsets[dim]:
            for b in dim_offsets:
                new_offsets.append(a + b)
        offsets[dim] = new_offsets
    return traverse_ct_dims_helper(remaining_ct_dims[1:], offsets)


def traverse_ct_dims(ct_dims):
    offsets = {}
    offsets = traverse_ct_dims_helper(ct_dims, offsets)

    indices = []
    if len(offsets) == 1:
        for k, v in offsets.items():
            for i in v:
                indices.append({k: i})
    elif len(offsets) == 2:
        for k, v in offsets.items():
            if not indices:
                for i in v:
                    indices.append({k: i})
            else:
                new_indices = []
                for index in indices:
                    for i in v:
                        new_index = copy(index)
                        new_index[k] = i
                        new_indices.append(new_index)
                indices = new_indices

    # Compute tile size for each dimension (minimum non-zero offset, or difference between offsets)
    tile_sizes = {}
    for dim, dim_offsets in offsets.items():
        sorted_offsets = sorted(set(dim_offsets))
        if len(sorted_offsets) > 1:
            # Use the difference between the first two offsets as tile size
            tile_sizes[dim] = sorted_offsets[1] - sorted_offsets[0]
        else:
            # Only one offset, use the offset value itself as tile size
            tile_sizes[dim] = sorted_offsets[0] if sorted_offsets else 1

    output_offsets = []
    for offset in indices:
        # Get tile size for each dimension, default to 1 if not found
        dim0_tile = tile_sizes.get(0, 1)
        dim1_tile = tile_sizes.get(1, 1)

        dim0_start = offset.get(0, 0)
        dim1_start = offset.get(1, 0)

        output_offsets.append(
            [
                dim0_start,
                dim0_start + dim0_tile,
                dim1_start,
                dim1_start + dim1_tile,
            ]
        )

    return output_offsets


def check_slot_dims_in_tiles(a_kernel, b_kernel):
    a_slot_dims = a_kernel.layout.slot_dims
    b_slot_dims = b_kernel.layout.slot_dims

    # check that extents match
    a_extents = [dim.extent for dim in a_slot_dims]
    b_extents = [dim.extent for dim in b_slot_dims]
    if a_extents != b_extents:
        return False

    # check that summation dimensions are aligned
    for a_dim, b_dim in zip(a_slot_dims, b_slot_dims):
        if a_dim.dim is not None and b_dim.dim is not None and a_dim.dim == b_dim.dim:
            return False

    return True


def check_ct_dims_in_tiles(a_kernel, b_kernel):
    a_ct_dims = a_kernel.layout.ct_dims
    b_ct_dims = b_kernel.layout.ct_dims

    # check that both are tiled into squares
    if (
        len(set([ct_dim.dim for ct_dim in a_ct_dims])) != 2
        or len(set([ct_dim.dim for ct_dim in b_ct_dims])) != 2
    ):
        return False
    # TODO: check that both can be tiled into squares
    return True


def check_no_rolls(kernel):
    return not kernel.layout.rolls


def check_strassens(a_kernel, b_kernel):
    """
    Check if two kernels are compatible for Strassen's algorithm.
    
    Requires that input kernels have no rolls (for tiling purposes).
    The internal matmuls within strassens will use roll-based matmul
    when roll_flag is True, but the input kernels must not have rolls.
    """
    return (
        check_slot_dims_in_tiles(a_kernel, b_kernel)
        and check_ct_dims_in_tiles(a_kernel, b_kernel)
        and check_no_rolls(a_kernel)
        and check_no_rolls(b_kernel)
    )


def add_tiles(kernel_map, a_kernels, b_kernels, roll_flag, network):
    output_kernels = {}
    output_kernel_costs = {}
    for layout_str in a_kernels.keys():
        a_kernel = a_kernels[layout_str]
        b_kernel = b_kernels[layout_str]
        a_kernel = copy(a_kernel)
        b_kernel = copy(b_kernel)

        term = a_kernel.layout.term + b_kernel.layout.term
        term.offset = {}
        add_kernels = gen_binop(
            term,
            [[a_kernel], [b_kernel]],
            [get_shape(a_kernel), get_shape(b_kernel)],
            roll_flag,
        )
        optimizer = Optimizer(roll_flag)
        add_kernels = optimizer.run(add_kernels)
        if term not in kernel_map:
            kernel_map[term] = []
        kernel_map[term] += add_kernels
        
        for add_kernel in add_kernels:
            layout = add_kernel.layout
            layout_str = layout_to_str(layout)
            cost = KernelCost(add_kernel, network).total_cost()
            if layout_str not in output_kernels:
                output_kernels[layout_str] = add_kernel
                output_kernel_costs[layout_str] = cost
            elif cost < output_kernel_costs[layout_str]:
                output_kernels[layout_str] = add_kernel
                output_kernel_costs[layout_str] = cost
    return output_kernels


def sub_tiles(kernel_map, a_kernels, b_kernels, roll_flag, network):
    output_kernels = {}
    output_kernel_costs = {}
    for layout_str in a_kernels.keys():
        a_kernel = a_kernels[layout_str]
        b_kernel = b_kernels[layout_str]
        a_kernel = copy(a_kernel)
        b_kernel = copy(b_kernel)

        term = a_kernel.layout.term - b_kernel.layout.term
        sub_kernels = gen_binop(
            term,
            [[a_kernel], [b_kernel]],
            [get_shape(a_kernel), get_shape(b_kernel)],
            roll_flag,
        )
        optimizer = Optimizer(roll_flag)
        sub_kernels = optimizer.run(sub_kernels)
        if term not in kernel_map:
            kernel_map[term] = []
        kernel_map[term] += sub_kernels
        
        for sub_kernel in sub_kernels:
            layout = sub_kernel.layout
            layout_str = layout_to_str(layout)
            cost = KernelCost(sub_kernel, network).total_cost()
            if layout_str not in output_kernels:
                output_kernels[layout_str] = sub_kernel
                output_kernel_costs[layout_str] = cost
            elif cost < output_kernel_costs[layout_str]:
                output_kernels[layout_str] = sub_kernel
                output_kernel_costs[layout_str] = cost
    return output_kernels


def estimate_downstream_cost(matmul_kernel, network):
    """Estimate the cost of downstream operations that will use this matmul kernel."""
    num_ct = matmul_kernel.layout.num_ct()
    base_add_cost = 0.1
    roll_penalty = 0.0
    if matmul_kernel.layout.rolls:
        roll_penalty = 0.05 * len(matmul_kernel.layout.rolls)
    return num_ct * base_add_cost + roll_penalty


def check_optimal_tile_matmul_layout(a_kernel, b_kernel):
    """
    Check if kernels match the optimal tile matmul input pattern.
    
    The optimal pattern is:
    - A: [1:size][0:size] (dim 1, then dim 0)
    - B: [0:size][1:size] (dim 0, then dim 1)
    
    Both should have no rolls initially.
    
    Args:
        a_kernel: Input kernel A
        b_kernel: Input kernel B
        
    Returns:
        bool: True if kernels match the optimal pattern, False otherwise
    """
    # Check that both kernels have no rolls
    if a_kernel.layout.rolls or b_kernel.layout.rolls:
        return False
    
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    
    # Check both have exactly 2 dimensions
    if len(a_dims) != 2 or len(b_dims) != 2:
        return False
    
    # Extract dimension indices
    a_dim_0 = None
    a_dim_1 = None
    for dim in a_dims:
        if dim.dim == 0:
            a_dim_0 = dim
        elif dim.dim == 1:
            a_dim_1 = dim
    
    b_dim_0 = None
    b_dim_1 = None
    for dim in b_dims:
        if dim.dim == 0:
            b_dim_0 = dim
        elif dim.dim == 1:
            b_dim_1 = dim
    
    # Verify we found the expected dimensions
    if (a_dim_0 is None or a_dim_1 is None or 
        b_dim_0 is None or b_dim_1 is None):
        return False
    
    # Check that dimensions match in extent
    if (a_dim_0.extent != b_dim_0.extent or 
        a_dim_1.extent != b_dim_1.extent):
        return False
    
    # Check order: A should be [1][0], B should be [0][1]
    pattern1 = (a_dims[0].dim == 1 and a_dims[1].dim == 0 and
                 b_dims[0].dim == 0 and b_dims[1].dim == 1)
    
    return pattern1


def matmul_tiles(kernel_map, a_kernels, b_kernels, roll_flag, network):
    output_kernels = {}
    for a_kernel in a_kernels.values():
        for b_kernel in b_kernels.values():
            a_kernel = copy(a_kernel)
            b_kernel = copy(b_kernel)
            term = a_kernel.layout.term @ b_kernel.layout.term

            matmul_kernels = []
            if check_optimal_tile_matmul_layout(a_kernel, b_kernel) and roll_flag:
                optimal_kernel = gen_optimal_tile_matmul(term, a_kernel, b_kernel, kernel_map)
                if term not in kernel_map:
                    kernel_map[term] = []
                kernel_map[term] += [optimal_kernel]
                return {layout_to_str(optimal_kernel.layout): optimal_kernel}
            else:
                matmul_kernels = gen_binop(
                    term,
                [[a_kernel], [b_kernel]],
                [get_shape(a_kernel), get_shape(b_kernel)],
                roll_flag,
            )
                optimizer = Optimizer(roll_flag)
                matmul_kernels = optimizer.run(matmul_kernels)
            if term not in kernel_map:
                kernel_map[term] = []
            kernel_map[term] += matmul_kernels
    return output_kernels


def gen_optimal_tile_matmul(term, a_kernel, b_kernel, kernel_map):
    """
    Generate the optimal tile matmul sequence directly.
    
    Optimal sequence:
    - A: [1:size][0:size] -> BSGS_ROLL(1,0) -> REPLICATE -> ROLL(2,1) -> ROLL(0,1)
    - B: [0:size][1:size] -> REPLICATE -> ROT_ROLL(0,1)
    - Final: MATMUL
    
    Args:
        term: The matmul term (a_term @ b_term)
        a_kernel: Input kernel A with layout [1:size][0:size]
        b_kernel: Input kernel B with layout [0:size][1:size]
        kernel_map: Dictionary to add intermediate kernels to
        
    Returns:
        Kernel: The final matmul kernel with optimal sequence
    """
    a_term = a_kernel.layout.term
    b_term = b_kernel.layout.term
    n = a_kernel.layout.n
    secret = a_kernel.layout.secret
    
    # Extract dimensions
    a_dims = a_kernel.layout.get_dims()
    b_dims = b_kernel.layout.get_dims()
    
    # Extract dimension objects
    a_dim_1 = None
    a_dim_0 = None
    for dim in a_dims:
        if dim.dim == 1:
            a_dim_1 = dim
        elif dim.dim == 0:
            a_dim_0 = dim
    
    b_dim_0 = None
    b_dim_1 = None
    for dim in b_dims:
        if dim.dim == 0:
            b_dim_0 = dim
        elif dim.dim == 1:
            b_dim_1 = dim
    
    if a_dim_1 is None or a_dim_0 is None or b_dim_0 is None or b_dim_1 is None:
        raise ValueError("Expected dimensions 0 and 1 in both kernels")
    
    size = a_dim_1.extent
    
    # Construct optimal sequence for A: BSGS_ROLL(1,0) -> REPLICATE -> ROLL(2,1)
    # Step 1: BSGS_ROLL(1,0) on A
    roll_1_0_a = Roll(a_dim_0, a_dim_1)
    a_bsgs_rolled_layout = Layout(
        a_term,
        [roll_1_0_a],
        [a_dim_1, a_dim_0],
        n,
        secret,
    )
    a_bsgs_rolled = Kernel(KernelOp.BSGS_ROLL, [roll_1_0_a, a_kernel], layout=a_bsgs_rolled_layout)
    kernel_map[a_bsgs_rolled_layout.term] += [a_bsgs_rolled]
    
    # Step 2: REPLICATE A
    replicate_dim = Dim(None, size, 1)
    a_replicated_layout = Layout(
        a_term,
        a_bsgs_rolled_layout.rolls,
        [replicate_dim, a_dim_1, a_dim_0],
        n,
        secret,
    )
    a_replicated = Kernel(KernelOp.REPLICATE, [a_bsgs_rolled], layout=a_replicated_layout)
    kernel_map[a_replicated_layout.term] += [a_replicated]

    # Step 3: ROLL on A
    roll_2_1_a = Roll(a_dim_1, replicate_dim)
    a_replicated_rolled_layout_1 = Layout(
        a_term,
        [roll_2_1_a, Roll(a_dim_0, replicate_dim)],
        [a_dim_1, a_dim_0, replicate_dim],
        n,
        secret,
    )
    a_final = Kernel(KernelOp.ROLL, [roll_2_1_a, a_replicated], layout=a_replicated_rolled_layout_1)
    kernel_map[a_replicated_rolled_layout_1.term] += [a_final]
    
    # Construct optimal sequence for B: REPLICATE -> ROT_ROLL
    # Step 1: REPLICATE B
    b_replicated_layout = Layout(
        b_term,
        [],
        [replicate_dim, b_dim_0, b_dim_1],
        n,
        secret,
    )
    b_replicated = Kernel(KernelOp.REPLICATE, [b_kernel], layout=b_replicated_layout)
    kernel_map[b_replicated_layout.term] += [b_replicated]

    # Step 2: ROT_ROLL on B
    roll_0_1_b = Roll(b_dim_0, replicate_dim)
    b_rot_rolled_layout = Layout(
        b_term,
        [roll_0_1_b],
        [b_dim_0, replicate_dim, b_dim_1],
        n,
        secret,
    )
    b_final = Kernel(KernelOp.ROT_ROLL, [roll_0_1_b, b_replicated], layout=b_rot_rolled_layout)
    
    # Final matmul
    matmul_output_dims = [a_dim_0, b_dim_1]
    matmul_layout = Layout(term, [Roll(a_dim_0, b_dim_1)], matmul_output_dims, n, secret)
    matmul_kernel = Kernel(KernelOp.MATMUL, [a_final, b_final], layout=matmul_layout)
    
    return matmul_kernel


def create_combine_kernel(term, kernel_map, C1, C2, C3, C4):
    assert (
        C1.layout.layout_str()
        == C2.layout.layout_str()
        == C3.layout.layout_str()
        == C4.layout.layout_str()
    )

    extents = {}
    for dim in C1.layout.get_dims():
        if dim.dim is None:
            continue

        if dim.dim not in extents:
            extents[dim.dim] = dim.extent
        else:
            extents[dim.dim] *= dim.extent

    # Ensure we have dimensions 0 and 1 for the 2x2 tile structure
    if 0 not in extents or 1 not in extents:
        raise ValueError(f"Expected dimensions 0 and 1 in extents, got {extents}")

    dims = [Dim(0, 2, extents[0]), Dim(1, 2, extents[1])]
    combine_layout_dims = dims + copy(C1.layout.get_dims())
    combine_layout = Layout(
        term, C1.layout.rolls, combine_layout_dims, C1.layout.n, C1.layout.secret
    )
    combine_kernel = Kernel(KernelOp.COMBINE, [C1, C2, C3, C4], combine_layout)
    if term not in kernel_map:
        kernel_map[term] = []
    kernel_map[term].append(combine_kernel)
    return combine_kernel


def strassens_opt(term, kernel_map, a_tiles, b_tiles, roll_flag, network):
    # perform addition and matmul
    A1 = {layout_to_str(a_tiles[0].layout): a_tiles[0]}
    A2 = {layout_to_str(a_tiles[1].layout): a_tiles[1]}
    A3 = {layout_to_str(a_tiles[2].layout): a_tiles[2]}
    A4 = {layout_to_str(a_tiles[3].layout): a_tiles[3]}
    B1 = {layout_to_str(b_tiles[0].layout): b_tiles[0]}
    B2 = {layout_to_str(b_tiles[1].layout): b_tiles[1]}
    B3 = {layout_to_str(b_tiles[2].layout): b_tiles[2]}
    B4 = {layout_to_str(b_tiles[3].layout): b_tiles[3]}

    # T1 = A1 + A4
    T1 = add_tiles(kernel_map, A1, A4, roll_flag, network)

    # T2 = B1 + B4
    T2 = add_tiles(kernel_map, B1, B4, roll_flag, network)
    
    # M1 = T1 @ T2
    M1 = matmul_tiles(kernel_map, T1, T2, roll_flag, network)

    # T3 = A3 + A4
    T3 = add_tiles(kernel_map, A3, A4, roll_flag, network)

    # M2 = T3 @ B1
    M2 = matmul_tiles(kernel_map, T3, B1, roll_flag, network)

    # T4 = B2 - B4
    T4 = sub_tiles(kernel_map, B2, B4, roll_flag, network)

    # M3 = A1 @ T4
    M3 = matmul_tiles(kernel_map, A1, T4, roll_flag, network)

    # T5 = B3 - B1
    T5 = sub_tiles(kernel_map, B3, B1, roll_flag, network)

    # M4 = A4 @ T5
    M4 = matmul_tiles(kernel_map, A4, T5, roll_flag, network)

    # T6 = A1 + A2
    T6 = add_tiles(kernel_map, A1, A2, roll_flag, network)

    # M5 = T6 @ B4
    M5 = matmul_tiles(kernel_map, T6, B4, roll_flag, network)

    # T7 = A3 - A1
    T7 = sub_tiles(kernel_map, A3, A1, roll_flag, network)
    # T8 = B1 + B2
    T8 = add_tiles(kernel_map, B1, B2, roll_flag, network)
    # M6 = T7 @ T8
    M6 = matmul_tiles(kernel_map, T7, T8, roll_flag, network)

    # T9 = A2 - A4
    T9 = sub_tiles(kernel_map, A2, A4, roll_flag, network)
    # T10 = B3 + B4
    T10 = add_tiles(kernel_map, B3, B4, roll_flag, network)
    # M7 = T9 @ T10
    M7 = matmul_tiles(kernel_map, T9, T10, roll_flag, network)

    C1 = add_tiles(
        kernel_map,
        sub_tiles(
            kernel_map,
            add_tiles(kernel_map, M1, M4, roll_flag, network),
            M5,
            roll_flag,
            network,
        ),
        M7,
        roll_flag,
        network,
    )
    C2 = add_tiles(kernel_map, M3, M5, roll_flag, network)
    C3 = add_tiles(kernel_map, M2, M4, roll_flag, network)
    C4 = add_tiles(
        kernel_map,
        add_tiles(
            kernel_map,
            sub_tiles(kernel_map, M1, M2, roll_flag, network),
            M3,
            roll_flag,
            network,
        ),
        M6,
        roll_flag,
        network,
    )

    output_kernels = []
    for layout in C1.keys():
        c1_kernel = C1[layout]
        c2_kernel = C2[layout]
        c3_kernel = C3[layout]
        c4_kernel = C4[layout]
        combine_kernel = create_combine_kernel(
            term, kernel_map, c1_kernel, c2_kernel, c3_kernel, c4_kernel
        )
        output_kernels.append(combine_kernel)
    return output_kernels


def gen_strassens(term, cs_kernels, roll_flag, network):
    """
    Generate Strassen's algorithm kernels for matrix multiplication.
    
    Note: Input kernels must have no rolls (for tiling), but internal matmuls
    will use roll-based matmul when roll_flag is True for better performance.
    """
    kernel_map = {}
    a_kernels_no_rolls = [k for k in cs_kernels[0] if not k.layout.rolls]
    b_kernels_no_rolls = [k for k in cs_kernels[1] if not k.layout.rolls]
    
    for a_kernel in a_kernels_no_rolls:
        for b_kernel in b_kernels_no_rolls:
            if check_strassens(a_kernel, b_kernel):
                # create cs_placeholders
                a_cs_placeholder = Kernel(KernelOp.CS, [0], a_kernel.layout)
                b_cs_placeholder = Kernel(KernelOp.CS, [0], b_kernel.layout)
                if a_kernel.layout.num_ct() == 4:
                    # assumption, a_kernel and b_kernel have their ct-dims row-major packed
                    # create indexed tiles
                    a_ct_offsets = traverse_ct_dims(a_kernel.layout.ct_dims)
                    b_ct_offsets = traverse_ct_dims(b_kernel.layout.ct_dims)

                    # create offsets
                    indexed_a_kernels = []
                    for i, offset in enumerate(a_ct_offsets):
                        # create indexed term:
                        indexed_term = copy(a_kernel.layout.term)
                        indexed_term = indexed_term[
                            offset[0] : offset[1], offset[2] : offset[3]
                        ]
                        indexed_a_layout = Layout(
                            indexed_term,
                            a_kernel.layout.rolls,
                            a_kernel.layout.slot_dims,
                            a_kernel.layout.n,
                            a_kernel.layout.secret,
                        )
                        indexed_a_kernel = Kernel(
                            KernelOp.SELECT,
                            [a_cs_placeholder, i],
                            indexed_a_layout,
                        )
                        indexed_a_kernels.append(indexed_a_kernel)

                        if indexed_term not in kernel_map:
                            kernel_map[indexed_term] = []
                        kernel_map[indexed_term].append(indexed_a_kernel)

                    indexed_b_kernels = []
                    for i, offset in enumerate(b_ct_offsets):
                        # create indexed term:
                        indexed_term = copy(b_kernel.layout.term)
                        indexed_term = indexed_term[
                            offset[0] : offset[1], offset[2] : offset[3]
                        ]
                        indexed_b_layout = Layout(
                            indexed_term,
                            b_kernel.layout.rolls,
                            b_kernel.layout.slot_dims,
                            b_kernel.layout.n,
                            b_kernel.layout.secret,
                        )
                        indexed_b_kernel = Kernel(
                            KernelOp.SELECT,
                            [b_cs_placeholder, i],
                            indexed_b_layout,
                        )
                        indexed_b_kernels.append(indexed_b_kernel)
                        if indexed_term not in kernel_map:
                            kernel_map[indexed_term] = []
                        kernel_map[indexed_term].append(indexed_b_kernel)

                    strassens_opt(
                        term,
                        kernel_map,
                        indexed_a_kernels,
                        indexed_b_kernels,
                        roll_flag,
                        network,
                    )

    return kernel_map
