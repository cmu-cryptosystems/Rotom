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
        
        # Debug: Check add kernel costs
        add_with_rolls = [k for k in add_kernels if k.layout.rolls]
        add_without_rolls = [k for k in add_kernels if not k.layout.rolls]
        if add_with_rolls and add_without_rolls:
            costs_with_rolls = [KernelCost(k, network).total_cost() for k in add_with_rolls]
            costs_without_rolls = [KernelCost(k, network).total_cost() for k in add_without_rolls]
            min_cost_with_rolls = min(costs_with_rolls) if costs_with_rolls else float('inf')
            min_cost_without_rolls = min(costs_without_rolls) if costs_without_rolls else float('inf')
            if min_cost_with_rolls < min_cost_without_rolls:
                print(f"  ⚠ add_tiles: Rolled kernel cheaper ({min_cost_with_rolls} vs {min_cost_without_rolls}) for {term}")
        
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
        
        # Debug: Check sub kernel costs
        sub_with_rolls = [k for k in sub_kernels if k.layout.rolls]
        sub_without_rolls = [k for k in sub_kernels if not k.layout.rolls]
        if sub_with_rolls and sub_without_rolls:
            costs_with_rolls = [KernelCost(k, network).total_cost() for k in sub_with_rolls]
            costs_without_rolls = [KernelCost(k, network).total_cost() for k in sub_without_rolls]
            min_cost_with_rolls = min(costs_with_rolls) if costs_with_rolls else float('inf')
            min_cost_without_rolls = min(costs_without_rolls) if costs_without_rolls else float('inf')
            if min_cost_with_rolls < min_cost_without_rolls:
                print(f"  ⚠ sub_tiles: Rolled kernel cheaper ({min_cost_with_rolls} vs {min_cost_without_rolls}) for {term}")
        
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


def matmul_tiles(kernel_map, a_kernels, b_kernels, roll_flag, network):
    output_kernels = {}
    output_kernel_costs = {}
    for a_kernel in a_kernels.values():
        for b_kernel in b_kernels.values():
            a_kernel = copy(a_kernel)
            b_kernel = copy(b_kernel)
            term = a_kernel.layout.term @ b_kernel.layout.term
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
            
            # Debug: Detailed analysis of matmul kernels
            matmuls_with_rolls = [k for k in matmul_kernels if k.layout.rolls]
            matmuls_without_rolls = [k for k in matmul_kernels if not k.layout.rolls]
            print(f"\n  === matmul_tiles for {term} ===")
            print(f"  Total kernels: {len(matmul_kernels)}, With rolls: {len(matmuls_with_rolls)}, Without rolls: {len(matmuls_without_rolls)}")
            
            # Calculate costs for all kernels
            kernel_costs = []
            for k in matmul_kernels:
                cost = KernelCost(k, network).total_cost()
                has_rolls = bool(k.layout.rolls)
                kernel_costs.append((k, cost, has_rolls))
            
            # Sort by cost
            kernel_costs.sort(key=lambda x: x[1])
            
            print(f"  Top 5 kernels by cost:")
            for i, (k, cost, has_rolls) in enumerate(kernel_costs[:5]):
                print(f"    {i+1}. Cost: {cost}, Has rolls: {has_rolls}, Layout: {k.layout}")
            
            # Check if cheapest has rolls
            if kernel_costs:
                cheapest = kernel_costs[0]
                print(f"  Cheapest kernel: Cost={cheapest[1]}, Has rolls={cheapest[2]}")
            
            for matmul_kernel in matmul_kernels:
                layout = matmul_kernel.layout
                layout_str = layout_to_str(layout)
                cost = KernelCost(matmul_kernel, network).total_cost()
                if layout_str not in output_kernels:
                    output_kernels[layout_str] = matmul_kernel
                    output_kernel_costs[layout_str] = cost
                elif cost < output_kernel_costs[layout_str]:
                    output_kernels[layout_str] = matmul_kernel
                    output_kernel_costs[layout_str] = cost
            
            # When roll_flag is True, prefer rolled kernels over non-rolled ones
            # Create a layout key ignoring rolls to find duplicates
            if roll_flag:
                # Group kernels by layout ignoring rolls
                layout_groups = {}
                for layout_str, kernel in output_kernels.items():
                    # Create a key that ignores rolls - use just the dimensions
                    dims_key = str(kernel.layout.get_dims())
                    if dims_key not in layout_groups:
                        layout_groups[dims_key] = []
                    layout_groups[dims_key].append((layout_str, kernel, output_kernel_costs[layout_str]))
                
                # For each group, if there are both rolled and non-rolled, prefer rolled
                filtered_output_kernels = {}
                filtered_output_kernel_costs = {}
                for dims_key, kernels in layout_groups.items():
                    rolled_kernels = [(ls, k, c) for ls, k, c in kernels if k.layout.rolls]
                    non_rolled_kernels = [(ls, k, c) for ls, k, c in kernels if not k.layout.rolls]
                    
                    if rolled_kernels and non_rolled_kernels:
                        # Prefer rolled kernels - keep the cheapest rolled one
                        rolled_kernels.sort(key=lambda x: x[2])  # Sort by cost
                        best_rolled = rolled_kernels[0]
                        filtered_output_kernels[best_rolled[0]] = best_rolled[1]
                        filtered_output_kernel_costs[best_rolled[0]] = best_rolled[2]
                        print(f"  ✓ Preferring rolled kernel (cost={best_rolled[2]}) over non-rolled for dims {dims_key[:60]}")
                    else:
                        # No choice - keep all
                        for ls, k, c in kernels:
                            filtered_output_kernels[ls] = k
                            filtered_output_kernel_costs[ls] = c
                
                output_kernels = filtered_output_kernels
                output_kernel_costs = filtered_output_kernel_costs
            
            # Debug: Which kernel was selected for output
            selected_layouts = list(output_kernels.keys())
            print(f"  Selected {len(selected_layouts)} unique layouts for output")
            for layout_str in selected_layouts[:3]:  # Show first 3
                k = output_kernels[layout_str]
                cost = output_kernel_costs[layout_str]
                print(f"    Selected: Cost={cost}, Has rolls={bool(k.layout.rolls)}, Layout={layout_str[:80]}")
            print()
    return output_kernels


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
        # Debug: Check C kernel layouts
        c1_kernel = C1[layout]
        c2_kernel = C2[layout]
        c3_kernel = C3[layout]
        c4_kernel = C4[layout]
        print(f"\n=== C kernels for layout {layout} ===")
        print(f"C1 has rolls: {bool(c1_kernel.layout.rolls)}, rolls: {c1_kernel.layout.rolls}")
        print(f"C2 has rolls: {bool(c2_kernel.layout.rolls)}, rolls: {c2_kernel.layout.rolls}")
        print(f"C3 has rolls: {bool(c3_kernel.layout.rolls)}, rolls: {c3_kernel.layout.rolls}")
        print(f"C4 has rolls: {bool(c4_kernel.layout.rolls)}, rolls: {c4_kernel.layout.rolls}")
        
        combine_kernel = create_combine_kernel(
            term, kernel_map, c1_kernel, c2_kernel, c3_kernel, c4_kernel
        )
        # Calculate combine kernel's own cost
        combine_own_cost = KernelCost(combine_kernel, network).total_cost()
        
        # Calculate child (CS) kernel costs - these reference the C kernels
        from util.kernel_util import get_cs_op_kernels
        from ir.layout_utils import dimension_merging
        cs_kernels = get_cs_op_kernels(combine_kernel)
        cs_costs = 0
        for cs_kernel in cs_kernels:
            # CS kernels reference terms in kernel_map, get their costs
            cs_term = cs_kernel.layout.term
            if cs_term in kernel_map:
                # Get the cost of the C kernel from kernel_map
                # We need to find the cost of the C kernel layout
                c_kernel_layout = cs_kernel.layout
                # The C kernel cost should be in kernel_map, but we need to calculate it
                # For now, just get the direct cost
                cs_cost = KernelCost(cs_kernel, network).total_cost()
                cs_costs += cs_cost
            else:
                cs_cost = KernelCost(cs_kernel, network).total_cost()
                cs_costs += cs_cost
        
        # Also calculate costs of C1, C2, C3, C4 kernels themselves
        c1_cost = KernelCost(c1_kernel, network).total_cost()
        c2_cost = KernelCost(c2_kernel, network).total_cost()
        c3_cost = KernelCost(c3_kernel, network).total_cost()
        c4_cost = KernelCost(c4_kernel, network).total_cost()
        total_c_costs = c1_cost + c2_cost + c3_cost + c4_cost
        
        total_cost = combine_own_cost + cs_costs
        
        print(f"Combine kernel has rolls: {bool(combine_kernel.layout.rolls)}, rolls: {combine_kernel.layout.rolls}")
        print(f"Combine kernel own cost: {combine_own_cost}")
        print(f"C1 cost: {c1_cost}, C2 cost: {c2_cost}, C3 cost: {c3_cost}, C4 cost: {c4_cost}")
        print(f"Total C kernels cost: {total_c_costs}")
        print(f"Child (CS) kernels cost: {cs_costs}")
        print(f"Total cost (combine + CS): {total_cost}")
        print(f"Estimated total (combine + C kernels): {combine_own_cost + total_c_costs}")
        print(f"Layout: {combine_kernel.layout}")
        print("=" * 60)
        output_kernels.append(combine_kernel)
    return output_kernels


def gen_strassens(term, cs_kernels, roll_flag, network):
    """
    Generate Strassen's algorithm kernels for matrix multiplication.
    
    Note: Input kernels must have no rolls (for tiling), but internal matmuls
    will use roll-based matmul when roll_flag is True for better performance.
    """
    kernel_map = {}
    # Debug: Print input kernels
    print(f"\n=== gen_strassens for {term} ===")
    print(f"roll_flag: {roll_flag}")
    print(f"Total a_kernels: {len(cs_kernels[0])}")
    print(f"Total b_kernels: {len(cs_kernels[1])}")
    a_kernels_with_rolls = [k for k in cs_kernels[0] if k.layout.rolls]
    b_kernels_with_rolls = [k for k in cs_kernels[1] if k.layout.rolls]
    print(f"a_kernels with rolls: {len(a_kernels_with_rolls)}")
    print(f"b_kernels with rolls: {len(b_kernels_with_rolls)}")
    
    # Filter to only kernels without rolls, as strassens requires no-roll inputs for tiling
    a_kernels_no_rolls = [k for k in cs_kernels[0] if not k.layout.rolls]
    b_kernels_no_rolls = [k for k in cs_kernels[1] if not k.layout.rolls]
    print(f"a_kernels without rolls: {len(a_kernels_no_rolls)}")
    print(f"b_kernels without rolls: {len(b_kernels_no_rolls)}")
    
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
