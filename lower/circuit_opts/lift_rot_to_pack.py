"""
Optimization to lift rotations to the packing phase.

Instead of rotating ciphertexts homomorphically, we can pre-rotate them
during packing and send multiple CTs.

Pattern: ROT(CS(PACK(layout, metadata)), rot_amt)
  -> CS(PACK(layout, metadata + " rot:" + str(rot_amt)))

This allows the backend to rotate during packing (cheaper) instead of
homomorphically (more expensive).
"""

from collections import defaultdict

from ir.he import HEOp, HETerm


def lift_rotations_to_pack(he_term):
    """
    Lift rotations to the packing phase by creating pre-rotated pack operations.

    This optimization finds patterns like:
      ROT(CS(PACK(layout, metadata)), rot_amt)

    And replaces them with:
      CS(PACK(layout, metadata + " rot:" + str(rot_amt)))

    The backend can then handle these pre-rotated packs by rotating during
    packing instead of homomorphically.

    Args:
        he_term: The HE term to optimize

    Returns:
        Optimized HE term with rotations lifted to packing
    """
    update_map = {}
    rotation_cache = {}  # (pack_term, rot_amt) -> new_pre_rotated_pack

    def find_pack_base(term, depth=0, max_depth=10):
        """Find the base PACK term, following only CS and ROT chains.

        This is conservative: we only lift rotations that are directly on CS(PACK(...))
        or ROT(CS(PACK(...))) patterns. We do NOT trace through ADD, MUL, SUB operations
        to avoid incorrectly lifting rotations in non-conv operations like matmul.

        The pattern we're looking for is:
        - ROT(CS(PACK(...)), rot_amt) -> can lift
        - ROT(ROT(CS(PACK(...)), ...), rot_amt) -> can lift (after replication)
        """
        if depth > max_depth or not hasattr(term, "op"):
            return None

        if term.op == HEOp.PACK:
            return term
        elif term.op == HEOp.CS:
            # CS term points to a ciphertext - trace through it
            if len(term.cs) > 0:
                return find_pack_base(term.cs[0], depth + 1, max_depth)
        elif term.op == HEOp.ROT:
            # If we hit a rotation, trace through to find the base
            # This handles cases where replication created rotated CTs
            # Replication uses rotate_and_sum which creates chains: ROT(ROT(...ROT(PACK)...))
            if len(term.cs) > 0:
                return find_pack_base(term.cs[0], depth + 1, max_depth)

        # Do NOT trace through ADD, MUL, SUB - these are operations that should not
        # have their rotations lifted (e.g., matmul uses MUL operations that shouldn't be lifted)
        return None

    # First pass: identify rotations that can be lifted
    rotations_to_lift = []
    for term in he_term.post_order():
        if term.op == HEOp.ROT:
            base = term.cs[0]
            rot_amt = term.cs[1]

            # Check if base is CS(PACK(...)) or just PACK(...)
            pack_base = find_pack_base(base)

            if pack_base is not None:
                # This rotation can be lifted to packing
                rotations_to_lift.append(
                    {
                        "rot_term": term,
                        "base": base,
                        "pack_base": pack_base,
                        "rot_amt": rot_amt,
                    }
                )

    # Second pass: create pre-rotated pack operations and update terms
    for term in he_term.post_order():
        # Update term children first
        for i, cs in enumerate(term.cs):
            if term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
                continue
            if cs in update_map:
                term.cs[i] = update_map[cs]

        match term.op:
            case HEOp.ROT:
                base = term.cs[0]
                rot_amt = term.cs[1]

                # Get actual base after updates
                actual_base = update_map.get(base, base)
                pack_base = find_pack_base(actual_base)

                if pack_base is not None:
                    # Only lift filter alignment rotations (small rotations, typically |rot_amt| < 100)
                    # Don't lift channel summation rotations (large rotations, |rot_amt| >= 100)
                    # Channel summation rotations are part of rotate-and-sum and should stay homomorphic
                    # This optimization is primarily for convolution filter alignment
                    # Be conservative: only lift small rotations that are clearly for filter alignment
                    # Filter alignment rotations in conv are typically small (e.g., -33, -32, -31, etc.)
                    if abs(rot_amt) >= 100:
                        # This is a channel summation rotation, don't lift it
                        update_map[term] = term
                        continue

                    # Check if we've already created a pre-rotated pack for this
                    cache_key = (pack_base, rot_amt)
                    if cache_key in rotation_cache:
                        # Use existing pre-rotated pack
                        pre_rotated_pack = rotation_cache[cache_key]

                        # If the base was CS(PACK(...)), create CS wrapper
                        if actual_base.op == HEOp.CS:
                            update_map[term] = HETerm(
                                HEOp.CS,
                                [pre_rotated_pack],
                                pre_rotated_pack.secret,
                                (
                                    actual_base.metadata
                                    if hasattr(actual_base, "metadata")
                                    else ""
                                ),
                            )
                        else:
                            update_map[term] = pre_rotated_pack
                    else:
                        # Create new pre-rotated pack
                        # Extract layout and metadata from pack_base
                        layout = pack_base.cs[0]
                        old_metadata = pack_base.metadata if pack_base.metadata else ""

                        # Parse existing metadata to get packing index
                        # Format is typically "packing_idx kernel_info" or just "packing_idx"
                        metadata_parts = old_metadata.split()
                        packing_idx = metadata_parts[0] if metadata_parts else "0"
                        kernel_info = (
                            " ".join(metadata_parts[1:])
                            if len(metadata_parts) > 1
                            else ""
                        )

                        # Add rotation info to metadata
                        if kernel_info:
                            new_metadata = f"{packing_idx} {kernel_info} rot:{rot_amt}"
                        else:
                            new_metadata = f"{packing_idx} rot:{rot_amt}"

                        # Create new PACK with rotation metadata
                        pre_rotated_pack = HETerm(
                            HEOp.PACK,
                            [layout],
                            pack_base.secret,
                            new_metadata,
                        )
                        rotation_cache[cache_key] = pre_rotated_pack

                        # If the base was CS(PACK(...)), create CS wrapper
                        if actual_base.op == HEOp.CS:
                            update_map[term] = HETerm(
                                HEOp.CS,
                                [pre_rotated_pack],
                                pre_rotated_pack.secret,
                                (
                                    actual_base.metadata
                                    if hasattr(actual_base, "metadata")
                                    else ""
                                ),
                            )
                        else:
                            update_map[term] = pre_rotated_pack
                else:
                    # Cannot lift this rotation, keep as-is
                    update_map[term] = term
            case _:
                update_map[term] = term

    # Final update of root term's children
    for i, cs in enumerate(he_term.cs):
        if he_term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
            continue
        if cs in update_map:
            he_term.cs[i] = update_map[cs]

    return update_map.get(he_term, he_term)
