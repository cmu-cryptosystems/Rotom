"""
Optimization to lift rotations to the packing phase.

This implements the Viaduct e2_o1 strategy: instead of rotating ciphertexts
homomorphically, we can pre-rotate them during packing and send multiple CTs.

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
    
    def find_pack_base(term, depth=0, max_depth=15):
        """Find the base PACK term, following CS chains, operations, and replication.
        
        We can lift rotations even if they're applied after operations like MUL, ADD, etc.,
        or after replication, as long as we can trace back to a PACK operation.
        """
        if depth > max_depth or not hasattr(term, 'op'):
            return None
        
        if term.op == HEOp.PACK:
            return term
        elif term.op == HEOp.CS:
            # CS term points to a ciphertext - trace through it
            if len(term.cs) > 0:
                # The CS term's child is the actual ciphertext term
                # This could be a PACK, or it could be the result of replication
                # In replication, the CT might be rotated, so we need to trace through
                return find_pack_base(term.cs[0], depth + 1, max_depth)
        elif term.op == HEOp.ROT:
            # If we hit a rotation, trace through to find the base
            # This handles cases where replication created rotated CTs
            # Replication uses rotate_and_sum which creates chains: ROT(ROT(...ROT(PACK)...))
            if len(term.cs) > 0:
                return find_pack_base(term.cs[0], depth + 1, max_depth)
        elif term.op == HEOp.ADD:
            # For ADD operations (from rotate_and_sum in replication), trace through both operands
            # rotate_and_sum creates: ADD(ROT(base, rot_amt), base)
            # So one operand is ROT(base, ...) and the other is base
            # We want to find the base which should have the PACK
            if len(term.cs) >= 2:
                left = term.cs[0]
                right = term.cs[1]
                
                # Pattern: ADD(ROT(base, a), base) - the non-rotated operand is the base
                # Check if one is a rotation
                if hasattr(left, 'op') and left.op == HEOp.ROT:
                    # Left is rotation, right is likely the base
                    pack = find_pack_base(right, depth + 1, max_depth)
                    if pack is not None:
                        return pack
                    # Also check the base of the rotation
                    if len(left.cs) > 0:
                        pack = find_pack_base(left.cs[0], depth + 1, max_depth)
                        if pack is not None:
                            return pack
                elif hasattr(right, 'op') and right.op == HEOp.ROT:
                    # Right is rotation, left is likely the base
                    pack = find_pack_base(left, depth + 1, max_depth)
                    if pack is not None:
                        return pack
                    # Also check the base of the rotation
                    if len(right.cs) > 0:
                        pack = find_pack_base(right.cs[0], depth + 1, max_depth)
                        if pack is not None:
                            return pack
                
                # If neither is clearly a rotation, check both (might be nested)
                for child in term.cs:
                    if hasattr(child, 'op'):
                        pack = find_pack_base(child, depth + 1, max_depth)
                        if pack is not None:
                            return pack
        elif term.op in [HEOp.MUL, HEOp.SUB]:
            # Check if one of the operands is a PACK (for plaintext operations)
            for child in term.cs:
                if hasattr(child, 'op'):
                    pack = find_pack_base(child, depth + 1, max_depth)
                    if pack is not None:
                        return pack
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
                rotations_to_lift.append({
                    'rot_term': term,
                    'base': base,
                    'pack_base': pack_base,
                    'rot_amt': rot_amt,
                })
    
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
                    # Only lift filter alignment rotations (small rotations)
                    # Don't lift channel summation rotations (large rotations, |rot_amt| >= 100)
                    # Channel summation rotations are part of rotate-and-sum and should stay homomorphic
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
                                actual_base.metadata if hasattr(actual_base, 'metadata') else "",
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
                        kernel_info = " ".join(metadata_parts[1:]) if len(metadata_parts) > 1 else ""
                        
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
                                actual_base.metadata if hasattr(actual_base, 'metadata') else "",
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
