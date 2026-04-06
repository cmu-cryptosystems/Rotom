"""Fused circuit optimizations for lowered HE terms.

``Lower.opt`` historically ran four separate ``post_order`` traversals per
ciphertext root (``rot_zero_opt``, ``zero_mask_opt``, ``mask_identity_opt``,
``zero_mask_identity_opt``).  Each pass walks the full DAG and patches child
references from an ``update_map``.  A single fused pass applies the same rules
in the original order with one traversal, which scales better for large
convolution circuits.
"""

from __future__ import annotations

from ir.he import HEOp


def _substitute_children(term, update_map: dict) -> None:
    for i, cs in enumerate(term.cs):
        if term.op in (HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK):
            continue
        if isinstance(cs, dict):
            continue
        if cs in update_map:
            term.cs[i] = update_map[cs]


def _rewrite_root(he_term, update_map: dict) -> None:
    for i, cs in enumerate(he_term.cs):
        if he_term.op in (HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK):
            continue
        if isinstance(cs, dict):
            continue
        if cs in update_map:
            he_term.cs[i] = update_map[cs]


def fused_circuit_opts(he_term):
    """Apply rot/zero-mask/mask-identity/zero-mask-identity rules in one ``post_order``.

    Semantics match running, in order:
    ``rot_zero_opt`` → ``zero_mask_opt`` → ``mask_identity_opt`` →
    ``zero_mask_identity_opt``.
    """
    update_map: dict = {}
    for term in he_term.post_order():
        _substitute_children(term, update_map)

        # 1) rot_zero_opt
        if term.op == HEOp.ROT and term.cs[1] == 0:
            update_map[term] = term.cs[0]
            continue

        # 2) zero_mask_opt
        if term.op == HEOp.MASK:
            if not any(term.cs[0]):
                term.op = HEOp.ZERO_MASK
            update_map[term] = term
            continue

        # 3) mask_identity_opt
        if term.op == HEOp.MUL:
            if term.cs[0].op == HEOp.MASK:
                if all(term.cs[0].cs[0]):
                    update_map[term] = term.cs[1]
                else:
                    update_map[term] = term
            elif term.cs[1].op == HEOp.MASK:
                if all(term.cs[1].cs[0]):
                    update_map[term] = term.cs[0]
                else:
                    update_map[term] = term
            else:
                update_map[term] = term
            continue

        # 4) zero_mask_identity_opt
        match term.op:
            case HEOp.ADD:
                if term.cs[0].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[1]
                elif term.cs[1].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[0]
                else:
                    update_map[term] = term
            case HEOp.MUL:
                if term.cs[0].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[0]
                elif term.cs[1].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[1]
                else:
                    update_map[term] = term
            case HEOp.ROT:
                if term.cs[0].op == HEOp.ZERO_MASK:
                    update_map[term] = term.cs[0]
                else:
                    update_map[term] = term
            case _:
                update_map[term] = term

    _rewrite_root(he_term, update_map)
    return update_map[he_term]
