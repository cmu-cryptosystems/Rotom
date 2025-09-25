from ir.he import HEOp


def rot_zero_opt(he_term):
    """
    Rotations by 0 can be removed
    """
    update_map = {}
    for term in he_term.post_order():
        # update term cs
        for i, cs in enumerate(term.cs):
            if term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
                continue
            if cs in update_map:
                term.cs[i] = update_map[cs]

        # update kernel with rewrites
        match term.op:
            case HEOp.ROT:
                if term.cs[1] == 0:
                    update_map[term] = term.cs[0]
                else:
                    update_map[term] = term
            case _:
                update_map[term] = term
    # update term cs
    for i, cs in enumerate(he_term.cs):
        if he_term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
            continue
        if cs in update_map:
            he_term.cs[i] = update_map[cs]
    return update_map[he_term]


def join_rot(he_term):
    update_map = {}
    for term in he_term.post_order():
        # update term cs
        for i, cs in enumerate(term.cs):
            if term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
                continue
            if cs in update_map:
                term.cs[i] = update_map[cs]

        # update kernel with rewrites
        match term.op:
            case HEOp.ROT:
                if term.cs[0].op == HEOp.ROT:
                    rot_amt = term.cs[1] + term.cs[0].cs[1]
                    rot_term = term.cs[0].cs[0] << rot_amt
                    update_map[term] = rot_term
                else:
                    update_map[term] = term
            case _:
                update_map[term] = term
    # update term cs
    for i, cs in enumerate(he_term.cs):
        if he_term.op in [HEOp.PACK, HEOp.MASK, HEOp.ZERO_MASK]:
            continue
        if cs in update_map:
            he_term.cs[i] = update_map[cs]
    return update_map[he_term]
